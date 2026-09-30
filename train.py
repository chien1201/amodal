import os
import glob
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from efficient_sam.build_efficient_sam import build_efficient_sam_vits

# =========================================================================
# 1. 損失函數定義 (SAMEO Loss Formulation)
# =========================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, targets):
        pred_probs = torch.sigmoid(pred_logits)    # 轉換為 0~1 機率值 (M_hat)
        pred_flat = pred_probs.view(-1)
        targets_flat = targets.view(-1)            # M_gt
        intersection = (pred_flat * targets_flat).sum()      # |M_hat ∩ M_gt|
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')     # -log(p_t)
        pred_probs = torch.sigmoid(pred_logits)
        # 計算 p_t: 正樣本取 pred_probs，負樣本取 1 - pred_probs
        p_t = targets * pred_probs + (1.0 - targets) * (1.0 - pred_probs)
        # -(1 - p_t)^gamma * log(p_t)
        loss = self.alpha * ((1.0 - p_t) ** self.gamma) * bce_loss
        return loss.mean()

def calculate_iou_loss(pred_iou, pred_mask_logits, targets):
    # 1. 若 pred_iou 是 [B, 3] 或 [B, 1, 3]，取出與主 Mask 對應的第 0 個預測分數
    if pred_iou.dim() >= 2 and pred_iou.shape[-1] == 3:
        pred_iou = pred_iou[..., 0]  # 取 index 0，形狀變為 [B]

    # 2. 計算模型預測之遮罩與真實 GT 的實際 IoU
    pred_probs = (torch.sigmoid(pred_mask_logits) > 0.5).float()
    intersection = (pred_probs * targets).sum(dim=(-1, -2))
    union = (
        pred_probs.sum(dim=(-1, -2)) + targets.sum(dim=(-1, -2)) - intersection
    )   
    real_iou = torch.clamp(intersection / (union + 1e-6), 0.0, 1.0)

    # 確保兩者展平為一維向量 [B] 後計算均方誤差
    return F.mse_loss(pred_iou.view(-1), real_iou.view(-1))


# =========================================================================
# 2. 支援 2x2 消融的資料集載入器 (Dataset)
# =========================================================================
class AblationAmodalDataset(Dataset):
    def __init__(
            self, 
            data_root="data/output_samples", 
            edge_mode="taos", 
            dual_mode=False, 
            img_size=1024):
        """
        :param edge_mode: 'hard' 或 'taos'
        :param dual_mode: True (50% 原圖 + 50% 遮擋) 或 False (100% 遮擋)
        """
        # 正規化路徑
        data_root = os.path.normpath(data_root)

        #依據資料夾結尾名稱 (_hard 或 _taos) 精準分流載入
        all_dirs = sorted(glob.glob(os.path.join(data_root, "sample_*")))

        # 加上除錯資訊，讓你在終端機一眼看出抓到了什麼
        print(f"DEBUG: 正在讀取資料目錄 -> {os.path.abspath(data_root)}")
        print(f"DEBUG: 該目錄下搜尋到的 sample_* 總數: {len(all_dirs)}")

        # 2. 嚴謹比對資料夾名稱 (使用 os.path.basename 避免結尾斜線影響)
        target_suffix = f"_{edge_mode.lower()}"
        self.sample_dirs = [
            p for p in all_dirs if os.path.basename(p).endswith(target_suffix)
        ]

        self.dual_mode = dual_mode
        self.img_size = img_size

        print(
            f"[{edge_mode.upper()} 模式] 成功載入 {len(self.sample_dirs)}"
            " 組樣本資料夾！"
        )

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_path = self.sample_dirs[idx]

        # 讀取標準檔案
        img_bgr = cv2.imread(os.path.join(sample_path, "image.jpg"))
        mask_gt = cv2.imread(
            os.path.join(sample_path, "amodal_mask.png"), cv2.IMREAD_GRAYSCALE
        )

        with open(os.path.join(sample_path, "box_prompt.json"), "r") as f:
            box_data = json.load(f)

        # SAMEO 提示詞隨機挑選機制：50% 可見框, 50% 完整框
        chosen_box_key = "modal_box" if np.random.rand() > 0.5 else "amodal_box"
        chosen_box = box_data[chosen_box_key]

        h_orig, w_orig = img_bgr.shape[:2]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        mask_resized = cv2.resize(mask_gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 1. 計算縮放比例 (Scale Factors)
        scale_x = self.img_size / float(w_orig)
        scale_y = self.img_size / float(h_orig)
        scaled_box = [
            chosen_box[0] * scale_x, chosen_box[1] * scale_y,
            chosen_box[2] * scale_x, chosen_box[3] * scale_y
        ]

        # 2. 影像前處理與維度轉換 (Channel First HWC to CHW & Normalization)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # 3. 遮罩轉二值張量 (Binary Mask Tensor)
        mask_tensor = torch.from_numpy((mask_resized > 128).astype(np.float32)).unsqueeze(0)

        # 4. 建立 EfficientSAM 專屬的提示座標與標籤 (Box Prompt Encoding)
        coords = torch.tensor([[scaled_box[0], scaled_box[1]], [scaled_box[2], scaled_box[3]]]).float()
        labels = torch.tensor([2, 3]).long()

        return img_tensor, mask_tensor, coords, labels

# =========================================================================
# 3. 模型建立 (凍結 Encoder，僅訓練 Decoder)
# =========================================================================
def build_sameo_model(weights_path="weights/efficient_sam_vits.pt", device="cuda"):
    model = build_efficient_sam_vits()
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    # 凍結 Image/Prompt Encoder
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = True

    model.to(device)
    return model

# =========================================================================
# 4. 主訓練迴圈
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="SAMEO 純遮擋消融實驗微調訓練 (Pure Occlusion)")
    parser.add_argument("--edge_mode", type=str, choices=["hard", "taos"], required=True, help="邊界模式: hard 或 taos")
    parser.add_argument("--exp_name", type=str, required=True, help="實驗標籤名稱 (如: exp1_hard_pure)")
    parser.add_argument("--data_root", type=str, default="data/output_samples", help="合成資料目錄")
    parser.add_argument("--epochs", type=int, default=10, help="訓練輪數")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch 大小 (預設 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Mask Decoder 學習率")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"啟動訓練任務: {args.exp_name}")
    print(f"模式設定: 邊界={args.edge_mode} | 遮擋比例=100% (無 Dual) | 設備={device}")
    print("=" * 60)

    # 載入純遮擋資料集
    dataset = AblationAmodalDataset(
        data_root=args.data_root,
        edge_mode=args.edge_mode,
        dual_mode=False,
        img_size=1024
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 載入並凍結權重
    model = build_sameo_model(weights_path="weights/efficient_sam_vits.pt", device=device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    dice_loss_fn = DiceLoss()
    focal_loss_fn = FocalLoss()
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch_idx, (imgs, masks, coords, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            coords = coords.unsqueeze(1).to(device)  # 維度符合: [B, 1, 2, 2]
            labels = labels.unsqueeze(1).to(device)  # 維度符合: [B, 1, 2]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                # EfficientSAM 前向推論
                pred_masks, pred_ious = model(imgs, coords, labels)

                # 若輸出多個候選遮罩，選取置信度最高的第一個遮罩
                if pred_masks.dim() == 5:
                    pred_masks = pred_masks[:, :, 0, :, :]
                if pred_masks.shape[-2:] != masks.shape[-2:]:
                    pred_masks = F.interpolate(pred_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False)

                # 損失計算 (Dice + Focal + 0.05 * IoU Loss)
                loss_dice = dice_loss_fn(pred_masks, masks)
                loss_focal = focal_loss_fn(pred_masks, masks)
                loss_iou = calculate_iou_loss(pred_ious, pred_masks, masks)

                loss = loss_dice + loss_focal + 0.05 * loss_iou

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{args.epochs}] - Average Loss: {avg_loss:.4f}")

    # 儲存 Decoder 權重
    os.makedirs("weights", exist_ok=True)
    save_path = f"weights/decoder_{args.exp_name}.pth"
    torch.save(model.mask_decoder.state_dict(), save_path)
    print(f"[{args.exp_name}] 訓練完成！Decoder 權重已儲存至：{save_path}\n")

if __name__ == "__main__":
    main()