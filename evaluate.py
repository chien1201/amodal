# 比較metrics之計算與對比

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import cv2
import json
import numpy as np

# =========================================================================
# 1. 自定義 Dataset：讀取我們剛剛產生的合成資料
# =========================================================================
class AmodalDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []
        # 自動掃描 output_samples 底下的所有資料夾
        for sample_name in os.listdir(data_root):
            sample_dir = os.path.join(data_root, sample_name)
            if os.path.isdir(sample_dir):
                img_path = os.path.join(sample_dir, "image.jpg")
                mask_path = os.path.join(sample_dir, "amodal_mask.png")
                json_path = os.path.join(sample_dir, "box_prompt.json")
                if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(json_path):
                    self.samples.append((img_path, mask_path, json_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, json_path = self.samples[idx]
        
        # 讀取圖片與 Amodal Ground Truth Mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256)) # 為了訓練方便可統一解析度
        
        amodal_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        amodal_mask = cv2.resize(amodal_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        amodal_mask = (amodal_mask > 127).astype(np.float32)

        with open(json_path, 'r') as f:
            prompts = json.load(f)
            modal_box = np.array(prompts['modal_box'], dtype=np.float32)

        # 轉換為 PyTorch Tensor
        image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.tensor(amodal_mask).unsqueeze(0).float()
        box_tensor = torch.tensor(modal_box).float()

        return image_tensor, box_tensor, mask_tensor

# =========================================================================
# 2. 組合損失函數 (Dice Loss + Focal Loss + IoU Loss)
# =========================================================================
class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_mask, target_mask):
        # 簡化的組合損失範例
        bce_loss = self.bce(pred_mask, target_mask)
        
        # Dice Loss
        pred_sig = torch.sigmoid(pred_mask)
        intersection = (pred_sig * target_mask).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (pred_sig.sum() + target_mask.sum() + 1e-5)
        
        return bce_loss + dice_loss

# =========================================================================
# 3. 主訓練流程
# =========================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device} 進行訓練...")

    # 載入基底模型 (此處以 vit_b 代替 Efficient SAM 架構演示)
    weights_path = "weights/sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=weights_path)
    sam.to(device)

    # 核心策略：凍結 Encoder，僅微調 Decoder
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
        
    # 確保 Mask Decoder 保持可訓練
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    # 準備資料集
    dataset = AmodalDataset("data/output_samples")
    if len(dataset) == 0:
        print("錯誤：找不到訓練資料，請先執行資料生成腳本！")
        return
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sam.parameters()), lr=1e-4)
    criterion = CompositeLoss()

    sam.train()
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        for images, boxes, gt_masks in dataloader:
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            
            optimizer.zero_grad()
            
            # 提取特徵並預測
            image_embeddings = sam.image_encoder(images)
            
            # 這裡簡化呼叫流程，實際串接時需配合 SAM Prompt 格式
            # 預測輸出 logits...
            # loss = criterion(pred_logits, gt_masks)
            
            # loss.backward()
            # optimizer.step()
            # total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] 訓練中... (示範架構已就緒)")

if __name__ == "__main__":
    main()