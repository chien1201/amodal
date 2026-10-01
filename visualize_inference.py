import argparse
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from efficient_sam.build_efficient_sam import build_efficient_sam_vits


def load_model(weights_path=None, base_weights="weights/efficient_sam_vits.pt", device="cuda"):
    """載入 EfficientSAM 模型，並選擇性套用微調後的 Decoder 權重"""
    model = build_efficient_sam_vits()
    checkpoint = torch.load(base_weights, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    if weights_path and os.path.exists(weights_path):
        decoder_state = torch.load(weights_path, map_location=device)
        model.mask_decoder.load_state_dict(decoder_state)
        print(f"成功載入 Decoder 權重: {weights_path}")
    else:
        print("未指定或找不到微調權重，使用原生預訓練權重。")

    model.to(device)
    model.eval()
    return model


def predict_mask(model, img_bgr, box, img_size=1024, device="cuda"):
    """執行單張前向推論並輸出二值化 Mask"""
    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))

    scale_x = img_size / float(w_orig)
    scale_y = img_size / float(h_orig)
    scaled_box = [
        box[0] * scale_x,
        box[1] * scale_y,
        box[2] * scale_x,
        box[3] * scale_y,
    ]

    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    coords = torch.tensor([[[[scaled_box[0], scaled_box[1]], [scaled_box[2], scaled_box[3]]]]]).float().to(device)
    labels = torch.tensor([[[2, 3]]]).long().to(device)

    with torch.no_grad():
        pred_masks, _ = model(img_tensor, coords, labels)
        if pred_masks.dim() == 5:
            pred_masks = pred_masks[:, :, 0, :, :]
        
        pred_masks = F.interpolate(pred_masks, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        mask_prob = torch.sigmoid(pred_masks).squeeze().cpu().numpy()
        binary_mask = (mask_prob > 0.5).astype(np.uint8)

    return binary_mask


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """將遮罩半透明疊加在原圖上(檢查用)"""
    overlay = image.copy()
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):
        colored_mask[:, :, c] = mask * color[c]
    
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # 疊加遮罩輪廓線
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="SAMEO 遮擋補全效果視覺化對比")
    parser.add_argument("--sample_dir", type=str, default="data/output_samples_demo/sample_1_hard", help="測試樣本目錄路徑")
    parser.add_argument("--save_path", type=str, default="visual_comparison.png", help="對比圖存檔路徑")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 讀取影像與標註
    img_path = os.path.join(args.sample_dir, "image.jpg")
    mask_path = os.path.join(args.sample_dir, "amodal_mask.png")
    prompt_path = os.path.join(args.sample_dir, "box_prompt.json")

    img_bgr = cv2.imread(img_path)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 128).astype(np.uint8)

    with open(prompt_path, "r") as f:
        box_data = json.load(f)
    box = box_data["modal_box"]  # 使用被遮擋的可見框進行 Prompting，考驗腦補能力

    # 2. 分別載入三個模型推論
    # (A) 原生 EfficientSAM
    model_orig = load_model(weights_path=None, device=device)
    pred_orig = predict_mask(model_orig, img_bgr, box, device=device)

    # (B) Hard 模式微調權重
    model_hard = load_model(weights_path="weights/decoder_exp1_hard_pure.pth", device=device)
    pred_hard = predict_mask(model_hard, img_bgr, box, device=device)

    # (C) TAOS 模式微調權重
    model_taos = load_model(weights_path="weights/decoder_exp3_taos_pure.pth", device=device)
    pred_taos = predict_mask(model_taos, img_bgr, box, device=device)

    # 3. 繪製並排對照圖
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 標上提示框
    cv2.rectangle(img_rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

    vis_gt = overlay_mask(img_rgb, gt_mask, color=(255, 215, 0), alpha=0.45)        # 金黃色: GT
    vis_orig = overlay_mask(img_rgb, pred_orig, color=(0, 165, 255), alpha=0.45)    # 橘色: 原生 SAM
    vis_hard = overlay_mask(img_rgb, pred_hard, color=(255, 0, 255), alpha=0.45)    # 紫色: Hard Cut-Paste
    vis_taos = overlay_mask(img_rgb, pred_taos, color=(0, 255, 0), alpha=0.45)      # 綠色: TAOS (Ours)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = [
        "Ground Truth (Amodal)",
        "Original EfficientSAM\n(Modal-centric)",
        "Exp1: Hard Cut-Paste\n(Decoder Fine-tuned)",
        "Exp3: TAOS Edge Blur\n(Decoder Fine-tuned)"
    ]
    images = [vis_gt, vis_orig, vis_hard, vis_taos]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300, bbox_inches="tight")
    print(f"對比圖已成功儲存至: {args.save_path}")


if __name__ == "__main__":
    main()