# 生成不同重疊比例對比圖 occlusion_ratios_comparison.png (修正幾何覆蓋面積問題)

import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_taos_blur(occluder_mask, ksize=7, sigma=1.2):
    """
    TAOS occluder_mask: [H, W]，數值為 0 或 1 的遮擋物遮罩
    """
    if ksize == 0:
        return occluder_mask.astype(np.float32)

    kernel_size = max(3, ksize // 2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    dilated = cv2.dilate(occluder_mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(occluder_mask.astype(np.uint8), kernel, iterations=1)
    edge_band = dilated - eroded  

    blurred_mask = cv2.GaussianBlur(
        occluder_mask.astype(np.float32), (ksize, ksize), sigma
    )

    smooth_alpha = np.where(edge_band == 1, blurred_mask, occluder_mask)
    return smooth_alpha


def synthesize_occlusion_pair(
    target_img, target_mask, occluder_img, occluder_mask, use_taos=True, k=11
):
    """
    合成遮擋成對訓練資料
    """
    if use_taos:
        alpha = apply_taos_blur(occluder_mask, ksize=k, sigma=max(1.0, k / 4.0))
    else:
        alpha = occluder_mask.astype(np.float32)

    alpha_3ch = np.expand_dims(alpha, axis=-1)

    occluded_img = (
        (1.0 - alpha_3ch) * target_img + alpha_3ch * occluder_img
    ).astype(np.uint8)

    modal_mask = np.logical_and(
        target_mask, np.logical_not(occluder_mask)
    ).astype(np.uint8)
    amodal_mask = target_mask

    return occluded_img, modal_mask, amodal_mask


if __name__ == "__main__":
    H, W = 400, 400
    ratios = [0.2, 0.4, 0.6, 0.8]
    fig, axes = plt.subplots(1, len(ratios), figsize=(18, 4))

    # Target 邊界: x 從 80 到 320，總寬度 = 240
    target_x_min, target_x_max = 80, 320
    target_width = target_x_max - target_x_min  # 240
    
    # 放大圓形半徑
    radius = 120  

    for i, ratio in enumerate(ratios):
        # 1. 建立測試 Target
        target_img = np.zeros((H, W, 3), dtype=np.uint8)
        target_img[:] = (200, 180, 160)  
        target_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(target_img, (target_x_min, 130), (target_x_max, 300), (255, 255, 255), -1)
        target_mask[130:300, target_x_min:target_x_max] = 1

        # 2. 建立測試 Occluder (放大後的圓形障礙物)
        occluder_img = np.zeros((H, W, 3), dtype=np.uint8)
        occluder_img[:] = (200, 180, 160)  
        occluder_mask = np.zeros((H, W), dtype=np.uint8)

        # 【關鍵修正】確保圓的「最左邊界」精準切在目標的比例位置上
        overlap_len = target_width * ratio
        # circle_x 是圓心，要讓圓的左邊界落在 (target_x_max - overlap_len)，圓心就必須再往右加一個半徑
        circle_x = int(target_x_max - overlap_len + radius)
        circle_y = 215

        cv2.circle(occluder_img, (circle_x, circle_y), radius, (40, 40, 180), -1)
        cv2.circle(occluder_mask, (circle_x, circle_y), radius, 1, -1)

        # 為了驗證，我們計算實際覆蓋的「像素面積比例」
        actual_overlap_pixels = np.sum(np.logical_and(target_mask, occluder_mask))
        total_target_pixels = np.sum(target_mask)
        actual_ratio = actual_overlap_pixels / total_target_pixels

        # 3. 進行合成 (帶有 TAOS 效果)
        occluded_img, _, _ = synthesize_occlusion_pair(
            target_img, target_mask, occluder_img, occluder_mask, use_taos=True, k=11
        )

        # 4. 繪製圖表面板
        axes[i].imshow(cv2.cvtColor(occluded_img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Target Overlap: {int(ratio*100)}%\n(Actual Area: {int(actual_ratio*100)}%)", fontsize=12)
        axes[i].axis("off")

    plt.suptitle("Occlusion Ratios Comparison with TAOS (Enlarged Occluder)", fontsize=16, y=1.02)
    plt.tight_layout()
    save_path = "occlusion_ratios_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"修正覆蓋比例的對比圖已成功儲存至: {save_path}")