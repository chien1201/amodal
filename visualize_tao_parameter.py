# 生成帶有各別 k 值 Zoom-in 邊界特寫的 taos_params_comparison.png

import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_taos_blur(occluder_mask, ksize=7, sigma=1.2):
    """
    TAOS occluder_mask: [H, W]，數值為 0 或 1 的遮擋物遮罩
    """
    if ksize == 0:
        return occluder_mask.astype(np.float32)

    # 【修改點】讓邊緣帶的寬度隨著 ksize 放大，否則大 k 值會被固定的 3x3 邊緣限制住
    kernel_size = max(3, ksize // 2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 1. 提取隨 k 變寬的遮擋物邊界
    dilated = cv2.dilate(occluder_mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(occluder_mask.astype(np.uint8), kernel, iterations=1)
    edge_band = dilated - eroded  

    # 2. 針對邊緣帶進行高斯模糊
    blurred_mask = cv2.GaussianBlur(
        occluder_mask.astype(np.float32), (ksize, ksize), sigma
    )

    # 3. 套用模糊平滑
    smooth_alpha = np.where(edge_band == 1, blurred_mask, occluder_mask)
    return smooth_alpha


def synthesize_occlusion_pair(
    target_img, target_mask, occluder_img, occluder_mask, use_taos=False, k=7
):
    """
    合成遮擋成對訓練資料
    """
    if use_taos:
        alpha = apply_taos_blur(occluder_mask, ksize=k, sigma=max(1.0, k / 4.0))
    else:
        alpha = occluder_mask.astype(np.uint8)

    alpha_3ch = np.expand_dims(alpha, axis=-1)

    # 圖像融合: 新圖 = (1 - alpha) * 目標圖 + alpha * 遮擋物圖
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

    # 1. 建立測試 Target: 色彩背景上的白色方塊
    target_img = np.zeros((H, W, 3), dtype=np.uint8)
    target_img[:] = (200, 180, 160)  
    target_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(target_img, (80, 150), (320, 280), (255, 255, 255), -1)
    target_mask[150:280, 80:320] = 1

    # 2. 建立測試 Occluder: 圓形障礙物
    occluder_img = np.zeros((H, W, 3), dtype=np.uint8)
    occluder_img[:] = (200, 180, 160)  
    occluder_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(occluder_img, (260, 200), 70, (40, 40, 180), -1)
    cv2.circle(occluder_mask, (260, 200), 70, 1, -1)

    # 3. 設定參數與繪圖版面 (2 行 x 5 列)
    k_values = [0, 3, 7, 11, 15]
    fig, axes = plt.subplots(2, len(k_values), figsize=(18, 8))

    # 鎖定紅圓與白方塊交界的特寫區塊 (x: 180~220, y: 170~210)
    zoom_box = (slice(170, 210), slice(180, 220))

    for i, k in enumerate(k_values):
        use_taos = (k > 0)
        occluded_img, _, _ = synthesize_occlusion_pair(
            target_img, target_mask, occluder_img, occluder_mask, use_taos=use_taos, k=k
        )
        
        # --- 第一列：全景圖 ---
        axes[0, i].imshow(cv2.cvtColor(occluded_img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"k = {k} (TAOS)" if k > 0 else "k = 0 (Hard)", fontsize=12)
        axes[0, i].axis("off")

        # --- 第二列：對應的邊界 Zoom-in 特寫 ---
        zoomed_patch = occluded_img[zoom_box]
        axes[1, i].imshow(cv2.cvtColor(zoomed_patch, cv2.COLOR_BGR2RGB))
        axes[1, i].axis("off")

    # 加入行標題說明
    fig.text(0.08, 0.75, "Row 1: Full View\n(k = 0, 3, 7, 11, 15)", va='center', ha='center', fontsize=11, rotation='vertical', weight='bold')
    fig.text(0.08, 0.28, "Row 2: Zoom-in on Boundary\n(Pixel-level details)", va='center', ha='center', fontsize=11, rotation='vertical', weight='bold')

    plt.suptitle("TAOS Blur Parameter Comparison (k-values): Full View and Zoomed Edges", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0.05, 0, 1, 0.93])
    
    save_path = "taos_params_comparison.png"
    plt.savefig(save_path, dpi=200)
    print(f"羽化參數對比圖（含對應 Zoom-in）已成功儲存至: {save_path}")