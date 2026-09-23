import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def apply_taos_blur(occluder_mask, ksize=7, sigma=1.2):
    """
    TAOS occluder_mask: [H, W]，數值為 0 或 1 的遮擋物遮罩
    """
    if ksize == 0:
        return occluder_mask.astype(np.float32)

    # 1. 提取遮擋物的邊界
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(occluder_mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(occluder_mask.astype(np.uint8), kernel, iterations=1)
    edge_band = dilated - eroded  

    # 2. 針對邊緣帶進行高斯模糊
    blurred_mask = cv2.GaussianBlur(
        occluder_mask.astype(np.float32), (ksize, ksize), sigma
    )

    # 3. 只有邊緣處套用模糊平滑，物體內部維持實心 1.0
    smooth_alpha = np.where(edge_band == 1, blurred_mask, occluder_mask)
    return smooth_alpha


def synthesize_occlusion_pair(
    target_img, target_mask, occluder_img, occluder_mask, use_taos=False, k=7
):
    """
    合成遮擋成對訓練資料
    """
    # H, W, _ = target_img.shape

    # 1. 決定遮擋物 Alpha 通道 (是否有 TAOS 羽化)
    if use_taos:
        alpha = apply_taos_blur(occluder_mask, ksize=k)
    else:
        alpha = occluder_mask.astype(np.float32)

    alpha_3ch = np.expand_dims(alpha, axis=-1)

    # 2. 圖像融合: 新圖 = (1 - alpha) * 目標圖 + alpha * 遮擋物圖
    occluded_img = (
        (1.0 - alpha_3ch) * target_img + alpha_3ch * occluder_img
    ).astype(np.uint8)

    # 3. 標籤計算 (遮罩運算)
    amodal_mask = target_mask  # 完整的目標永遠是 target_mask
    modal_mask = np.logical_and(
        target_mask, np.logical_not(occluder_mask)
    ).astype(
        np.uint8
    )  # 扣掉被遮擋物蓋住的部分

    return occluded_img, modal_mask, amodal_mask


# 只是測試用而已
if __name__ == "__main__":
    H, W = 400, 400

    # 1. 自動建立測試 Target: 藍色背景上的白色方塊
    target_img = np.zeros((H, W, 3), dtype=np.uint8)
    target_img[:] = (200, 180, 160)  
    target_mask = np.zeros((H, W), dtype=np.uint8)
    # 畫一個白色長方形當 Target (例如車身)
    cv2.rectangle(target_img, (80, 150), (320, 280), (255, 255, 255), -1)
    target_mask[150:280, 80:320] = 1

    # 2. 自動建立測試 Occluder: 擋在前方的圓形障礙物
    occluder_img = np.zeros((H, W, 3), dtype=np.uint8)
    occluder_img[:] = (200, 180, 160)  # 背景色相同
    occluder_mask = np.zeros((H, W), dtype=np.uint8)
    # 畫一個圓形擋在右半部 (中心點在 x=260, y=200, 半徑 70)
    cv2.circle(occluder_img, (260, 200), 70, (40, 40, 180), -1)
    cv2.circle(occluder_mask, (260, 200), 70, 1, -1)

    # 3. 分別生成 Hard Cut-Paste 與 TAOS Gaussian Blur 影像
    hard_img, hard_modal, _ = synthesize_occlusion_pair(
        target_img, target_mask, occluder_img, occluder_mask, use_taos=False
    )
    taos_img, taos_modal, amodal_gt = synthesize_occlusion_pair(
        target_img, target_mask, occluder_img, occluder_mask, use_taos=True, k=11
    )

    # 4. 存檔與局部放大視覺化
    # 鎖定交界處 (x: 180~220, y: 170~210) 放大特寫觀察像素接縫
    zoom_box = (slice(170, 210), slice(180, 220))

    plt.figure(figsize=(14, 8))

    # 全景圖
    plt.subplot(2, 3, 1)
    plt.title("A. Hard Cut-Paste (SAMEO style)")
    plt.imshow(cv2.cvtColor(hard_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("B. TAOS Gaussian Blur (Amodal SAM)")
    plt.imshow(cv2.cvtColor(taos_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("C. Ground Truth Amodal Mask")
    plt.imshow(amodal_gt, cmap="gray")
    plt.axis("off")

    # 特寫放大圖 (Zoom-in 觀察交界處像素)
    plt.subplot(2, 3, 4)
    plt.title("Zoom-in: Hard Boundary (Artifacts)")
    plt.imshow(cv2.cvtColor(hard_img[zoom_box], cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Zoom-in: TAOS Smooth Edge (Gaussian)")
    plt.imshow(cv2.cvtColor(taos_img[zoom_box], cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("Modal Mask")
    plt.imshow(taos_modal, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    save_path = "taos_vs_hard_comparison.png"
    plt.savefig(save_path, dpi=200)
    print(f"驗證成功！對比圖已成功儲存至: {save_path}")