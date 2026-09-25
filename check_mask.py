import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 尋找第一個可用的樣本資料夾進行檢查
    sample_dir = "data/output_samples/sample_4_taos"
    img_path = os.path.join(sample_dir, "image.jpg")
    mask_path = os.path.join(sample_dir, "amodal_mask.png")
    json_path = os.path.join(sample_dir, "box_prompt.json")

    if os.path.exists(img_path) and os.path.exists(json_path):
        # 1. 先讀取圖片與遮罩
        image = cv2.imread(img_path)
        amodal_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        with open(json_path, "r") as f:
            prompts = json.load(f)
            modal_box = prompts["modal_box"]
            amodal_box = prompts["amodal_box"]

        print("原始 JSON Modal Box:", modal_box)
        print("原始 JSON Amodal Box:", amodal_box)

        # 💡 自動修正：直接從 amodal_mask 計算出真正的 amodal_box
        y_indices, x_indices = np.where(amodal_mask > 0)
        if len(x_indices) > 0 and len(y_indices) > 0:
            amodal_box = [int(np.min(x_indices)), int(np.min(y_indices)), 
                          int(np.max(x_indices)), int(np.max(y_indices))]
            print("修正後 Amodal Box (從 Mask 計算):", amodal_box)

        # 2. 在 BGR 格式下畫框：(0, 0, 255) 是紅色，(0, 255, 0) 是綠色
        # 先畫紅框（Amodal），再畫綠框（Modal），確保綠框不會把紅框完全蓋住（若有交疊）
        cv2.rectangle(image, (amodal_box[0], amodal_box[1]), (amodal_box[2], amodal_box[3]), (0, 0, 255), 2)
        cv2.rectangle(image, (modal_box[0], modal_box[1]), (modal_box[2], modal_box[3]), (0, 255, 0), 2)

        # 3. 畫完之後再轉成 RGB 給 matplotlib 顯示
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("Boxes Inspection\n(Red: Amodal, Green: Modal)")
        axes[0].axis("off")

        axes[1].imshow(amodal_mask, cmap="gray")
        axes[1].set_title("Amodal Mask Ground Truth")
        axes[1].axis("off")

        plt.tight_layout()
        save_path = "annotation_inspection.png"
        plt.savefig(save_path, dpi=200)
        print(f"標註檢查圖已成功儲存至: {save_path}")
    else:
        print("找不到指定的樣本資料夾，請確認是否已生成範例數據！")