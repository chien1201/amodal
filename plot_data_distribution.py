import os
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_root = "data/output_samples"
    areas = []

    if os.path.exists(data_root):
        for sample_name in os.listdir(data_root):
            json_path = os.path.join(data_root, sample_name, "box_prompt.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                    box = data["amodal_box"]
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    areas.append(width * height)

    if len(areas) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(areas, bins=10, color='skyblue', edgecolor='black')
        plt.title("Amodal Bounding Box Area Distribution")
        plt.xlabel("Area (Pixels)")
        plt.ylabel("Count")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        save_path = "dataset_distribution.png"
        plt.savefig(save_path, dpi=200)
        print(f"資料集分佈直方圖已儲存至: {save_path}")
    else:
        print("目前尚無足夠的樣本數據可供統計，請先執行資料生成腳本！")