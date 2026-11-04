import json
import os
import urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as maskUtils

# ==========================================
# 階段一：強健地解析 Detectron 預測結果 JSON / JSONL
# ==========================================

target_image_names = [
    "COCO_val2014_000000013882", "COCO_val2014_000000007682",
    "COCO_val2014_000000034193", "COCO_val2014_000000034818",
    "COCO_val2014_000000055074", "COCO_val2014_000000057238",
    "COCO_val2014_000000078200", "COCO_val2014_000000145426",
    "COCO_val2014_000000238866", "COCO_val2014_000000297604",
    "COCO_val2014_000000324476", "COCO_val2014_000000373356",
    "COCO_val2014_000000484441", "COCO_val2014_000000506127"
]

target_ids = set()
for name in target_image_names:
    try:
        numeric_id = int(name.strip().split('_')[-1])
        target_ids.add(numeric_id)
    except ValueError:
        pass

json_path = "COCO_amodal_test2014_detectron.json"
print(f"正在載入 Detectron 結果檔案: {json_path} ...")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"找不到檔案 {json_path}，請確保與程式放在同一資料夾下。")

# 強健的讀取機制：同時支援 標準 JSON 與 JSON Lines (.jsonl)
detectron_results = []
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        # 嘗試用標準 JSON 讀取
        try:
            detectron_results = json.loads(content)
        except json.JSONDecodeError:
            # 如果失敗，嘗試以行分隔 JSON (JSONL) 逐行讀取
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    detectron_results.append(json.loads(line))
except Exception as e:
    raise RuntimeError(f"讀取 JSON 失敗: {e}")

print(f"載入完成！總共包含 {len(detectron_results)} 筆資料。正在進行過濾...")

# COCO 80 類別對應表
coco_classes = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
    65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

# 確保只處理型態為 dict 的項目，避免字串錯誤
annotations_by_image = {}
for item in detectron_results:
    if not isinstance(item, dict):
        continue  # 如果不是字典就跳過
    img_id = item.get('image_id')
    if img_id in target_ids:
        annotations_by_image.setdefault(img_id, []).append(item)

print("\n=== 開始下載圖片並整理預測結果 ===\n")
output_image_dir = "downloaded_14_images"
os.makedirs(output_image_dir, exist_ok=True)

results = {}
for img_id in target_ids:
    file_name = f"COCO_val2014_{img_id:012d}.jpg"
    anns = annotations_by_image.get(img_id, [])
    
    formatted_anns = []
    for ann in anns:
        cat_id = ann.get('category_id')
        cat_name = coco_classes.get(cat_id, f"class_{cat_id}")
        ann_info = ann.copy()
        ann_info["category_name"] = cat_name
        formatted_anns.append(ann_info)

    base_name = f"COCO_val2014_{img_id:012d}"
    results[base_name] = {
        "image_id": img_id,
        "file_name": file_name,
        "annotations_count": len(formatted_anns),
        "annotations": formatted_anns
    }
    print(f"[成功] 圖片 ID {img_id} -> 對應到 {len(formatted_anns)} 個預測物件")

    # 下載圖片
    local_img_path = os.path.join(output_image_dir, file_name)
    if not os.path.exists(local_img_path):
        url = f"http://images.cocodataset.org/val2014/{file_name}"
        try:
            urllib.request.urlretrieve(url, local_img_path)
        except Exception as e:
            print(f" [錯誤] 圖片下載失敗 {file_name}: {e}")