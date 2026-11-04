import json
import os
import urllib.request

# 1. 定義您指定的 14 張圖片名稱
target_image_names = [
    "COCO_val2014_000000013882",
    "COCO_val2014_000000007682",
    "COCO_val2014_000000034193",
    "COCO_val2014_000000034818",
    "COCO_val2014_000000055074",
    "COCO_val2014_000000057238",
    "COCO_val2014_000000078200",
    "COCO_val2014_000000145426",
    "COCO_val2014_000000238866",
    "COCO_val2014_000000297604",
    "COCO_val2014_000000324476",
    "COCO_val2014_000000373356",
    "COCO_val2014_000000484441",
    "COCO_val2014_000000506127"
]

# 將名稱標準化（去除可能的空白與副檔名）
target_set = {os.path.splitext(name.strip())[0] for name in target_image_names}

# 讀取指定的 JSON 檔案
json_path = "COCO_amodal_val2014_with_classes.json"
print(f"正在載入 JSON 檔案: {json_path} ...")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"找不到檔案 {json_path}，請確保程式與 JSON 檔案放在同一個資料夾下。")

with open(json_path, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

print("載入完成！正在建立強健的索引與對應表...")

# 2. 建立類別 ID 到類別資訊的對應字典
categories_dict = {}
for cat in coco_data.get('categories', []):
    categories_dict[cat['id']] = {
        "name": cat.get('name', 'unknown'),
        "supercategory": cat.get('supercategory', cat.get('hypercategory', 'unknown'))
    }

# 3. 建立圖片查詢對應表（同時支援檔名、字串 ID、整數 ID）
images_dict = {}
for img in coco_data.get('images', []):
    file_name_base = os.path.splitext(img['file_name'])[0]
    images_dict[file_name_base] = img
    images_dict[str(img['id'])] = img
    images_dict[int(img['id'])] = img

# 4. 建立 image_id 到標註列表的對應表（確保不漏掉任何標註）
annotations_dict = {}
for ann in coco_data.get('annotations', []):
    img_id = ann['image_id']
    annotations_dict.setdefault(img_id, []).append(ann)
    # 同時用字串形式備份一次鍵值，防止後續查詢型態不合
    if not isinstance(img_id, str):
        annotations_dict.setdefault(str(img_id), []).append(ann)

print("\n=== 開始正確提取 14 張圖的完整資訊並下載圖片 ===\n")

results = {}
output_image_dir = "downloaded_14_images"
os.makedirs(output_image_dir, exist_ok=True)

for name in target_set:
    img_info = None
    # 嘗試用檔名尋找
    if name in images_dict:
        img_info = images_dict[name]
    else:
        # 嘗試從檔名尾端解析出數值 ID 尋找
        try:
            numeric_id = int(name.split('_')[-1])
            if numeric_id in images_dict:
                img_info = images_dict[numeric_id]
            elif str(numeric_id) in images_dict:
                img_info = images_dict[str(numeric_id)]
        except ValueError:
            pass

    if img_info:
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # 透過多種型態安全地取得該圖片的所有 annotations
        anns = annotations_dict.get(img_id, [])
        if not anns and str(img_id) in annotations_dict:
            anns = annotations_dict[str(img_id)]
        elif not anns and int(img_id) in annotations_dict:
            anns = annotations_dict[int(img_id)]
        
        formatted_anns = []
        for ann in anns:
            cat_id = ann.get('category_id')
            cat_info = categories_dict.get(cat_id, {"name": "unknown", "supercategory": "unknown"})
            
            # 完整保留原 ann 的所有欄位，並加上可讀的類別名稱
            ann_info = ann.copy()
            ann_info["category_name"] = cat_info["name"]
            ann_info["supercategory"] = cat_info["supercategory"]
            
            formatted_anns.append(ann_info)

        results[name] = {
            "image_info": img_info,
            "annotations_count": len(formatted_anns),
            "annotations": formatted_anns
        }
        
        print(f"[成功] 圖片 {name} (ID: {img_id}) -> 成功對應 {len(formatted_anns)} 個 Instance")

        # 處理圖片下載或複製
        local_img_path = os.path.join(output_image_dir, file_name)
        local_source_candidates = [
            os.path.join("val2014", file_name),
            os.path.join("coco", "val2014", file_name),
            file_name
        ]
        
        copied = False
        for src in local_source_candidates:
            if os.path.exists(src):
                import shutil
                shutil.copy(src, local_img_path)
                copied = True
                break
        
        if not copied:
            url = f"http://images.cocodataset.org/val2014/{file_name}"
            try:
                urllib.request.urlretrieve(url, local_img_path)
            except Exception as e:
                print(f"  [錯誤] 圖片下載失敗: {e}")

    else:
        print(f"[警告] 在 JSON 中找不到圖片: {name}")
        results[name] = {"error": "Image not found in JSON"}

# 5. 將結果輸出成獨立的 JSON 檔案
output_json_path = "extracted_14_images_info.json"
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n========================================")
print(f"重新提取完畢！")
print(f"- 詳細 JSON 已更新儲存至: {output_json_path}")
print(f"========================================")