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

print("載入完成！正在建立索引與對應表...")

# 2. 建立類別 ID 到類別資訊的對應字典
categories_dict = {}
for cat in coco_data.get('categories', []):
    categories_dict[cat['id']] = {
        "name": cat.get('name', 'unknown'),
        "supercategory": cat.get('supercategory', cat.get('hypercategory', 'unknown'))
    }

# 3. 建立圖片查詢對應表
images_dict = {}
for img in coco_data.get('images', []):
    file_name_base = os.path.splitext(img['file_name'])[0]
    images_dict[file_name_base] = img
    images_dict[img['id']] = img

# 4. 建立 image_id 到標註（annotations）列表的對應表
annotations_dict = {}
for ann in coco_data.get('annotations', []):
    img_id = ann['image_id']
    if img_id not in annotations_dict:
        annotations_dict[img_id] = []
    annotations_dict[img_id].append(ann)

print("\n=== 開始提取這 14 張圖的完整資訊（保留所有原始欄位）並下載圖片 ===\n")

results = {}
output_image_dir = "downloaded_14_images"
os.makedirs(output_image_dir, exist_ok=True)

first_ann_checked = False

for name in target_set:
    img_info = None
    if name in images_dict:
        img_info = images_dict[name]
    else:
        try:
            numeric_id = int(name.split('_')[-1])
            if numeric_id in images_dict:
                img_info = images_dict[numeric_id]
        except ValueError:
            pass

    if img_info:
        img_id = img_info['id']
        file_name = img_info['file_name']  # 例如 "COCO_val2014_000000013882.jpg"
        anns = annotations_dict.get(img_id, [])
        
        # 示範印出第一個找到的 annotation 包含哪些欄位，方便除錯確認
        if anns and not first_ann_checked:
            print(f"[提示] 檢視該 JSON 內 Annotation 的可用欄位 (Keys): {list(anns[0].keys())}")
            first_ann_checked = True

        formatted_anns = []
        for ann in anns:
            cat_id = ann.get('category_id')
            cat_info = categories_dict.get(cat_id, {"name": "unknown", "supercategory": "unknown"})
            
            # 使用 .copy() 完整保留原本 annotation 的所有欄位，並加上可讀的類別名稱
            ann_info = ann.copy()
            ann_info["category_name"] = cat_info["name"]
            ann_info["supercategory"] = cat_info["supercategory"]
            
            formatted_anns.append(ann_info)

        results[name] = {
            "image_info": img_info,
            "annotations_count": len(formatted_anns),
            "annotations": formatted_anns
        }

        # 處理圖片下載或複製
        local_img_path = os.path.join(output_image_dir, file_name)
        
        # 檢查本機是否已經有 val2014 資料夾可以複製
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
                print(f"[成功] {name} -> 從本機路徑複製成功")
                copied = True
                break
        
        if not copied:
            # 如果本機沒有，則從官方網址下載
            url = f"http://images.cocodataset.org/val2014/{file_name}"
            try:
                print(f"正在從網路下載 {file_name} ...")
                urllib.request.urlretrieve(url, local_img_path)
                print(f"[成功] {name} -> 網路下載成功")
            except Exception as e:
                print(f"[錯誤] {name} 下載失敗: {e}")

    else:
        print(f"[警告] 在 JSON 中找不到圖片: {name}")
        results[name] = {"error": "Image not found in JSON"}

# 5. 將結果輸出成獨立的 JSON 檔案
output_json_path = "extracted_14_images_info.json"
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n========================================")
print(f"處理完畢！")
print(f"- 詳細 JSON（含所有原始欄位）已儲存至: {output_json_path}")
print(f"- 圖片已儲存至資料夾: {output_image_dir}/")
print(f"========================================")