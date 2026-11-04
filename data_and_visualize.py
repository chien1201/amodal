import json
import os
import shutil
import urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as maskUtils

# ==========================================
# 階段一：設定與資料提取
# ==========================================

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

# 3. 建立圖片查詢對應表
images_dict = {}
for img in coco_data.get('images', []):
    file_name_base = os.path.splitext(img['file_name'])[0]
    images_dict[file_name_base] = img
    images_dict[str(img['id'])] = img
    images_dict[int(img['id'])] = img

# 4. 建立 image_id 到標註列表的對應表
annotations_dict = {}
for ann in coco_data.get('annotations', []):
    img_id = ann['image_id']
    annotations_dict.setdefault(img_id, []).append(ann)
    if not isinstance(img_id, str):
        annotations_dict.setdefault(str(img_id), []).append(ann)

print("\n=== 開始提取 14 張圖的完整資訊並下載圖片 ===\n")

results = {}
output_image_dir = "downloaded_14_images"
os.makedirs(output_image_dir, exist_ok=True)

for name in target_set:
    img_info = None
    if name in images_dict:
        img_info = images_dict[name]
    else:
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
        anns = annotations_dict.get(img_id, [])
        if not anns and str(img_id) in annotations_dict:
            anns = annotations_dict[str(img_id)]
        elif not anns and int(img_id) in annotations_dict:
            anns = annotations_dict[int(img_id)]
            
        formatted_anns = []
        for ann in anns:
            cat_id = ann.get('category_id')
            cat_info = categories_dict.get(cat_id, {"name": "unknown", "supercategory": "unknown"})
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
                shutil.copy(src, local_img_path)
                copied = True
                break
        if not copied:
            url = f"http://images.cocodataset.org/val2014/{file_name}"
            try:
                urllib.request.urlretrieve(url, local_img_path)
            except Exception as e:
                print(f" [錯誤] 圖片下載失敗: {e}")

    else:
        print(f"[警告] 在 JSON 中找不到圖片: {name}")
        results[name] = {"error": "Image not found in JSON"}

# 5. 將結果輸出成獨立的 JSON 檔案
output_json_path = "extracted_14_images_info.json"
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n資料提取完畢，JSON 已儲存至: {output_json_path}\n")


# ==========================================
# 階段二：視覺化繪製 (無條件全部繪製)
# ==========================================

output_vis_dir = "visualized_14_images_with_mask"
os.makedirs(output_vis_dir, exist_ok=True)

print("=== 開始將所有 Annotation 無條件繪製到圖片上 ===\n")

for img_name, data in results.items():
    if "error" in data:
        print(f"[跳過] {img_name}: {data['error']}")
        continue
        
    img_info = data["image_info"]
    file_name = img_info["file_name"]
    img_path = os.path.join(output_image_dir, file_name)
    if not os.path.exists(img_path):
        print(f"[警告] 找不到圖片檔案: {img_path}")
        continue
        
    # 開啟圖片並轉成 RGBA 模式
    base_image = Image.open(img_path).convert("RGBA")
    width, height = base_image.size
    
    # 建立透明圖層畫 Mask
    overlay = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(base_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        
    annotations = data["annotations"]
    for ann in annotations:
        cat_name = ann.get("category_name", "unknown")
        segmentation = ann.get("segmentation")
        
        binary_mask = None
        
        # 1. 無條件解析 Segmentation (Polygon 或 RLE)
        if segmentation:
            if isinstance(segmentation, list):
                # 移除 >= 6 的限制，只要有點就盡量畫
                for poly in segmentation:
                    if len(poly) >= 4:
                        polygon_points = [(poly[i], poly[i+1]) for i in range(0, len(poly) - len(poly)%2, 2)]
                        if len(polygon_points) >= 3:
                            draw_overlay.polygon(polygon_points, fill=(255, 0, 0, 80), outline=(255, 0, 0, 200))
                        elif len(polygon_points) == 2:
                            draw_overlay.line(polygon_points, fill=(255, 0, 0, 200), width=2)
            elif isinstance(segmentation, dict):
                try:
                    binary_mask = maskUtils.decode(segmentation)
                    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
                    red_tint = Image.new("RGBA", (width, height), (255, 0, 0, 100))
                    overlay.paste(red_tint, (0, 0), mask_img)
                except Exception as e:
                    print(f"[提示] RLE 解碼失敗，嘗試略過或使用備用框 (ID: {ann.get('id')}): {e}")

        # 2. 繪製 Modal Bounding Box (藍色) - 若無 bbox 則嘗試從 mask 推導
        bbox = ann.get("bbox")
        if not bbox and binary_mask is not None:
            # 從 mask 計算 bbox [x, y, w, h]
            y_indices, x_indices = np.where(binary_mask > 0)
            if len(x_indices) > 0 and len(y_indices) > 0:
                x0, x1 = x_indices.min(), x_indices.max()
                y0, y1 = y_indices.min(), y_indices.max()
                bbox = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]

        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            x0, y0, x1, y1 = x, y, x + w, y + h
            draw_base.rectangle([x0, y0, x1, y1], outline="blue", width=2)
            draw_base.text((x0, max(0, y0 - 16)), f"{cat_name} (modal)", fill="blue", font=font)

        # 3. 繪製 Amodal Bounding Box (紅色)
        amodal_bbox = ann.get("amodal_bbox")
        if amodal_bbox and len(amodal_bbox) == 4:
            ax, ay, aw, ah = amodal_bbox
            ax0, ay0, ax1, ay1 = ax, ay, ax + aw, ay + ah
            draw_base.rectangle([ax0, ay0, ax1, ay1], outline="red", width=2)
            draw_base.text((ax0, max(18, ay0 - 32)), f"{cat_name} (amodal)", fill="red", font=font)

    # 合併圖片與半透明 Mask 圖層
    combined_image = Image.alpha_composite(base_image, overlay).convert("RGB")

    output_path = os.path.join(output_vis_dir, file_name)
    combined_image.save(output_path)
    print(f"[成功] 已儲存視覺化圖片: {output_path}")

print(f"\n========================================")
print(f"全部執行完畢！")
print(f"- 圖片已儲存至資料夾: {output_vis_dir}/")
print(f"========================================")