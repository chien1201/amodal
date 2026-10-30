import json
import os
import random
import re
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pycocotools.coco import COCO
from tqdm import tqdm # 引入進度條工具

# 載入 Meta 官方 Segment Anything 套件
from segment_anything import SamPredictor, sam_model_registry

from dataset_generator import apply_taos_blur, synthesize_occlusion_pair


# =========================================================================
# 輔助工具：帶有進度條的檔案下載器
# =========================================================================
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_progress(url, output_path, desc=None):
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to
        )


# =========================================================================
# 模組 A：Mask 轉 Bounding Box 為 prompt
# =========================================================================
def mask_to_bbox(mask):
    """從二值遮罩 [H, W] 計算出包含整個物件的最小外接矩形 [x1, y1, x2, y2]"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 0, 0]
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [int(xmin), int(ymin), int(xmax + 1), int(ymax + 1)]


# =========================================================================
# 新增功能：修復破碎 Mask 與填補內部孔洞（同時回傳填補前後的實心指標）
# =========================================================================
def refine_and_fill_mask(mask):
    """處理 Mask 內部破碎、孔洞或零碎斷開問題：
    1. 使用形態學「閉合運算 (Closing)」填補內部小孔洞。
    2. 尋找最大外輪廓，過濾掉周邊不相連的雜訊斑點，保留完整主體。
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # 記錄原始未填補前的像素面積
    raw_area = np.sum(mask_uint8 > 0)

    # 1. 閉合運算 (Closing) 填補內小孔
    closed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2. 尋找最大外輪廓
    contours, _ = cv2.findContours(
        closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(mask, dtype=np.uint8), 0.0

    max_contour = max(contours, key=cv2.contourArea)

    filled_mask = np.zeros_like(mask_uint8)
    cv2.drawContours(filled_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    filled_area = np.sum(filled_mask > 0)

    # 計算「實心扎實度 (Solidity) / 內部殘缺率」
    solidity = float(raw_area) / float(filled_area) if filled_area > 0 else 0.0

    final_mask = (filled_mask > 0).astype(np.uint8)
    return final_mask, solidity


# =========================================================================
# 模組 B-1：SAM 提煉器初始化 (方案 A 核心)
# =========================================================================
def init_sam_refiner(weights_dir="weights"):
    """初始化 SAM 模型 (vit_b 輕量版) 作為高品質 Mask 提煉工具"""
    os.makedirs(weights_dir, exist_ok=True)
    sam_checkpoint = os.path.join(weights_dir, "sam_vit_b_01ec64.pth")
    if not os.path.exists(sam_checkpoint):
        sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        download_with_progress(
            sam_url,
            sam_checkpoint,
            desc="下載 SAM 權重 (sam_vit_b_01ec64.pth)",
        )
        print("SAM 權重下載完成！")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在載入 SAM 模型至運算裝置: {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def refine_mask_with_sam(predictor, img_bgr, bbox):
    """輸入原始影像與粗略 Bounding Box，由 SAM 重新預測像素級貼合物體輪廓的 Mask"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    box_np = np.array(bbox)[None, :]  # Shape: (1, 4)
    masks, scores, _ = predictor.predict(box=box_np, multimask_output=False)
    raw_refined_mask = masks[0].astype(np.uint8)

    # 呼叫修復與填補函式，同時取得實心扎實度 (Solidity)
    refined_mask, solidity = refine_and_fill_mask(raw_refined_mask)
    return refined_mask, solidity


# =========================================================================
# 模組 B-2：結合 InstaOrder 資料集的 COCO 訓練集讀取與無遮擋實例篩選
# （支援快取讀取與從已產生的 clean 樣本池中還原）
# =========================================================================
def prepare_coco_samples(
    predictor, download_dir="data/baseline/coco_train_raw", output_base_dir="data/baseline/output_train_samples", num_imgs=80000, target_instances_count=4000
):
    pool_cache_path = os.path.join(download_dir, "instances_pool_cache.pkl")
    # 嘗試從先前產生的 _clean 資料夾或快取中還原，避免重複從網路載入 COCO 影像
    instances_pool = []
    if os.path.exists(output_base_dir):
        print(f"偵測到已存在的輸出目錄 {output_base_dir}，正在嘗試從中還原已收集的實例池...")
        sample_dirs = sorted([d for d in os.listdir(output_base_dir) if "_clean" in d], 
        key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])
        for s_dir in tqdm(sample_dirs, desc="還原已收集的實例"):
            clean_dir_path = os.path.join(output_base_dir, s_dir)
            json_path = os.path.join(clean_dir_path, "original_info.json")
            target_img_path = os.path.join(clean_dir_path, "target_image.jpg")
            target_mask_path = os.path.join(clean_dir_path, "target_mask.png")
            occluder_img_path = os.path.join(clean_dir_path, "occluder_image.jpg")
            occluder_mask_path = os.path.join(clean_dir_path, "occluder_mask.png")
            if os.path.exists(json_path) and os.path.exists(target_img_path) and os.path.exists(occluder_img_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                t_img = cv2.imread(target_img_path)
                t_mask = (cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
                o_img = cv2.imread(occluder_img_path)
                o_mask = (cv2.imread(occluder_mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
                if t_img is not None and t_mask.sum() > 0:
                    tx1, ty1, tx2, ty2 = mask_to_bbox(t_mask)
                    instances_pool.append({
                        "original_img": t_img,
                        "full_mask": t_mask,
                        "crop_img": t_img[ty1:ty2, tx1:tx2],
                        "crop_mask": t_mask[ty1:ty2, tx1:tx2],
                        "category_id": meta["target_info"]["category_id"],
                        "category_name": meta["target_info"]["category_name"],
                        "bbox": [tx1, ty1, tx2, ty2],
                        "img_id": s_dir + "_target"  # 確保還原時也有唯一的識別來源
                    })
                if o_img is not None and o_mask.sum() > 0:
                    ox1, oy1, ox2, oy2 = mask_to_bbox(o_mask)
                    instances_pool.append({
                        "original_img": o_img,
                        "full_mask": o_mask,
                        "crop_img": o_img[oy1:oy2, ox1:ox2],
                        "crop_mask": o_mask[oy1:oy2, ox1:ox2],
                        "category_id": meta["occluder_info"]["category_id"],
                        "category_name": meta["occluder_info"]["category_name"],
                        "bbox": [ox1, oy1, ox2, oy2],
                        "img_id": s_dir + "_occluder"  # 確保還原時也有唯一的識別來源
                    })
            if len(instances_pool) >= target_instances_count:
                break
        if len(instances_pool) >= 20:
            print(f"成功從現有檔案還原了 {len(instances_pool)} 個實例！")
            return instances_pool

    # 若無法還原則執行原本的 COCO 收集流程
    os.makedirs(download_dir, exist_ok=True)
    ann_path = os.path.join(download_dir, "instances_train2017.json")

    if not os.path.exists(ann_path):
        ann_url = (
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        )
        zip_path = os.path.join(download_dir, "annotations.zip")
        download_with_progress(ann_url, zip_path, desc="下載 COCO trainval2017 標註檔")
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract("annotations/instances_train2017.json", download_dir)
        os.rename(
            os.path.join(download_dir, "annotations/instances_train2017.json"),
            ann_path,
        )
        if os.path.exists(zip_path):
            os.remove(zip_path)

    coco = COCO(ann_path)

    instaorder_path = os.path.join(download_dir, "InstaOrder_train2017.json")
    img_id_to_instaorder = {}

    if os.path.exists(instaorder_path):
        print(
            f"成功偵測到 InstaOrder 標註檔：{instaorder_path}，開始建立深度層次對應表..."
        )
        with open(instaorder_path, "r", encoding="utf-8") as f:
            instaorder_raw = json.load(f)

        annotations_list = (
            instaorder_raw.get("annotations", [])
            if isinstance(instaorder_raw, dict)
            else instaorder_raw
        )
        for item in annotations_list:
            if "image_id" in item:
                img_id_to_instaorder[item["image_id"]] = item
    else:
        print(f"[提示] 未在 {instaorder_path} 偵測到 InstaOrder 標註檔。")

    img_ids = coco.getImgIds()[:num_imgs]
    stuff_blacklist = [
        "wall",
        "floor",
        "ceiling",
        "sky",
        "road",
        "grass",
        "dirt",
        "pavement",
    ]

    img_pbar = tqdm(img_ids, desc="掃描 COCO 訓練集影像")
    instance_pbar = tqdm(total=target_instances_count, desc="收集頂層無遮擋實例")

    for img_id in img_pbar:
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info["coco_url"]
        local_img_path = os.path.join(download_dir, img_info["file_name"])

        if not os.path.exists(local_img_path):
            try:
                urllib.request.urlretrieve(img_url, local_img_path)
            except Exception:
                continue
        img_bgr = cv2.imread(local_img_path)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        image_area = float(H * W)

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        unoccluded_coco_ann_ids = set()
        if img_id in img_id_to_instaorder:
            io_record = img_id_to_instaorder[img_id]
            instance_ids_list = io_record.get("instance_ids", [])
            depth_list = io_record.get("depth", [])

            occluded_indices = set()
            for d in depth_list:
                order_str = d.get("order", "")
                if "<" in order_str:
                    parts = order_str.split("<")
                    if len(parts) == 2:
                        try:
                            occluded_idx = int(parts[1].strip())
                            occluded_indices.add(occluded_idx)
                        except ValueError:
                            pass

            for idx, coco_ann_id in enumerate(instance_ids_list):
                if idx not in occluded_indices:
                    unoccluded_coco_ann_ids.add(coco_ann_id)

        img_extracted_count = 0

        for ann in anns:
            if img_extracted_count >= 3:
                break

            cat_id = ann["category_id"]
            cat_name = coco.loadCats(cat_id)[0]["name"].lower()
            if any(bad_word in cat_name for bad_word in stuff_blacklist):
                continue

            if ann["area"] < 5000 or not isinstance(ann["segmentation"], list):
                continue

            if ann["area"] / image_area > 0.9:
                continue

            if (
                img_id in img_id_to_instaorder
                and ann["id"] not in unoccluded_coco_ann_ids
            ):
                continue

            mask = coco.annToMask(ann)
            coarse_box = mask_to_bbox(mask)
            x1, y1, x2, y2 = coarse_box
            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < 50 or box_height < 50:
                continue

            margin = 10
            if (
                x1 <= margin
                or y1 <= margin
                or x2 >= (W - margin)
                or y2 >= (H - margin)
            ):
                continue

            refined_mask, solidity = refine_mask_with_sam(
                predictor, img_bgr, coarse_box
            )

            if solidity < 0.85:
                continue

            ref_x1, ref_y1, ref_x2, ref_y2 = mask_to_bbox(refined_mask)
            ref_w = ref_x2 - ref_x1
            ref_h = ref_y2 - ref_y1

            if ref_w < 40 or ref_h < 40:
                continue

            if (
                ref_x1 <= margin
                or ref_y1 <= margin
                or ref_x2 >= (W - margin)
                or ref_y2 >= (H - margin)
            ):
                continue

            refined_mask_area = np.sum(refined_mask > 0)
            box_area = float(ref_w * ref_h)
            if box_area > 0 and (refined_mask_area / box_area < 0.1):
                continue

            fill_ratio = refined_mask_area / box_area
            if fill_ratio < 0.50:
                continue

            instances_pool.append({
                "original_img": img_bgr,
                "full_mask": refined_mask,
                "crop_img": img_bgr[ref_y1:ref_y2, ref_x1:ref_x2],
                "crop_mask": refined_mask[ref_y1:ref_y2, ref_x1:ref_x2],
                "category_id": cat_id,
                "category_name": cat_name,
                "bbox": [ref_x1, ref_y1, ref_x2, ref_y2],
                "img_id": img_id,
            })

            img_extracted_count += 1
            instance_pbar.update(1)

            if len(instances_pool) >= target_instances_count:
                break

        if len(instances_pool) >= target_instances_count:
            break

    img_pbar.close()
    instance_pbar.close()

    print(f"\n成功載入或提取了 {len(instances_pool)} 個實例！")
    return instances_pool


# =========================================================================
# 模組 B-3：基於 CLIP Vision 圖像特徵與 Cosine Similarity 的互斥配對模組
# （已加入同源過濾與 CLIP 相似度不能大於 0.95 的嚴格過濾）
# =========================================================================
def generate_semantic_similarity_pairs(pool, total_pairs=2000, similarity_threshold=0.6):
    """使用 CLIP Vision 萃取物件圖片的外觀特徵，計算 Cosine Similarity 進行互斥配對，
    同時過濾掉來自同一張原始 COCO 影像 (img_id 相同) 的配對組合，
    並且強制規定 CLIP 相似度不能大於 0.95 (若 >= 0.95 則直接排除)。
    """
    print(f"\n正在載入 CLIP Vision 模型，準備從 {len(pool)} 個實例的圖片外觀計算相似度...")
    required_instances = total_pairs * 2
    if len(pool) < required_instances:
        raise ValueError(f"實例池數量 ({len(pool)}) 不足以組成指定的 {total_pairs} 組（需要 {required_instances} 個）。")

    rng = random.Random(42)

    try:
        from transformers import CLIPModel, CLIPProcessor
        model_id = "openai/clip-vit-base-patch32"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在載入 CLIP 模型 ({model_id}) 至裝置: {device}...")
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
    except ImportError:
        raise ImportError("請先安裝 transformers 套件以支援 CLIP 視覺特徵計算 (執行: pip install transformers)")

    max_candidate_pool = min(len(pool), required_instances)
    selected_indices = rng.sample(range(len(pool)), max_candidate_pool)

    print(f"正在為 {max_candidate_pool} 個候選實例萃取圖像視覺特徵 (CLIP Vision)...")
    images = [cv2.cvtColor(pool[idx]["crop_img"], cv2.COLOR_BGR2RGB) for idx in selected_indices]

    batch_size = 32
    all_image_features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="CLIP 視覺特徵萃取進度"):
            batch_imgs = images[i:i+batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            all_image_features.append(features.cpu())

    image_features = torch.cat(all_image_features, dim=0).to(device)

    print("正在計算視覺特徵的 Cosine Similarity 矩陣並執行互斥配對（已啟動同源過濾與 CLIP 相似度 < 0.95 限制）...")
    sim_matrix = torch.mm(image_features, image_features.t())

    used_mask = set()
    pairs = []
    semantic_matched_count = 0

    all_pairs_candidates = []
    n_sel = len(selected_indices)
    for i in range(n_sel):
        for j in range(i + 1, n_sel):
            t_idx = selected_indices[i]
            o_idx = selected_indices[j]
            # 檢查是否來自同一張原始影像，若是則跳過此對子
            if pool[t_idx]["img_id"] == pool[o_idx]["img_id"]:
                continue
            score = sim_matrix[i, j].item()

            # 【核心修改點】：強制規定 CLIP 相似度不能大於 0.95 (大於等於 0.95 直接排除)
            if score >= 0.95:
                continue

            all_pairs_candidates.append((score, i, j))
    all_pairs_candidates.sort(key=lambda x: x[0], reverse=True)

    for score, local_i, local_j in all_pairs_candidates:
        if local_i in used_mask or local_j in used_mask:
            continue
        t_idx = selected_indices[local_i]
        o_idx = selected_indices[local_j]

        is_match = score >= similarity_threshold
        pairs.append((t_idx, o_idx, is_match))
        used_mask.add(local_i)
        used_mask.add(local_j)

        if is_match:
            semantic_matched_count += 1

        if len(pairs) >= total_pairs:
            break

    # 若尚未湊滿，從剩餘未使用的實例隨機補齊（同樣確保不同源且 CLIP 相似度 < 0.95）
    remaining_locals = [i for i in range(n_sel) if i not in used_mask]
    rng.shuffle(remaining_locals)
    i_ptr = 0
    while i_ptr < len(remaining_locals) and len(pairs) < total_pairs:
        local_i = remaining_locals[i_ptr]
        t_idx = selected_indices[local_i]
        # 尋找一個不同源且 CLIP 相似度 < 0.95 的配對對象
        paired = False
        for j_ptr in range(i_ptr + 1, len(remaining_locals)):
            local_j = remaining_locals[j_ptr]
            o_idx = selected_indices[local_j]
            score_val = sim_matrix[local_i, local_j].item()
            if pool[t_idx]["img_id"] != pool[o_idx]["img_id"] and score_val < 0.95:
                pairs.append((t_idx, o_idx, False))
                used_mask.add(local_i)
                used_mask.add(local_j)
                remaining_locals.pop(j_ptr)
                remaining_locals.pop(i_ptr)
                paired = True
                break
        if not paired:
            i_ptr += 1

    random_matched_count = len(pairs) - semantic_matched_count
    print(f"[統計] 視覺相似對數 (>= {similarity_threshold} 且 < 0.95)：{semantic_matched_count} 組")
    print(f"[統計] 隨機補齊對數：{random_matched_count} 組")
    print(f"總共成功產生 {len(pairs)} 組互斥、不同源且 CLIP 相似度 < 0.95 的配對（共使用 {len(pairs) * 2} 個唯一 instance）！")
    return pairs


# =========================================================================
# 模組 C：手動對齊 Bounding Box 並指定重疊率
# =========================================================================
def overlay_occluder_on_original_scene(
    target_dict, occluder_dict, overlap_ratio=0.45
):
    target_original_img = target_dict["original_img"].copy()
    target_full_mask = target_dict["full_mask"].copy()
    occluder_crop_img = occluder_dict["crop_img"]
    occluder_crop_mask = occluder_dict["crop_mask"]

    H, W = target_original_img.shape[:2]
    tx1, ty1, tx2, ty2 = mask_to_bbox(target_full_mask)
    tw, th = (tx2 - tx1), (ty2 - ty1)

    oh, ow = occluder_crop_img.shape[:2]
    scale_factor = float(th) / float(oh) if oh > 0 else 1.0
    new_ow = int(ow * scale_factor)
    new_oh = th

    if new_ow <= 0 or new_oh <= 0:
        return (
            target_original_img,
            target_full_mask,
            np.zeros_like(target_original_img),
            np.zeros((H, W), dtype=np.uint8),
        )

    o_img_res = cv2.resize(
        occluder_crop_img, (new_ow, new_oh), interpolation=cv2.INTER_LINEAR
    )
    o_mask_res = cv2.resize(
        occluder_crop_mask, (new_ow, new_oh), interpolation=cv2.INTER_NEAREST
    )

    direction = random.choice([0, 1, 2, 3])

    if direction == 0:
        paste_x1 = int(tx2 - tw * overlap_ratio)
        paste_y1 = ty1
    elif direction == 1:
        paste_x1 = int(tx1 - new_ow + tw * overlap_ratio)
        paste_y1 = ty1
    elif direction == 2:
        paste_x1 = tx1
        paste_y1 = int(ty1 - new_oh + th * overlap_ratio)
    else:
        paste_x1 = tx1
        paste_y1 = int(ty2 - th * overlap_ratio)

    paste_x2 = paste_x1 + new_ow
    paste_y2 = paste_y1 + new_oh

    occluder_layer = np.zeros_like(target_original_img)
    occluder_full_mask = np.zeros((H, W), dtype=np.uint8)

    valid_y1 = max(0, paste_y1)
    valid_y2 = min(H, paste_y2)
    valid_x1 = max(0, paste_x1)
    valid_x2 = min(W, paste_x2)

    crop_oy1 = valid_y1 - paste_y1
    crop_oy2 = crop_oy1 + (valid_y2 - valid_y1)
    crop_ox1 = valid_x1 - paste_x1
    crop_ox2 = crop_ox1 + (valid_x2 - valid_x1)

    if valid_y2 > valid_y1 and valid_x2 > valid_x1:
        occ_idx = o_mask_res[crop_oy1:crop_oy2, crop_ox1:crop_ox2] > 0
        sub_img = occluder_layer[valid_y1:valid_y2, valid_x1:valid_x2]
        sub_crop = o_img_res[crop_oy1:crop_oy2, crop_ox1:crop_ox2]
        sub_img[occ_idx] = sub_crop[occ_idx]
        occluder_full_mask[valid_y1:valid_y2, valid_x1:valid_x2][occ_idx] = 1

    return (
        target_original_img,
        target_full_mask,
        occluder_layer,
        occluder_full_mask,
    )


# =========================================================================
# 模組 D：主流程執行
# =========================================================================
if __name__ == "__main__":
    predictor = init_sam_refiner(weights_dir="weights")

    output_base_dir = "data/baseline/output_train_samples"
    os.makedirs(output_base_dir, exist_ok=True)

    pool = prepare_coco_samples(
        predictor, 
        download_dir="data/baseline/coco_train_raw", 
        output_base_dir=output_base_dir,
        num_imgs=80000,
        target_instances_count=4000
    )

    total_pairs = 2000  # 總共產生 2000 組（剛好 4000 個 instance）
    if len(pool) < total_pairs * 2:
        print(
            f"警告：實例池數量 ({len(pool)}) 少於 {total_pairs * 2}"
            " 個，將以現有實例進行最大化循環配對。"
        )

    random.seed(42)
    pairs = generate_semantic_similarity_pairs(pool, total_pairs=total_pairs, similarity_threshold=0.6)

    preview_data = []

    print(f"\n開始從重組後的實例池合成 {total_pairs} 組基於 CLIP 視覺特徵相似度的遮擋資料...")
    for pair_idx, (t_id, o_id, is_semantic_match) in enumerate(
        tqdm(pairs, desc="合成訓練集遮擋樣本"), start=2001
    ):
        target = pool[t_id]
        occluder = pool[o_id]

        random_overlap_ratio = round(random.uniform(0.15, 0.55), 2)

        t_img, t_mask, o_layer, o_mask = overlay_occluder_on_original_scene(
            target, occluder, overlap_ratio=random_overlap_ratio
        )

        hard_img, modal_mask, amodal_mask = synthesize_occlusion_pair(
            t_img, t_mask, o_layer, o_mask, use_taos=False
        )
        taos_img, _, _ = synthesize_occlusion_pair(
            t_img, t_mask, o_layer, o_mask, use_taos=True, k=9
        )

        amodal_box = mask_to_bbox(amodal_mask)
        modal_box = mask_to_bbox(modal_mask)
        occluder_box = mask_to_bbox(o_mask)

        sample_hard_dir = os.path.join(output_base_dir, f"sample_{pair_idx}_hard")
        sample_taos_dir = os.path.join(output_base_dir, f"sample_{pair_idx}_taos")
        os.makedirs(sample_hard_dir, exist_ok=True)
        os.makedirs(sample_taos_dir, exist_ok=True)

        sample_clean_dir = os.path.join(output_base_dir, f"sample_{pair_idx}_clean")
        os.makedirs(sample_clean_dir, exist_ok=True)

        cv2.imwrite(
            os.path.join(sample_clean_dir, "target_image.jpg"), target["original_img"]
        )
        cv2.imwrite(
            os.path.join(sample_clean_dir, "target_mask.png"),
            target["full_mask"] * 255,
        )
        cv2.imwrite(
            os.path.join(sample_clean_dir, "occluder_image.jpg"),
            occluder["original_img"],
        )
        cv2.imwrite(
            os.path.join(sample_clean_dir, "occluder_mask.png"),
            occluder["full_mask"] * 255,
        )

        original_metadata = {
            "target_info": {
                "category_id": int(target["category_id"]),
                "category_name": target["category_name"],
                "bbox": target["bbox"],
            },
            "occluder_info": {
                "category_id": int(occluder["category_id"]),
                "category_name": occluder["category_name"],
                "bbox": occluder["bbox"],
            },
        }
        with open(
            os.path.join(sample_clean_dir, "original_info.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(original_metadata, f, ensure_ascii=False, indent=4)

        dual_annotation_metadata = {
            "modal_box": modal_box,
            "amodal_box": amodal_box,
            "occluder_box": occluder_box,
            "overlap_ratio": random_overlap_ratio,
            "target_info": {
                "category_id": int(target["category_id"]),
                "category_name": target["category_name"],
            },
            "occluder_info": {
                "category_id": int(occluder["category_id"]),
                "category_name": occluder["category_name"],
            },
            "is_semantic_match": is_semantic_match
        }

        cv2.imwrite(os.path.join(sample_hard_dir, "image.jpg"), hard_img)
        cv2.imwrite(os.path.join(sample_hard_dir, "modal_mask.png"), modal_mask * 255)
        cv2.imwrite(
            os.path.join(sample_hard_dir, "amodal_mask.png"), amodal_mask * 255
        )
        cv2.imwrite(
            os.path.join(sample_hard_dir, "occluder_mask.png"), o_mask * 255
        )
        with open(
            os.path.join(sample_hard_dir, "box_prompt.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(dual_annotation_metadata, f, ensure_ascii=False, indent=4)

        cv2.imwrite(os.path.join(sample_taos_dir, "image.jpg"), taos_img)
        cv2.imwrite(os.path.join(sample_taos_dir, "modal_mask.png"), modal_mask * 255)
        cv2.imwrite(
            os.path.join(sample_taos_dir, "amodal_mask.png"), amodal_mask * 255
        )
        cv2.imwrite(
            os.path.join(sample_taos_dir, "occluder_mask.png"), o_mask * 255
        )
        with open(
            os.path.join(sample_taos_dir, "box_prompt.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(dual_annotation_metadata, f, ensure_ascii=False, indent=4)

        # 收集前 15 組用於繪製預覽大圖（因 start=2001，改用 len(preview_data) < 15 來判斷）
        if len(preview_data) < 15:
            preview_data.append((hard_img, taos_img, modal_mask, amodal_mask, is_semantic_match))

    print(f"\n正在繪製並輸出前 15 組訓練集成果展示圖...")
    fig, axes = plt.subplots(len(preview_data), 4, figsize=(18, 4 * len(preview_data)))
    col_titles = [
        "Hard Cut-Paste (Real Scene)",
        "TAOS Edge Blur (Real Scene)",
        "Modal Mask (Visible)",
        "Amodal Mask (GT)",
    ]
    for r in range(len(preview_data)):
        h_img, t_img, m_m, am_m, is_match = preview_data[r]
        h_img_disp = cv2.cvtColor(h_img, cv2.COLOR_BGR2RGB).copy()

        axes[r, 0].imshow(h_img_disp)
        axes[r, 1].imshow(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
        axes[r, 2].imshow(m_m, cmap="gray")
        axes[r, 3].imshow(am_m, cmap="gray")
        if r == 0:
            for c in range(4):
                axes[r, c].set_title(col_titles[c], fontsize=13, fontweight="bold")
        for c in range(4):
            axes[r, c].axis("off")

    plt.tight_layout()
    preview_output = "coco_train_taos_15_comparison_new.png"
    plt.savefig(preview_output, dpi=200)
    print(f"全部完成！成果大圖已儲存為： {preview_output}")

    print("\n正在自動生成全域資料集索引總表 (dataset_train_manifest.json)...")
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    manifest = []
    for sample_name in sorted(os.listdir(output_base_dir), key=natural_sort_key):
        sample_dir = os.path.join(output_base_dir, sample_name)
        if os.path.isdir(sample_dir):
            if "_hard" in sample_name or "_taos" in sample_name:
                json_path = os.path.join(sample_dir, "box_prompt.json")
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    manifest.append({
                        "sample_name": sample_name,
                        "type": "hard_or_taos",
                        "image_path": os.path.join(sample_dir, "image.jpg"),
                        "modal_mask_path": os.path.join(sample_dir, "modal_mask.png"),
                        "amodal_mask_path": os.path.join(sample_dir, "amodal_mask.png"),
                        "occluder_mask_path": os.path.join(
                            sample_dir, "occluder_mask.png"
                        ),
                        **metadata,
                    })
            elif "_clean" in sample_name:
                clean_json_path = os.path.join(sample_dir, "original_info.json")
                if os.path.exists(clean_json_path):
                    with open(clean_json_path, "r", encoding="utf-8") as f:
                        clean_metadata = json.load(f)
                    manifest.append({
                        "sample_name": sample_name,
                        "type": "clean_inputs",
                        "target_image_path": os.path.join(
                            sample_dir, "target_image.jpg"
                        ),
                        "target_mask_path": os.path.join(sample_dir, "target_mask.png"),
                        "occluder_image_path": os.path.join(
                            sample_dir, "occluder_image.jpg"
                        ),
                        "occluder_mask_path": os.path.join(
                            sample_dir, "occluder_mask.png"
                        ),
                        **clean_metadata,
                    })

    manifest_output = "dataset_train_manifest_new.json"
    with open(manifest_output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=4)
    print(f"訓練集總表建立完成！已儲存總表至：{manifest_output}")