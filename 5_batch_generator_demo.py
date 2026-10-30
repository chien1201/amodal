import json
import os
import random
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pycocotools.coco import COCO
from tqdm import tqdm  # 引入進度條工具

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
# 模組 B-1：SAM 提煉器初始化 (方案 A 核心)
# =========================================================================
def init_sam_refiner(weights_dir="weights"):
  """初始化 SAM 模型 (vit_b 輕量版) 作為高品質 Mask 提煉工具"""
  os.makedirs(weights_dir, exist_ok=True)
  sam_checkpoint = os.path.join(weights_dir, "sam_vit_b_01ec64.pth")
  # 若本機無權重檔，自動自 Meta 官方載點下載 (約 375MB)
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
  refined_mask = masks[0].astype(np.uint8)
  return refined_mask


# =========================================================================
# 模組 B-2：COCO 驗證集資料讀取與高完整度濾鏡實作 (每張圖上限 3 個)
# =========================================================================
def prepare_coco_samples(
    predictor, download_dir="data/baseline/coco_val_raw", num_imgs=50
):
  """自動下載 COCO 驗證集標註檔與前 num_imgs 張真實影像

  並套用極高完整度過濾條件：嚴格排除邊緣切斷、形狀不扎實或已被遮擋的物件
  （限制每張圖片最多取 3 個實例）
  """
  os.makedirs(download_dir, exist_ok=True)
  ann_path = os.path.join(download_dir, "instances_val2017.json")

  if not os.path.exists(ann_path):
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = os.path.join(download_dir, "annotations.zip")
    download_with_progress(ann_url, zip_path, desc="下載 COCO trainval2017 標註檔")
    import zipfile
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
      zip_ref.extract("annotations/instances_val2017.json", download_dir)
    os.rename(
        os.path.join(download_dir, "annotations/instances_val2017.json"),
        ann_path,
    )
    if os.path.exists(zip_path):
      os.remove(zip_path)

  coco = COCO(ann_path)
  img_ids = coco.getImgIds()[:num_imgs]
  instances_pool = []
  target_instances_count = 10  # 5 對需要 10 個實例
  print(
      f"正在讀取 COCO 驗證集前 {num_imgs}"
      f" 張圖片，並啟用【100% 完全完整實例】嚴格過濾 (每張圖上限 3 個)..."
  )

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

  for img_id in tqdm(img_ids, desc="處理 COCO 驗證集影像"):
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info["coco_url"]
    local_img_path = os.path.join(download_dir, img_info["file_name"])

    if not os.path.exists(local_img_path):
      urllib.request.urlretrieve(img_url, local_img_path)
    img_bgr = cv2.imread(local_img_path)
    if img_bgr is None:
      continue
    H, W = img_bgr.shape[:2]
    image_area = float(H * W)

    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    img_extracted_count = 0

    for ann in anns:
      if img_extracted_count >= 3:  # 每張圖最多取 3 個實例
        break

      cat_id = ann["category_id"]
      cat_name = coco.loadCats(cat_id)[0]["name"].lower()
      if any(bad_word in cat_name for bad_word in stuff_blacklist):
        continue

      if ann["area"] < 5000 or not isinstance(ann["segmentation"], list):
        continue

      if ann["area"] / image_area > 0.9:
        continue

      mask = coco.annToMask(ann)
      coarse_box = mask_to_bbox(mask)
      x1, y1, x2, y2 = coarse_box
      box_width = x2 - x1
      box_height = y2 - y1

      if box_width < 50 or box_height < 50:
        continue

      # =========================================================================
      # 【防護 1：嚴格邊緣安全距離檢查】確保物件距離圖片邊緣至少 10 個像素
      # =========================================================================
      margin = 10
      if x1 <= margin or y1 <= margin or x2 >= (W - margin) or y2 >= (H - margin):
        continue

      # SAM 邊界精細化
      refined_mask = refine_mask_with_sam(predictor, img_bgr, coarse_box)
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

      # =========================================================================
      # 【防護 2：極高實體扎實度檢查】確保物件在邊界內扎實飽滿（排除殘缺、鏤空過大或被遮擋）
      # =========================================================================
      fill_ratio = refined_mask_area / box_area
      if fill_ratio < 0.50:  # 提高填充率門檻至 50% 以上，確保形狀非常完整
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

      if len(instances_pool) >= target_instances_count:
        break
    if len(instances_pool) >= target_instances_count:
      break

  print(
      f"嚴格過濾後，成功提取了 {len(instances_pool)}"
      f" 個 100% 完全完整且扎實的實例！"
  )
  return instances_pool


# =========================================================================
# 模組 C：手動對齊 Bounding Box 並指定重疊率
# =========================================================================
def overlay_occluder_on_original_scene(
    target_dict, occluder_dict, overlap_ratio=0.45
):
  """手動對齊 Bounding Box：將 Occluder 尺度調整到與 Target 相符，

  並依指定的 overlap_ratio 與隨機方向覆蓋在 Target 上
  """
  target_original_img = target_dict["original_img"].copy()
  target_full_mask = target_dict["full_mask"].copy()
  occluder_crop_img = occluder_dict["crop_img"]
  occluder_crop_mask = occluder_dict["crop_mask"]

  H, W = target_original_img.shape[:2]
  tx1, ty1, tx2, ty2 = mask_to_bbox(target_full_mask)
  tw, th = (tx2 - tx1), (ty2 - ty1)

  # 縮放 Occluder 符合 Target 高度
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

  # 隨機決定從哪個方向切入 (0: 右側切入, 1: 左側切入, 2: 上方切入, 3: 下方切入)
  direction = random.choice([0, 1, 2, 3])

  if direction == 0:  # 從右側向左覆蓋
    paste_x1 = int(tx2 - tw * overlap_ratio)
    paste_y1 = ty1
  elif direction == 1:  # 從左側向右覆蓋
    paste_x1 = int(tx1 - new_ow + tw * overlap_ratio)
    paste_y1 = ty1
  elif direction == 2:  # 從上方向下覆蓋
    paste_x1 = tx1
    paste_y1 = int(ty1 - new_oh + th * overlap_ratio)
  else:  # 從下方向上覆蓋
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
# 模組 D：主流程執行 (5組成對生成 + Hard/TAOS 雙版本保存 + 視覺化拼貼)
# =========================================================================
if __name__ == "__main__":

  # 0. 初始化 SAM 提煉器
  predictor = init_sam_refiner(weights_dir="weights")

  # 1. 取得經過極高完整度過濾條件篩選後的真實 COCO 驗證集實例影像 (設定抓取前 50 張)
  pool = prepare_coco_samples(
      predictor, download_dir="data/baseline/coco_val_raw", num_imgs=50
  )
  output_base_dir = "data/baseline/output_val_samples"
  os.makedirs(output_base_dir, exist_ok=True)

  # 確保實例池足夠組成 5 對 (需要 10 個實例)
  total_pairs = 5
  if len(pool) < total_pairs * 2:
    print(
        f"警告：實例池數量 ({len(pool)}) 少於 10 個，將以現有實例進行最大化循環配對。"
    )

  random.seed(42)
  indices = list(range(len(pool)))
  random.shuffle(indices)

  # 產生 5 組配對索引
  pairs = []
  for i in range(total_pairs):
    t_idx = indices[(i * 2) % len(indices)]
    o_idx = indices[(i * 2 + 1) % len(indices)]
    if t_idx == o_idx:
      o_idx = indices[(i * 2 + 2) % len(indices)]
    pairs.append((t_idx, o_idx))

  preview_data = []

  print(f"\n開始從驗證集合成 {total_pairs} 組遮擋資料 (含 Hard 與 TAOS 雙版本)...")
  for pair_idx, (t_id, o_id) in enumerate(
      tqdm(pairs, desc="合成驗證集遮擋樣本"), start=1
  ):
    target = pool[t_id]
    occluder = pool[o_id]

    # 將遮擋比例在 15% 至 55% 之間隨機抽樣 (0.15 ~ 0.55)
    random_overlap_ratio = round(random.uniform(0.15, 0.55), 2)

    t_img, t_mask, o_layer, o_mask = overlay_occluder_on_original_scene(
        target, occluder, overlap_ratio=random_overlap_ratio
    )

    # 呼叫 synthesize_occlusion_pair：分別產出 Hard 與 TAOS
    hard_img, modal_mask, amodal_mask = synthesize_occlusion_pair(
        t_img, t_mask, o_layer, o_mask, use_taos=False
    )
    taos_img, _, _ = synthesize_occlusion_pair(
        t_img, t_mask, o_layer, o_mask, use_taos=True, k=9
    )

    # 計算 Box Prompts
    amodal_box = mask_to_bbox(amodal_mask)
    modal_box = mask_to_bbox(modal_mask)
    occluder_box = mask_to_bbox(o_mask)

    # 建立目錄並存檔
    sample_hard_dir = os.path.join(output_base_dir, f"sample_{pair_idx}_hard")
    sample_taos_dir = os.path.join(output_base_dir, f"sample_{pair_idx}_taos")
    os.makedirs(sample_hard_dir, exist_ok=True)
    os.makedirs(sample_taos_dir, exist_ok=True)

    # =====================================================================
    # 獨立開一個 clean_inputs 資料夾
    # =====================================================================
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
    # =====================================================================

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
    }

    # 儲存 Hard 版
    cv2.imwrite(os.path.join(sample_hard_dir, "image.jpg"), hard_img)
    cv2.imwrite(os.path.join(sample_hard_dir, "modal_mask.png"), modal_mask * 255)
    cv2.imwrite(
        os.path.join(sample_hard_dir, "amodal_mask.png"), amodal_mask * 255
    )
    cv2.imwrite(
        os.path.join(sample_hard_dir, "occluder_mask.png"), o_mask * 255
    )
    with open(
        os.path.join(sample_hard_dir, "box_prompt.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(dual_annotation_metadata, f, ensure_ascii=False, indent=4)

    # 儲存 TAOS 版
    cv2.imwrite(os.path.join(sample_taos_dir, "image.jpg"), taos_img)
    cv2.imwrite(os.path.join(sample_taos_dir, "modal_mask.png"), modal_mask * 255)
    cv2.imwrite(
        os.path.join(sample_taos_dir, "amodal_mask.png"), amodal_mask * 255
    )
    cv2.imwrite(
        os.path.join(sample_taos_dir, "occluder_mask.png"), o_mask * 255
    )
    with open(
        os.path.join(sample_taos_dir, "box_prompt.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(dual_annotation_metadata, f, ensure_ascii=False, indent=4)

    preview_data.append((hard_img, taos_img, modal_mask, amodal_mask))

  # =========================================================================
  # 模組 E：自動輸出拼貼成果圖 (畫出全部 5 組)
  # =========================================================================
  print("\n正在繪製並輸出 5 組驗證集成果展示圖...")
  fig, axes = plt.subplots(5, 4, figsize=(18, 20))
  col_titles = [
      "Hard Cut-Paste (Real Scene)",
      "TAOS Edge Blur (Real Scene)",
      "Modal Mask (Visible)",
      "Amodal Mask (GT)",
  ]
  for r in range(len(preview_data)):
    h_img, t_img, m_m, am_m = preview_data[r]
    axes[r, 0].imshow(cv2.cvtColor(h_img, cv2.COLOR_BGR2RGB))
    axes[r, 1].imshow(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
    axes[r, 2].imshow(m_m, cmap="gray")
    axes[r, 3].imshow(am_m, cmap="gray")
    if r == 0:
      for c in range(4):
        axes[r, c].set_title(col_titles[c], fontsize=13, fontweight="bold")
    for c in range(4):
      axes[r, c].axis("off")

  plt.tight_layout()
  preview_output = "coco_val_taos_5_comparison.png"
  plt.savefig(preview_output, dpi=200)
  print(
      f"全部完成！5 組驗證集樣本已完整生成，成果大圖已儲存為： {preview_output}"
  )

  # =========================================================================
  # 模組 F：自動產生全域資料集索引總表
  # =========================================================================
  print("\n正在自動生成全域資料集索引總表 (dataset_val_manifest.json)...")
  manifest = []
  for sample_name in sorted(os.listdir(output_base_dir)):
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
              "occluder_mask_path": os.path.join(sample_dir, "occluder_mask.png"),
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

  manifest_output = "dataset_val_manifest.json"
  with open(manifest_output, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=4)
  print(f"驗證集總表建立完成！已儲存至：{manifest_output}")