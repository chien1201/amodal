import os
import json
import random
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pycocotools.coco import COCO

# 載入 Meta 官方 Segment Anything 套件
from segment_anything import sam_model_registry, SamPredictor

from dataset_generator import apply_taos_blur, synthesize_occlusion_pair

# =========================================================================
# 模組 A：Mask 轉 Bounding Box 為 prompt
# =========================================================================
def mask_to_bbox(mask):
    """
    從二值遮罩 [H, W] 計算出包含整個物件的最小外接矩形 [x1, y1, x2, y2]
    """
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
    """
    初始化 SAM 模型 (vit_b 輕量版) 作為高品質 Mask 提煉工具
    """
    os.makedirs(weights_dir, exist_ok=True)
    sam_checkpoint = os.path.join(weights_dir, "sam_vit_b_01ec64.pth")
    
    # 若本機無權重檔，自動自 Meta 官方載點下載 (約 375MB)
    if not os.path.exists(sam_checkpoint):
        print("正在下載 SAM 預訓練權重 (sam_vit_b_01ec64.pth，約 375MB)...")
        sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        urllib.request.urlretrieve(sam_url, sam_checkpoint)
        print("SAM 權重下載完成！")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在載入 SAM 模型至運算裝置: {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def refine_mask_with_sam(predictor, img_bgr, bbox):
    """
    輸入原始影像與粗略 Bounding Box，由 SAM 重新預測像素級貼合物體輪廓的 Mask
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    
    box_np = np.array(bbox)[None, :]  # Shape: (1, 4)
    masks, scores, _ = predictor.predict(
        box=box_np,
        multimask_output=False
    )
    refined_mask = masks[0].astype(np.uint8)
    return refined_mask

# =========================================================================
# 模組 B-2：COCO 資料讀取
# =========================================================================
def prepare_coco_samples(predictor, download_dir="data/coco_raw", num_imgs=10):
    """
    自動下載 COCO 驗證集標註檔與前 10 張真實影像
    並使用 SAM 重新提煉邊界，過濾背景噪聲
    """
    os.makedirs(download_dir, exist_ok=True)
    ann_path = os.path.join(download_dir, "instances_val2017.json")
    
    # 若本機無標註檔，從微軟官方輕量鏡像下載 val2017 標註 (壓縮後約 19MB)
    if not os.path.exists(ann_path):
        print("正在下載 COCO val2017 標註檔 (約 19MB)...")
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = os.path.join(download_dir, "annotations.zip")
        urllib.request.urlretrieve(ann_url, zip_path)
        import zipfile

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
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
    print(f"正在讀取 COCO 前 {num_imgs} 張圖片並提取獨立物件...")
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        local_img_path = os.path.join(download_dir, img_info['file_name'])
        
        # 下載該張真實照片
        if not os.path.exists(local_img_path):
            urllib.request.urlretrieve(img_url, local_img_path)
            
        img_bgr = cv2.imread(local_img_path)
        H, W = img_bgr.shape[:2]
        
        # 取得該圖片中的所有實例標註
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            # 篩選掉過小 (<5000 像素) 或非封閉多邊形的無效遮罩
            if ann['area'] < 5000 or not isinstance(ann['segmentation'], list):
                continue
                
            mask = coco.annToMask(ann) # 轉換為 [H, W] 二值圖 (0 或 1)
            coarse_box = mask_to_bbox(mask)

            x1, y1, x2, y2  = coarse_box
            
            # 確保物件寬高合理
            if (x2 - x1) < 50 or (y2 - y1) < 50:
                continue

            # SAM 邊界精細化
            refined_mask = refine_mask_with_sam(predictor, img_bgr, coarse_box)
            ref_x1, ref_y1, ref_x2, ref_y2 = mask_to_bbox(refined_mask)

            if (ref_x2 - ref_x1) < 40 or (ref_y2 - ref_y1) < 40:
                continue
            
            instances_pool.append({
                "original_img": img_bgr,              # 完整原圖 (含真實背景)
                "full_mask": refined_mask,                    # 物件在全圖中的位置 Mask
                "crop_img": img_bgr[ref_y1:ref_y2, ref_x1:ref_x2],    # 僅物件本身的彩色塊
                "crop_mask": refined_mask[ref_y1:ref_y2, ref_x1:ref_x2],      # 僅物件本身的 Mask
            })
            
            # 每張圖最多只抽取前 1~2 個清晰物體，避免背景重複
            if len(instances_pool) >= 10:
                break
        if len(instances_pool) >= 10:
            break
            
    print(f"成功提取了 {len(instances_pool)} 個實例（含真實背景！")
    return instances_pool

# =========================================================================
# 模組 C：手動對齊 Bounding Box 並指定重疊率
# =========================================================================
def overlay_occluder_on_original_scene(target_dict, occluder_dict, overlap_ratio=0.45):
    """
    手動對齊 Bounding Box：將 Occluder 尺度調整到與 Target 相符，
    並依指定的 overlap_ratio 覆蓋在 Target 上
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
    scale_factor = float(th) / float(oh)
    new_ow = int(ow * scale_factor)
    new_oh = th

    o_img_res = cv2.resize(
        occluder_crop_img, (new_ow, new_oh), interpolation=cv2.INTER_LINEAR
    )
    o_mask_res = cv2.resize(
          occluder_crop_mask, (new_ow, new_oh), interpolation=cv2.INTER_NEAREST
    )

    # 貼在 Target 右側，向左切入 overlap_ratio
    paste_x1 = int(tx2 - tw * overlap_ratio)
    paste_y1 = ty1
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

    # 1. 取得 10 個真實 COCO 實例
    pool = prepare_coco_samples(predictor, download_dir="data/coco_raw", num_imgs=10)
    
    output_base_dir = "data/output_samples"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 隨機打亂並兩兩配對，共 5 組對子
    random.seed(42)
    indices = list(range(len(pool)))
    random.shuffle(indices)
    pairs = [(indices[i*2], indices[i*2+1]) for i in range(5)]
    
    preview_data = []

    print("\n開始合成 5 組遮擋資料 (含 Hard 與 TAOS 雙版本)...")
    for pair_idx, (t_id, o_id) in enumerate(pairs, start=1):
        target = pool[t_id]
        occluder = pool[o_id]
        
        # 對齊放置物件（指定覆蓋約 45% 寬度）
        t_img, t_mask, o_layer, o_mask = overlay_occluder_on_original_scene(
            target, occluder, overlap_ratio=0.45
        )
        
        # 呼叫你現有的 synthesize_occlusion_pair：分別產出 Hard 與 TAOS
        hard_img, modal_mask, amodal_mask = synthesize_occlusion_pair(
            t_img, t_mask, o_layer, o_mask, use_taos=False
        )
        taos_img, _, _ = synthesize_occlusion_pair(
            t_img, t_mask, o_layer, o_mask, use_taos=True, k=9
        )
        
        # 計算 Box Prompts
        amodal_box = mask_to_bbox(amodal_mask)
        modal_box = mask_to_bbox(modal_mask)
        
        # 建立目錄並存檔
        sample_hard_dir = os.path.join(output_base_dir, f"sample_{pair_idx}_hard")
        sample_taos_dir = os.path.join(output_base_dir, f"sample_{pair_idx}_taos")
        os.makedirs(sample_hard_dir, exist_ok=True)
        os.makedirs(sample_taos_dir, exist_ok=True)
        
        # 儲存 Hard 版
        cv2.imwrite(os.path.join(sample_hard_dir, "image.jpg"), hard_img)
        cv2.imwrite(os.path.join(sample_hard_dir, "amodal_mask.png"), amodal_mask * 255)
        with open(os.path.join(sample_hard_dir, "box_prompt.json"), "w") as f:
            json.dump({"modal_box": modal_box, "amodal_box": amodal_box}, f)
            
        # 儲存 TAOS 版
        cv2.imwrite(os.path.join(sample_taos_dir, "image.jpg"), taos_img)
        cv2.imwrite(os.path.join(sample_taos_dir, "amodal_mask.png"), amodal_mask * 255)
        with open(os.path.join(sample_taos_dir, "box_prompt.json"), "w") as f:
            json.dump({"modal_box": modal_box, "amodal_box": amodal_box}, f)
            
        preview_data.append((hard_img, taos_img, modal_mask, amodal_mask))
        print(f"  - Pair #{pair_idx} 完成！Amodal Box: {amodal_box}, Modal Box: {modal_box}")

    # =========================================================================
    # 模組 E：自動輸出拼貼成果圖 (供報告使用)
    # =========================================================================
    print("\n正在繪製並輸出成果展示圖...")
    fig, axes = plt.subplots(5, 4, figsize=(18, 20))
    
    col_titles = [
        "Hard Cut-Paste (Real Scene)",
        "TAOS Edge Blur (Real Scene)",
        "Modal Mask (Visible)",
        "Amodal Mask (GT)",
    ]
    
    for r in range(5):
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
    preview_output = "coco_taos_5pairs_comparison.png"
    plt.savefig(preview_output, dpi=200)
    print(f"全部完成！成果大圖已儲存為：{preview_output}")