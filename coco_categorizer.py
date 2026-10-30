#!/usr/bin/env python3
"""
coco_categorizer.py
自動讀取 COCO 資料集標註檔並生成分類對照表
"""

import json
import argparse
import sys

def get_coco_category_groups(ann_file):
    try:
        with open(ann_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{ann_file}'，請確認路徑是否正確。", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"錯誤：檔案 '{ann_file}' 不是有效的 JSON 格式。", file=sys.stderr)
        sys.exit(1)
    
    grouped_categories = {}
    for cat in coco_data.get('categories', []):
        super_cat = cat.get('supercategory', 'unknown')
        cat_name = cat.get('name')
        
        if super_cat not in grouped_categories:
            grouped_categories[super_cat] = []
        if cat_name and cat_name not in grouped_categories[super_cat]:
            grouped_categories[super_cat].append(cat_name)
        
    return grouped_categories

def main():
    parser = argparse.ArgumentParser(description="自動從 COCO 標註檔提取並分組類別")
    parser.add_argument("ann_file", help="COCO 標註檔的路徑 (例如 instances_val2017.json)")
    parser.add_argument("-o", "--output", help="將結果輸出儲存為 JSON 檔案的路徑 (選填)", default=None)
    
    args = parser.parse_args()
    
    # 執行分組
    grouped = get_coco_category_groups(args.ann_file)
    
    # 輸出結果
    output_json = json.dumps(grouped, indent=4, ensure_ascii=False)
    print(output_json)
    
    # 如果指定了輸出檔名，則存檔
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_json)
        print(f"\n[成功] 分類對照表已儲存至：{args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()