#!/usr/bin/env python3
"""
PHAROS epirotic dataset -> PaddleX "COCOInstSegDataset" 形式への変換 (PP-DocLayoutV3用)

確定仕様 (paddlex 3.7.2, paddlex/modules/layout_analysis/ を直接確認済み):
  - 出力ディレクトリ構造:
      <out_dir>/images/<file_name>
      <out_dir>/annotations/instance_train.json
      <out_dir>/annotations/instance_val.json
  - COCO instance segmentation形式 (segmentation = polygon必須)
  - 各annotationに非負整数の "read_order" フィールドが必須
  - read_orderは画像ごとに 0,1,2,...,N-1 の連番でなければならない
  - categoriesはCOCO jsonからそのまま読まれる (coco.getCatIds())
      -> あなたの11カテゴリーをid 0..10のままそのまま使う。マッピング不要。

入力:
  <dataset_root>/selected.txt
    形式: INT1 - STR1 - INT2 - STR2 - INT3
      INT1: 0/1 フラグ (1のみ採用)
      STR1: 無視
      INT2: フォルダー名 (例: "000")
      STR2: サブフォルダー名 (例: "0013314efa2649acb840")
      INT3: pagesフォルダー内のjpgファイル名 (拡張子なし)

  画像: <dataset_root>/<INT2>/<STR2>/pages/<INT3>.jpg
  GT  : <dataset_root>/<INT2>/<STR2>/bbox_gt/<INT3>.txt
    各行: category_id x1 y1 x2 y2 ... xn yn  (polygon、ファイル内の行順=読み取り順)

使い方:
  python convert_to_coco.py \
      --dataset_root /media/StorageServer/PHAROS/pharos_epirotic \
      --out_dir /media/SSD/vl16/dataset_pharos \
      --val_ratio 0.15 \
      --seed 42
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

from PIL import Image

CATEGORY_NAMES = {
    0: "Text",
    1: "Header",
    2: "Paragraph Title",
    3: "Image",
    4: "Table",
    5: "Formula",
    6: "Page Number",
    7: "Document Title",
    8: "Footnote",
    9: "Caption",
    10: "Document Info",
}


def parse_selected(selected_path):
    entries = []
    with open(selected_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("-")]
            if len(parts) != 5:
                print(f"WARNING: selected.txt 行{lineno}が想定外の形式: {raw!r} -> skip",
                      file=sys.stderr)
                continue
            int1, _str1, int2, str2, int3 = parts
            try:
                int1 = int(int1)
            except ValueError:
                print(f"WARNING: selected.txt 行{lineno} INT1が数値でない: {raw!r} -> skip",
                      file=sys.stderr)
                continue
            if int1 != 1:
                continue
            entries.append((int2, str2, int3))
    return entries


def parse_gt_file(gt_path):
    """[(category_id, bbox_xywh, segmentation_polygon, read_order), ...] を返す。
    read_orderはファイル内の行の出現順そのまま (0始まり)。
    """
    anns = []
    read_order = 0
    with open(gt_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            toks = line.split()
            cat_id = int(toks[0])
            if cat_id not in CATEGORY_NAMES:
                print(f"WARNING: {gt_path} 行{lineno} 未知のcategory_id={cat_id} -> skip",
                      file=sys.stderr)
                continue
            coords = [float(x) for x in toks[1:]]
            if len(coords) < 6 or len(coords) % 2 != 0:
                print(f"WARNING: {gt_path} 行{lineno} 座標数が不正 ({len(coords)}個) -> skip",
                      file=sys.stderr)
                continue
            xs = coords[0::2]
            ys = coords[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            bbox = [x1, y1, x2 - x1, y2 - y1]
            anns.append((cat_id, bbox, coords, read_order))
            read_order += 1
    return anns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    categories = [{"id": cid, "name": name} for cid, name in sorted(CATEGORY_NAMES.items())]
    print(f"カテゴリー ({len(categories)}): {categories}")

    entries = parse_selected(os.path.join(args.dataset_root, "selected.txt"))
    print(f"selected.txt: 有効行(INT1==1) {len(entries)} 件")

    images_out_dir = os.path.join(args.out_dir, "images")
    annotations_out_dir = os.path.join(args.out_dir, "annotations")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(annotations_out_dir, exist_ok=True)

    # ドキュメント単位 (int2, str2) でtrain/val分割 (同一文書のページがtrain/valに分かれて
    # リークするのを防ぐ)
    by_doc = defaultdict(list)
    for int2, str2, int3 in entries:
        by_doc[(int2, str2)].append(int3)

    doc_keys = sorted(by_doc.keys())
    random.Random(args.seed).shuffle(doc_keys)
    n_val_docs = max(1, round(len(doc_keys) * args.val_ratio))
    val_docs = set(doc_keys[:n_val_docs])
    print(f"文書数: 全{len(doc_keys)} / train={len(doc_keys) - n_val_docs} / val={n_val_docs}")

    splits = {"train": [], "val": []}
    for (int2, str2), pages in by_doc.items():
        split = "val" if (int2, str2) in val_docs else "train"
        for int3 in pages:
            splits[split].append((int2, str2, int3))

    next_image_id = 1
    next_ann_id = 1
    n_missing_img = 0
    n_missing_gt = 0
    n_invalid_read_order_imgs = 0

    for split_name, items in splits.items():
        images_json = []
        anns_json = []
        for int2, str2, int3 in items:
            img_path = os.path.join(args.dataset_root, int2, str2, "pages", f"{int3}.jpg")
            gt_path = os.path.join(args.dataset_root, int2, str2, "bbox_gt", f"{int3}.txt")

            if not os.path.isfile(img_path):
                print(f"WARNING: 画像が見つかりません: {img_path} -> skip", file=sys.stderr)
                n_missing_img += 1
                continue
            if not os.path.isfile(gt_path):
                print(f"WARNING: GTが見つかりません: {gt_path} -> skip", file=sys.stderr)
                n_missing_gt += 1
                continue

            try:
                with Image.open(img_path) as im:
                    width, height = im.size
            except Exception as e:
                print(f"WARNING: 画像を開けません: {img_path} ({e}) -> skip", file=sys.stderr)
                continue

            rel_name = f"{int2}__{str2}__{int3}.jpg"
            link_path = os.path.join(images_out_dir, rel_name)
            if not os.path.exists(link_path):
                os.symlink(os.path.abspath(img_path), link_path)

            anns = parse_gt_file(gt_path)
            if not anns:
                # アノテーション0件の画像はPaddleXのread_order検証でエラーになりうるので除外
                print(f"WARNING: アノテーションが0件: {gt_path} -> 画像ごとskip", file=sys.stderr)
                continue

            # read_orderの連番性を最終確認 (parse_gt_fileで既に0始まり連番のはずだが、
            # skipされた行があるとズレるので、ここで詰め直す)
            anns = [(cat_id, bbox, seg, i) for i, (cat_id, bbox, seg, _old_order) in enumerate(anns)]

            image_id = next_image_id
            next_image_id += 1
            images_json.append({
                "id": image_id,
                "file_name": rel_name,  # NOTE: "images/"は付けない (paddlex側でimages/と結合される)
                "width": width,
                "height": height,
            })

            for cat_id, bbox, seg, read_order in anns:
                x, y, w, h = bbox
                area = max(w, 0) * max(h, 0)
                anns_json.append({
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "segmentation": [seg],
                    "area": area,
                    "iscrowd": 0,
                    "read_order": read_order,
                })
                next_ann_id += 1

        coco = {
            "info": {"description": "PHAROS epirotic dataset", "year": 2026},
            "licenses": [{"id": 0, "name": None, "url": None}],
            "images": images_json,
            "annotations": anns_json,
            "categories": categories,
        }
        out_path = os.path.join(annotations_out_dir, f"instance_{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f"{split_name}: images={len(images_json)}, annotations={len(anns_json)} -> {out_path}")

    print(f"missing images: {n_missing_img}, missing gt: {n_missing_gt}")
    print("完了。次は paddlex の check_dataset を実行してください。")


if __name__ == "__main__":
    main()
