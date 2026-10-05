#!/usr/bin/env python3
"""
run_v4_on_folder.py

フォルダ内の全画像にv4モデルを実行し、画像ごとにフォルダを作って、
  - <画像名>.txt      : ポリゴン＋カテゴリ（bbox_gtと同じ形式）
  - <画像名>_vis.jpg  : ポリゴンと読み順(1〜N)を描画した画像
を保存する。

txtの形式（1行 = 1ブロック、bbox_gtと同じ）:
    class_id x1 y1 x2 y2 x3 y3 ... xn yn
  例) 3 1 0 1649 0 1649 1906 1 1906
  - class_idはモデルのクラスID（0 Text, 1 Header, 2 Paragraph Title,
    3 Image, 4 Table, 5 Formula, 6 Page Number, 7 Document Title,
    8 Footnote, 9 Caption, 10 Document Info）
  - 行の順番がそのまま読み順（1行目が読み順1番）
  - 座標は整数に丸め、画像の範囲内にクリップ済み

使い方:
    python run_v4_on_folder.py /media/StorageServer/PHAROS/sample_360/
    python run_v4_on_folder.py /path/to/images --out_dir /path/to/output --threshold 0.30
"""

import argparse
import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# ====================================================================
# 設定
# ====================================================================

V4_SAFETENSORS_DIR = "/media/SSD/vl16/output_pharos_v4/safetensors_v4"
DEFAULT_THRESHOLD = 0.25
NMS_IOU_THRESH = 0.5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

COLORS = {
    "Text": (220, 50, 50), "Header": (50, 200, 50), "Paragraph Title": (50, 50, 220),
    "Image": (200, 200, 0), "Table": (180, 0, 180), "Formula": (0, 200, 200),
    "Page Number": (120, 0, 220), "Document Title": (220, 120, 0),
    "Footnote": (0, 120, 220), "Caption": (120, 220, 0), "Document Info": (220, 0, 120),
}


# ====================================================================
# ヘルパー
# ====================================================================

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes, iou_thresh: float = NMS_IOU_THRESH):
    """クラスを区別しない貪欲NMS（重複検出の除去）。"""
    if not boxes:
        return boxes
    kept = []
    for b in sorted(boxes, key=lambda b: -b["score"]):
        if not any(iou_xyxy(b["coordinate"], k["coordinate"]) >= iou_thresh for k in kept):
            kept.append(b)
    return kept


def renumber_reading_order(boxes):
    """生のorder値で並べ替え、1〜Nの連続した読み順(display_order)を振り直す。"""
    def sort_key(b):
        o = b.get("order")
        return (o is None, o if o is not None else 0, -b.get("score", 0))

    boxes_sorted = sorted(boxes, key=sort_key)
    for i, b in enumerate(boxes_sorted, start=1):
        b["display_order"] = i
    return boxes_sorted


def clip_polygon(points, width, height):
    """ポリゴンの各点を整数に丸め、画像の範囲内(0〜w-1, 0〜h-1)にクリップする。"""
    pts = np.array(points, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] = np.clip(np.round(pts[:, 0]), 0, width - 1)
    pts[:, 1] = np.clip(np.round(pts[:, 1]), 0, height - 1)
    return pts.astype(np.int32)


def box_to_polygon(coordinate):
    """ポリゴンが取れなかった場合のフォールバック: 矩形の4隅。"""
    x1, y1, x2, y2 = coordinate
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ====================================================================
# 推論
# ====================================================================

def predict(processor, model, device, image_path: Path, threshold: float):
    """1枚の画像を推論し、NMS後・読み順振り直し済みの検出リストを返す。"""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(height, width)], threshold=threshold
    )[0]

    order_seq = results.get("order_seq")
    boxes = []
    for i, (score, label_id, box, polygon) in enumerate(zip(
        results["scores"], results["labels"], results["boxes"], results["polygon_points"]
    )):
        coordinate = [float(v) for v in box.tolist()]
        poly = polygon.tolist() if hasattr(polygon, "tolist") else polygon
        if poly is None or len(poly) < 3:
            poly = box_to_polygon(coordinate)

        boxes.append({
            "class_id": int(label_id.item()),
            "label": model.config.id2label[label_id.item()],
            "score": float(score.item()),
            "coordinate": coordinate,
            "polygon": clip_polygon(poly, width, height),
            "order": order_seq[i].item() if order_seq is not None else None,
        })

    boxes = nms_boxes(boxes)
    boxes = renumber_reading_order(boxes)
    return boxes, (width, height)


# ====================================================================
# 出力（txt / 画像）
# ====================================================================

def write_polygon_txt(boxes, txt_path: Path):
    """1行1ブロック: class_id x1 y1 x2 y2 ... （読み順に並べて出力）"""
    with open(txt_path, "w", encoding="utf-8") as f:
        for b in boxes:  # renumber_reading_order済み＝読み順
            coords = " ".join(str(int(v)) for v in b["polygon"].flatten())
            f.write(f"{b['class_id']} {coords}\n")


def draw_visualization(image_path: Path, boxes, out_path: Path):
    img = cv2.imread(str(image_path))
    for b in boxes:
        color = COLORS.get(b["label"], (128, 128, 128))
        pts = b["polygon"].reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)

        x1, y1 = int(pts[0][0][0]), int(pts[0][0][1])
        text = f"{b['display_order']}:{b['label']}({b['score']:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 2, max(y1, th + 6)), color, -1)
        cv2.putText(img, text, (x1 + 1, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), img)


# ====================================================================
# メイン
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="フォルダ内の全画像にv4モデルを実行し、画像ごとにポリゴンtxtと可視化画像を保存する"
    )
    parser.add_argument("input_dir", type=str, help="画像が入っているフォルダ")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="出力先の親フォルダ（省略時は入力フォルダの中に画像名のフォルダを作る）")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help=f"信頼度しきい値 (デフォルト: {DEFAULT_THRESHOLD})")
    parser.add_argument("--skip_existing", action="store_true",
                         help="出力のtxtが既にある画像はスキップする")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"フォルダが見つかりません: {input_dir}")
    out_root = Path(args.out_dir) if args.out_dir else input_dir

    images = sorted(p for p in input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"画像が見つかりません: {input_dir}")
        return
    print(f"画像 {len(images)} 枚を処理します  (しきい値={args.threshold})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(V4_SAFETENSORS_DIR)
    model = AutoModelForObjectDetection.from_pretrained(V4_SAFETENSORS_DIR)
    model.to(device)
    model.eval()
    print(f"モデルロード: {time.time() - t0:.1f}s  (device={device})")

    n_ok, n_skip, failed = 0, 0, []
    for idx, image_path in enumerate(images, start=1):
        stem = image_path.stem
        out_folder = out_root / stem
        txt_path = out_folder / f"{stem}.txt"
        vis_path = out_folder / f"{stem}_vis.jpg"

        if args.skip_existing and txt_path.exists():
            n_skip += 1
            print(f"[{idx}/{len(images)}] {image_path.name}: スキップ（既存）")
            continue

        try:
            t1 = time.time()
            boxes, _ = predict(processor, model, device, image_path, args.threshold)
            out_folder.mkdir(parents=True, exist_ok=True)
            write_polygon_txt(boxes, txt_path)
            draw_visualization(image_path, boxes, vis_path)
            n_ok += 1
            print(f"[{idx}/{len(images)}] {image_path.name}: {len(boxes)}件  "
                  f"({time.time() - t1:.1f}s) -> {out_folder}")
        except Exception as e:  # 1枚の失敗でバッチ全体を止めない
            failed.append((image_path.name, str(e)))
            print(f"[{idx}/{len(images)}] {image_path.name}: 失敗 - {e}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n完了: 成功={n_ok}  スキップ={n_skip}  失敗={len(failed)}")
    for name, err in failed:
        print(f"  失敗: {name}: {err}")


if __name__ == "__main__":
    main()
