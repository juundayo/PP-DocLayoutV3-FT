#!/usr/bin/env python3
"""
PP-DocLayoutV3 バッチ推論スクリプト

処理対象:
  /media/StorageServer/Data/PHAROS/pharos_epirotic/{000-010}/{hash}/pages/*.jpg

結果保存:
  {hash}/bbox_res/{filename}.txt
  形式: cat_id x1 y1 x2 y1 x2 y2 x1 y2  (読み取り順、矩形を4点polygonで表現)
"""
import sys
import time
import glob
from pathlib import Path
from paddlex.inference.pipelines import create_pipeline

# ── カラー出力 ───────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
MAGENTA= "\033[95m"
RED    = "\033[91m"
GRAY   = "\033[90m"

def cprint(color, msg):
    print(f"{color}{msg}{RESET}", flush=True)

# ── 設定 ────────────────────────────────────────────────────────────────────
ROOT      = Path("/media/StorageServer/Data/PHAROS/pharos_epirotic")
MODEL_DIR = "/media/SSD/vl16/output_doclayoutv3_pharos/best_model/inference"
THRESHOLD = 0.3

LABEL_TO_ID = {
    "Text": 0, "Header": 1, "Paragraph Title": 2, "Image": 3,
    "Table": 4, "Formula": 5, "Page Number": 6, "Document Title": 7,
    "Footnote": 8, "Caption": 9, "Document Info": 10,
    # 素のモデル(小文字)も念のため
    "text": 0, "header": 1, "paragraph_title": 2, "image": 3,
    "table": 4, "display_formula": 5, "number": 6, "doc_title": 7,
    "footnote": 8, "figure_title": 9,
}

def make_config(model_dir):
    return {
        "pipeline_name": "PaddleOCR-VL-1.6",
        "use_doc_preprocessor": False,
        "use_layout_detection": True,
        "use_chart_recognition": False,
        "use_seal_recognition": False,
        "format_block_content": False,
        "merge_layout_blocks": False,
        "SubModules": {
            "LayoutDetection": {
                "module_name": "layout_detection",
                "model_name": "PP-DocLayoutV3",
                "model_dir": model_dir,
                "batch_size": 1,
                "threshold": THRESHOLD,
                "layout_nms": True,
            }
        },
    }

def boxes_to_txt(boxes):
    """読み取り順にソートしてtxt形式に変換"""
    sorted_boxes = sorted(boxes, key=lambda b: (b.get("order") or 0))
    lines = []
    for b in sorted_boxes:
        label = b.get("label", "Text")
        cat_id = LABEL_TO_ID.get(label, 0)
        x1, y1, x2, y2 = [int(v) for v in b["coordinate"]]
        # 矩形を4点polygon: 左上→右上→右下→左下
        lines.append(f"{cat_id} {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}")
    return "\n".join(lines)

def find_images():
    """全画像パスを (folder_num, hash, image_path) のリストで返す"""
    items = []
    for folder_num in sorted(ROOT.glob("[0-9][0-9][0-9]")):
        for hash_dir in sorted(folder_num.iterdir()):
            pages_dir = hash_dir / "pages"
            if not pages_dir.exists():
                continue
            imgs = sorted(pages_dir.glob("*.jpg")) + sorted(pages_dir.glob("*.png"))
            for img in imgs:
                items.append((folder_num.name, hash_dir.name, img))
    return items

def main():
    # パイプラインを一度だけロード
    cprint(CYAN + BOLD, "=" * 60)
    cprint(CYAN + BOLD, "  PP-DocLayoutV3 バッチ推論")
    cprint(CYAN + BOLD, "=" * 60)
    cprint(YELLOW, f"モデル: {MODEL_DIR}")
    cprint(YELLOW, f"ルート: {ROOT}")

    t0 = time.time()
    cprint(GRAY, "パイプラインをロード中...")
    pipeline = create_pipeline(config=make_config(MODEL_DIR), device="gpu:0")
    cprint(GREEN, f"ロード完了 ({time.time()-t0:.1f}s)")

    items = find_images()
    total = len(items)
    cprint(CYAN, f"\n処理対象: {total} 枚\n")

    done = 0
    errors = 0
    prev_folder = None
    prev_hash = None
    t_start = time.time()

    for folder_num, hash_name, img_path in items:
        # フォルダー/ハッシュが変わったら表示
        if folder_num != prev_folder:
            cprint(BOLD + MAGENTA, f"\n📁 フォルダー: {folder_num}")
            prev_folder = folder_num
            prev_hash = None
        if hash_name != prev_hash:
            cprint(CYAN, f"  📄 {hash_name[:20]}...")
            prev_hash = hash_name

        # 結果保存先
        bbox_res_dir = img_path.parent.parent / "bbox_res"
        bbox_res_dir.mkdir(exist_ok=True)
        out_txt = bbox_res_dir / (img_path.stem + ".txt")

        try:
            t1 = time.time()
            results = list(pipeline.predict(str(img_path)))
            elapsed = time.time() - t1

            boxes = results[0].get("layout_det_res", {}).get("boxes", [])
            if not boxes:
                boxes = results[0].get("boxes", [])

            txt = boxes_to_txt(boxes)
            out_txt.write_text(txt, encoding="utf-8")

            done += 1
            pct = done / total * 100
            eta = (time.time() - t_start) / done * (total - done)
            cprint(GREEN,
                f"    ✓ {img_path.name}  "
                f"{len(boxes)}件  {elapsed:.1f}s  "
                f"[{done}/{total} {pct:.0f}%  ETA {eta:.0f}s]")

        except Exception as e:
            errors += 1
            cprint(RED, f"    ✗ {img_path.name}  ERROR: {e}")

    total_time = time.time() - t_start
    cprint(BOLD + CYAN, "\n" + "=" * 60)
    cprint(BOLD + GREEN, f"完了: {done}/{total} 枚  エラー: {errors}  "
                         f"合計時間: {total_time:.0f}s  "
                         f"平均: {total_time/max(done,1):.1f}s/枚")
    cprint(BOLD + CYAN, "=" * 60)

if __name__ == "__main__":
    main()
