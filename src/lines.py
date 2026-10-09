#!/usr/bin/env python3
"""
detect_text_lines.py

ブロック画像（1枚）またはブロック画像のフォルダを入力にして、「テキスト行」の
ポリゴンだけを出力する。PaddleOCRのテキスト検出モジュール(TextDetection,
PP-OCRv5_server_det)を使う。レイアウト検出(PP-DocLayoutV3)や
PaddleOCR-VLパイプラインは「ブロック」単位なので、行ポリゴンは出せない。

出力（入力が .../blocks/0000_00005.jpg の場合、既定では .../line_polygons/ に保存）:
    0000_00005.txt       1行 = 1テキスト行。"x1 y1 x2 y2 x3 y3 x4 y4"（読み順＝上から下）
    0000_00005_vis.jpg   行ポリゴンと行番号(1..N)を描画した画像

  --class_id 0 を付けると、各行の先頭にクラスIDを付ける（bbox_gtと同じ形式）。

使い方:
    pip install paddleocr
    python detect_text_lines.py /media/StorageServer/PHAROS/pharos_epirotic/000/029a9a9d25d8ae829fe9661bc821dd1ab07d75a0/blocks/0000_00005.jpg
    python detect_text_lines.py /path/to/blocks/ --skip_existing
    python detect_text_lines.py img.jpg --unclip_ratio 1.3 --box_thresh 0.5   # 検出の調整
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
LINE_COLOR = (50, 50, 220)  # BGR


# ====================================================================
# 検出
# ====================================================================

def load_detector(model_name: str, device: str | None):
    from paddleocr import TextDetection  # 遅延import（無くても他の部分は動く）
    kwargs = {"model_name": model_name}
    if device:
        kwargs["device"] = device
    return TextDetection(**kwargs)


def _field(res, key):
    """結果オブジェクトからフィールドを取り出す（dict風 / .json['res'] の両方に対応）。"""
    try:
        return res[key]
    except Exception:
        return res.json["res"][key]


def detect_lines(detector, image_path: Path, predict_params: dict):
    """1枚の画像から行ポリゴンとスコアを返す。polys: list of (4,2) float array"""
    out = detector.predict(str(image_path), batch_size=1, **predict_params)
    res = out[0] if isinstance(out, (list, tuple)) else next(iter(out))
    polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in _field(res, "dt_polys")]
    scores = [float(s) for s in _field(res, "dt_scores")]
    return polys, scores


# ====================================================================
# 後処理
# ====================================================================

def sort_reading_order(polys, scores):
    """検出結果は順不同なので、行の中心のy座標（同じなら x）で上から下に並べる。"""
    order = sorted(range(len(polys)),
                   key=lambda i: (polys[i][:, 1].mean(), polys[i][:, 0].mean()))
    return [polys[i] for i in order], [scores[i] for i in order]


def clip_polygon(poly, width, height):
    """整数に丸め、画像の範囲内(0..w-1, 0..h-1)にクリップする。"""
    p = np.round(np.asarray(poly, dtype=np.float64)).astype(np.int64)
    p[:, 0] = np.clip(p[:, 0], 0, width - 1)
    p[:, 1] = np.clip(p[:, 1], 0, height - 1)
    return p.astype(np.int32)


def write_txt(polys, txt_path: Path, class_id: int | None):
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in polys:
            coords = " ".join(str(int(v)) for v in p.flatten())
            f.write(f"{class_id} {coords}\n" if class_id is not None else f"{coords}\n")


def draw_lines(img, polys, scores, out_path: Path):
    h, w = img.shape[:2]
    font_scale = max(0.45, h / 1500)
    thick_text = 1 if h < 1200 else 2
    thick_line = max(1, int(round(min(h, w) / 400)))
    vis = img.copy()
    for n, (p, s) in enumerate(zip(polys, scores), start=1):
        cv2.polylines(vis, [p.reshape(-1, 1, 2)], isClosed=True, color=LINE_COLOR, thickness=thick_line)
        label = f"{n}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick_text)
        x, y = int(p[:, 0].min()), int(p[:, 1].min())
        y0 = max(y - th - 4, 0)
        cv2.rectangle(vis, (x, y0), (x + tw + 4, y0 + th + 4), LINE_COLOR, -1)
        cv2.putText(vis, label, (x + 2, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thick_text)
    cv2.imwrite(str(out_path), vis)


# ====================================================================
# メイン
# ====================================================================

def default_out_dir(src: Path) -> Path:
    base = src if src.is_dir() else src.parent
    return (base.parent if base.name == "blocks" else base) / "line_polygons"


def process_image(detector, image_path: Path, out_dir: Path, predict_params, class_id):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("画像を読み込めません")
    h, w = img.shape[:2]
    polys, scores = detect_lines(detector, image_path, predict_params)
    polys, scores = sort_reading_order(polys, scores)
    polys = [clip_polygon(p, w, h) for p in polys]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_txt(polys, out_dir / f"{image_path.stem}.txt", class_id)
    draw_lines(img, polys, scores, out_dir / f"{image_path.stem}_vis.jpg")
    return len(polys)


def main():
    ap = argparse.ArgumentParser(description="ブロック画像からテキスト行のポリゴンを検出する")
    ap.add_argument("input", help="ブロック画像1枚、またはブロック画像のフォルダ")
    ap.add_argument("--out_dir", default=None, help="出力先（省略時は blocks/ の隣の line_polygons/）")
    ap.add_argument("--model", default="PP-OCRv5_server_det",
                    help="PP-OCRv5_server_det（高精度）/ PP-OCRv5_mobile_det（高速）")
    ap.add_argument("--device", default=None, help="例: gpu:0 / cpu（省略時は自動）")
    ap.add_argument("--class_id", type=int, default=None, help="指定すると各行の先頭にクラスIDを付ける")
    ap.add_argument("--skip_existing", action="store_true")
    # 検出パラメータ（省略時はモデルの既定値）
    ap.add_argument("--thresh", type=float, default=None, help="画素レベルの二値化しきい値")
    ap.add_argument("--box_thresh", type=float, default=None, help="行として採用する平均スコアのしきい値")
    ap.add_argument("--unclip_ratio", type=float, default=None,
                    help="行の膨張率。行間が詰まっていて隣の行とくっつく場合は下げる")
    ap.add_argument("--limit_side_len", type=int, default=None)
    ap.add_argument("--limit_type", default=None, choices=["min", "max"])
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(src)
    images = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"画像が見つかりません: {src}")
        return
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(src)

    predict_params = {k: v for k, v in {
        "thresh": args.thresh, "box_thresh": args.box_thresh, "unclip_ratio": args.unclip_ratio,
        "limit_side_len": args.limit_side_len, "limit_type": args.limit_type,
    }.items() if v is not None}

    t0 = time.time()
    detector = load_detector(args.model, args.device)
    print(f"モデルロード: {time.time() - t0:.1f}s  ({args.model})  画像 {len(images)} 枚 -> {out_dir}")

    n_ok, failed = 0, []
    for i, p in enumerate(images, start=1):
        if args.skip_existing and (out_dir / f"{p.stem}.txt").exists():
            print(f"[{i}/{len(images)}] {p.name}: スキップ（既存）")
            continue
        try:
            t1 = time.time()
            n = process_image(detector, p, out_dir, predict_params, args.class_id)
            n_ok += 1
            warn = "  ← 0行（パラメータを調整してください）" if n == 0 else ""
            print(f"[{i}/{len(images)}] {p.name}: {n}行  ({time.time() - t1:.2f}s){warn}")
        except Exception as e:  # 1枚の失敗でバッチを止めない
            failed.append((p.name, str(e)))
            print(f"[{i}/{len(images)}] {p.name}: 失敗 - {e}")
    print(f"\n完了: 成功={n_ok}  失敗={len(failed)}")
    for name, err in failed:
        print(f"  失敗: {name}: {err}")


if __name__ == "__main__":
    main()
