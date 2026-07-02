#!/usr/bin/env python3
"""
ファインチューニング後にエクスポートされた inference.yml の label_list を
新しいカテゴリーリストに差し替える。

使い方:
  python patch_inference_yml.py \
      --in_yml /path/to/exported/inference.yml \
      --label_list /media/SSD/vl16/dataset_coco/label_list.txt \
      --out_yml /path/to/exported/inference.yml   # 上書きでも別名でも可
"""
import argparse
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_yml", required=True)
    ap.add_argument("--label_list", required=True)
    ap.add_argument("--out_yml", required=True)
    args = ap.parse_args()

    with open(args.in_yml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(args.label_list, "r", encoding="utf-8") as f:
        labels = [l.strip() for l in f if l.strip()]

    old = cfg.get("label_list", [])
    print(f"旧label_list ({len(old)}): {old}")
    print(f"新label_list ({len(labels)}): {labels}")

    cfg["label_list"] = labels

    with open(args.out_yml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"書き込み完了: {args.out_yml}")
    print("注意: num_classesに相当する設定がconfig.json/inference.json側にもあれば、"
          "それも新しいクラス数に揃っているか必ず確認してください。")


if __name__ == "__main__":
    main()
