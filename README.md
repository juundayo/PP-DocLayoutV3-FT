# PP-DocLayoutV3 ファインチューニング パイプライン

paddlex 3.7.2の実コード(`paddlex/modules/layout_analysis/`)を直接確認して確定した仕様に基づきます。
旧バージョンの`category_mapping.yaml`/`convert_to_coco.py`(25クラスへのマッピング版)は
**廃棄**してください。`document_info`等のマッピングは不要で、あなたの11カテゴリーを
そのままid 0〜10で使います。

## 確定仕様

- 訓練対象モジュール: `paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml`
  (`layout_detection`モジュールではなく`layout_analysis`モジュール。別物なので注意)
- データセットディレクトリ構造:
  ```
  <dataset_dir>/
    images/
      <file_name1>.jpg
      ...
    annotations/
      instance_train.json
      instance_val.json
  ```
- COCO instance segmentation形式。各annotationの`segmentation`にpolygon必須。
- 各annotationに**`read_order`**(0始まり非負整数)が必須。画像ごとに0,1,2,...,N-1の連番。
- `categories`はCOCO json中のものがそのまま使われる(`coco.getCatIds()`で自動カウント)ので、
  あなたの11カテゴリー(id 0〜10)をそのまま使えばよい。

## 1. データセット変換

```bash
conda activate vl16
cd /home/claude/pharos_finetune   # ご自身の作業場所にコピーしてください
pip install pillow --break-system-packages   # 既にあれば不要

python scripts/convert_to_coco_v3_final.py \
    --dataset_root /media/StorageServer/PHAROS/pharos_epirotic \
    --out_dir /media/SSD/vl16/dataset_pharos \
    --val_ratio 0.15 \
    --seed 42
```

ポイント:
- `selected.txt`のINT1==0行は無視。
- train/valは**文書単位**(`INT2/STR2`フォルダ単位)で分割し、同一文書の異なるページが
  train/val両方に混ざるリークを防止。
- 約100画像なので`--val_ratio 0.15`で15〜16文書程度をvalに回す想定です(文書数=画像数の
  場合。1文書に複数ページが入っている場合は文書数の方が少なくなるので、出力ログの
  「文書数: 全X / train=Y / val=Z」を確認してください)。
- アノテーションが0件の画像は自動的に除外されます(read_order検証エラーを防ぐため)。
- WARNINGログ(画像/GT欠損、座標異常、未知のcategory_id)は必ず目視確認してください。

変換後の中身を簡易チェック:

```bash
python - <<'EOF'
import json, collections
for split in ["train", "val"]:
    d = json.load(open(f"/media/SSD/vl16/dataset_pharos/annotations/instance_{split}.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    cnt = collections.Counter(a["category_id"] for a in d["annotations"])
    print(split, "images:", len(d["images"]), "anns:", len(d["annotations"]))
    for cid, n in sorted(cnt.items()):
        print(f"  {cats[cid]:>18s}: {n}")
EOF
```

カテゴリーごとの件数に極端な偏りがないか確認してください。新聞紙写真なら`Text`/`Header`/
`Image`は多く、`Document Info`/`Caption`/`Formula`あたりは少なめになりがちです。
件数が極端に少ない(数件レベル)クラスは、ファインチューニングしてもほぼ改善しません。

## 2. データセット検証 (check_dataset)

```bash
cd /media/SSD/vl16
python main.py \
    -c $(python -c "import paddlex,os;print(os.path.join(os.path.dirname(paddlex.__file__),'configs/modules/layout_analysis/PP-DocLayoutV3.yaml'))") \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=/media/SSD/vl16/dataset_pharos
```

`main.py`が手元に無い場合は、PaddleX付属のCLIエントリポイントを直接使ってください
(`paddlex`コマンド経由でも同様のオプション指定ができるはずです。バージョンによって
呼び出し方が異なるので、うまくいかなければ`paddlex --help`や`paddlex train --help`で
確認してください):

```bash
paddlex --config /media/SSD/vl16/vl16/lib/python3.10/site-packages/paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=/media/SSD/vl16/dataset_pharos
```

`./output/check_dataset_result.json`に`check_pass: true`、ログに
`read_order validation pass rate = 100.00%` が出れば成功です。
`./output/check_dataset/demo_img/`にbbox+maskの可視化画像が出るので、必ず目視で
ラベル・座標・読み取り順が正しいか確認してください(`draw_bbox`が`read_order`も描画する
ようなので、番号の並びが新聞紙の実際の読み順と一致しているか特に注意)。

pass rateが100%未満の場合、`convert_to_coco_v3_final.py`のread_order採番ロジック
(行スキップでズレるケース)に問題がある可能性があるので、教えてください。

## 3. ファインチューニング実行

```bash
paddlex --config /media/SSD/vl16/vl16/lib/python3.10/site-packages/paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=/media/SSD/vl16/dataset_pharos \
    -o Global.output=/media/SSD/vl16/output_doclayoutv3_pharos \
    -o Train.num_classes=11 \
    -o Train.batch_size=2 \
    -o Train.learning_rate=0.00002 \
    -o Train.epochs_iters=60 \
    -o Train.warmup_steps=50 \
    -o Train.eval_interval=1
```

デフォルト値からの変更理由:
- `Train.batch_size`: デフォルト4だが、画像が「でかい新聞紙の写真」=高解像度かつ
  RTX 2080Ti(11GB級)なので、まず2から試す。OOMしなければ4に戻して良い。
- `Train.learning_rate`: デフォルト1e-4は事前学習済みモデルのフルスクラッチ的な学習率に
  近いので、約100枚という小規模データでは過学習しやすい。2e-5程度に下げて様子を見る。
- `Train.epochs_iters`: デフォルト100は小規模データには過剰。60程度から始め、
  val側の指標(後述)が頭打ち/悪化し始めたタイミングのcheckpointを採用する。
- `pretrain_weight_path`はyamlのデフォルト(公式PP-DocLayoutV3事前学習済み重み)を
  そのまま使う(オーバーライド不要)。フルスクラッチではなくここから微調整します。

学習中、`Global.output`配下にログとcheckpoint(`best_model/`等)が出力されます。
`eval_interval=1`なので毎epoch評価が走り、ログに精度推移が出ます。

## 4. 評価

```bash
paddlex --config .../PP-DocLayoutV3.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=/media/SSD/vl16/dataset_pharos \
    -o Evaluate.weight_path=/media/SSD/vl16/output_doclayoutv3_pharos/best_model/best_model.pdparams
```

mAP系の指標に加えて、reading orderの一致度合いの指標がログに出るか確認してください。
出ない場合、val.json側の`read_order`と実際の推論結果の並び順を突き合わせて
Kendall's tauやARD(reading order精度の標準的指標)を自分で計算するスクリプトが必要に
なります。必要なら作成しますので教えてください。

## 5. エクスポート

```bash
paddlex --config .../PP-DocLayoutV3.yaml \
    -o Global.mode=export \
    -o Global.dataset_dir=/media/SSD/vl16/dataset_pharos \
    -o Export.weight_path=/media/SSD/vl16/output_doclayoutv3_pharos/best_model/best_model.pdparams \
    -o Global.output=/media/SSD/vl16/exported_doclayoutv3_pharos
```

エクスポート結果に`inference.yml`が含まれるはずです。`label_list`が
`["Text","Header","Paragraph Title","Image","Table","Formula","Page Number",
"Document Title","Footnote","Caption","Document Info"]`(あなたの11カテゴリー、
id順)になっているか必ず確認してください。元のV3公式モデルの25クラスlabel_listとは
別物になります。

## 6. PaddleOCR-VL-1.6パイプラインへの組み込み

```bash
mkdir -p ~/.paddlex/official_models/PP-DocLayoutV3-pharos
cp -r /media/SSD/vl16/exported_doclayoutv3_pharos/* ~/.paddlex/official_models/PP-DocLayoutV3-pharos/
```

単体での動作確認:

```bash
paddleocr layout_detection \
    -i /path/to/test_page.jpg \
    --model_name PP-DocLayoutV3-pharos \
    --model_dir ~/.paddlex/official_models/PP-DocLayoutV3-pharos
```

問題なければ、PaddleOCR-VLパイプライン設定(YAML)のlayout検出モデルの`model_dir`を
このパスに差し替えてください。reading orderは推論結果の並び順そのものに反映されます
(後段の`xycut_enhanced`等のヒューリスティック後処理が、モデルが出すread_order/bbox/
ラベルに基づいて最終的な並び順を決めている可能性が高いので、bbox精度と分類精度の
向上が reading order全体の精度向上にも効くはずです)。

## 7. ありがちな落とし穴

- **クラス不均衡**: Step 1のカテゴリー件数チェックで偏りが大きい場合、過学習を防ぐために
  少数クラスをoversamplingするか、loss重み付けの調整余地が`Train`セクションに無いか
  yamlを見てください(今回確認した範囲には無かったので、無ければPaddleDetection側の
  訓練コードを直接覗く必要があります)。
- **read_orderの連番ズレ**: GTファイルの一部の行が座標異常等でskipされると、後続行の
  read_orderがズレる可能性があるため、`convert_to_coco_v3_final.py`では画像ごとに
  最終的に0始まり連番へ詰め直しています。それでもズレが疑われる場合は、元の
  bbox_gt/*.txtを直接見直してください。
- **画像サイズ**: 新聞紙の高解像度写真の場合、`Preprocess`のtarget_size(800x800、元の
  inference.ymlより)へのResizeで小さい文字(Page NumberやFootnote等)が潰れる可能性が
  あります。精度が頭打ちになる場合、訓練時のinput resolution設定がyaml内に無いか
  (今回のyamlには明示が無かったので、PP-DocLayoutV3のモデルconfig
  `_config_pp_doclayout_v3.py`側を見る必要があります)確認の余地があります。

## ファイル一覧

- `scripts/convert_to_coco_v3_final.py` — selected.txt + bbox_gt -> PaddleX COCOInstSegDataset形式変換(最終版・これだけ使ってください)
- `scripts/convert_to_coco.py` / `configs/category_mapping.yaml` — 旧版(25クラスマッピング)。**廃棄、使わないでください**
- `scripts/patch_inference_yml.py` — 今回は基本不要(label_listはエクスポート時に自動で11クラスになるはず)。ズレていた場合の保険として残しています。
