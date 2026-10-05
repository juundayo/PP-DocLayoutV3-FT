import random
import shutil
from pathlib import Path

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

TOTAL = 360
ELIA_RATIO = 0.12

ELIA_ROOT = Path("/media/StorageServer/PHAROS/pharos_elia_text")
EPIROTIC_ROOT = Path("/media/StorageServer/PHAROS/pharos_epirotic")
EPIROTIC_SUBDIRS = [f"{i:03d}" for i in range(0, 11)]   # 000 〜 010

# 既に選ばれた epirotic のファイル一覧（ここにあるハッシュ名は除外）
EPIROTIC_EXCLUDE_TXT = EPIROTIC_ROOT / "selected.txt"

# コピー先とパス一覧の保存先
OUTPUT_DIR = Path("/media/StorageServer/PHAROS/sample_360")
IMAGES_DIR = OUTPUT_DIR / "images"
PATHS_TXT = OUTPUT_DIR / "selected_paths.txt"      # 元画像のパス（1行1枚）
MAPPING_TSV = OUTPUT_DIR / "mapping.tsv"           # 元パス ⇔ コピー先

SEED = 42          # 同じ結果を再現したい場合は固定。毎回変えたいなら None


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def list_subdirs(path: Path):
    if not path.is_dir():
        return []
    return sorted(
        p for p in path.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def is_real_jpeg(path: Path) -> bool:
    """中身が本物の JPEG か（先頭 2 バイトが FF D8）を確認"""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\xff\xd8"
    except OSError:
        return False


def load_excluded_hashes(txt_path: Path) -> set:
    """
    selected.txt の各行
      1 - nchr - 000 - 00ddc7cc8e470685322d52126083e5a66c9ea7ef - 0001
    から、40 桁のハッシュ名（ファイル名）を取り出す
    """
    excluded = set()
    if not txt_path.is_file():
        print(f"WARNING | 除外リストが見つかりません: {txt_path}")
        return excluded

    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            for part in line.split("-"):
                part = part.strip()
                if len(part) == 40 and all(c in "0123456789abcdef" for c in part.lower()):
                    excluded.add(part.lower())
    return excluded


def collect_groups_elia():
    """
    グループ = 各タイトルのフォルダ（例: ΑΘΗΝΑ_000100-20_492311）
    ドキュメント = PDF ごとのフォルダ（例: .../1）
    """
    groups = {}
    for group_dir in list_subdirs(ELIA_ROOT):
        docs = [
            d for d in list_subdirs(group_dir)
            if (d / "pages").is_dir()
        ]
        if docs:
            groups[group_dir.name] = docs
    return groups


def collect_groups_epirotic(excluded: set):
    """
    グループ = 000 〜 010
    ドキュメント = ハッシュ名のフォルダ（例: ff9657414fe3...）
    excluded に含まれるハッシュ名のフォルダは候補から外す
    """
    groups = {}
    skipped = 0
    for sub in EPIROTIC_SUBDIRS:
        group_dir = EPIROTIC_ROOT / sub
        docs = []
        for d in list_subdirs(group_dir):
            if not (d / "pages").is_dir():
                continue
            if d.name.lower() in excluded:
                skipped += 1
                continue
            docs.append(d)
        if docs:
            groups[sub] = docs
    print(f"EPIROTIC: selected.txt により {skipped} PDF を除外")
    return groups


def sample_pages(groups: dict, n: int, rng: random.Random):
    """
    グループを順番に回りながら、ランダムなドキュメントの
    ランダムなページを 1 枚ずつ選ぶ。
    同じドキュメントは、全ドキュメントを一巡するまで再度選ばない。
    """
    queues = {}
    for g, docs in groups.items():
        docs = docs[:]
        rng.shuffle(docs)
        queues[g] = docs

    page_cache = {}
    selected = []
    used = set()

    group_names = list(queues.keys())

    while len(selected) < n:
        rng.shuffle(group_names)
        progress = False

        for g in group_names:
            if len(selected) >= n:
                break

            queue = queues[g]
            tried = 0

            # このグループからまだ選べるページを探す
            while queue and tried < len(queue):
                doc = queue.pop(0)
                queue.append(doc)          # 末尾に回す（ラウンドロビン）
                tried += 1

                if doc not in page_cache:
                    page_cache[doc] = sorted((doc / "pages").glob("*.jpg"))

                candidates = [
                    p for p in page_cache[doc]
                    if p not in used
                ]
                rng.shuffle(candidates)

                chosen = None
                for p in candidates:
                    if is_real_jpeg(p):
                        chosen = p
                        break
                    used.add(p)            # 壊れた画像は今後も候補から外す

                if chosen is None:
                    continue

                used.add(chosen)
                selected.append((g, doc.name, chosen))
                progress = True
                break

        if not progress:
            print(f"WARNING | 画像が足りません: {len(selected)}/{n} 枚のみ選択")
            break

    return selected


# --------------------------------------------------
# MAIN
# --------------------------------------------------

rng = random.Random(SEED)

n_elia = round(TOTAL * ELIA_RATIO)
n_epirotic = TOTAL - n_elia

print("Scanning folders ...")
elia_groups = collect_groups_elia()
excluded_hashes = load_excluded_hashes(EPIROTIC_EXCLUDE_TXT)
print(f"selected.txt: {len(excluded_hashes)} 件のファイル名を読み込み")
epirotic_groups = collect_groups_epirotic(excluded_hashes)

print(
    f"ELIA: {len(elia_groups)} groups, "
    f"{sum(len(v) for v in elia_groups.values())} PDFs"
)
print(
    f"EPIROTIC: {len(epirotic_groups)} groups, "
    f"{sum(len(v) for v in epirotic_groups.values())} PDFs"
)

elia_selected = sample_pages(elia_groups, n_elia, rng)
epirotic_selected = sample_pages(epirotic_groups, n_epirotic, rng)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

all_selected = (
    [("elia", *s) for s in elia_selected] +
    [("epirotic", *s) for s in epirotic_selected]
)

with open(PATHS_TXT, "w", encoding="utf-8") as f_txt, \
     open(MAPPING_TSV, "w", encoding="utf-8") as f_map:

    f_map.write("index\tdataset\tsource_path\tcopied_path\n")

    for idx, (dataset, group, doc, src) in enumerate(all_selected, start=1):

        # 全部 1.jpg などの同名になるので、一意のファイル名を付ける
        dst_name = f"{idx:03d}_{dataset}_{group}_{doc}_{src.stem}.jpg"
        dst = IMAGES_DIR / dst_name

        shutil.copy2(src, dst)

        f_txt.write(f"{src}\n")
        f_map.write(f"{idx}\t{dataset}\t{src}\t{dst}\n")

print(f"ELIA: {len(elia_selected)} / {n_elia}")
print(f"EPIROTIC: {len(epirotic_selected)} / {n_epirotic}")
print(f"Copied to: {IMAGES_DIR}")
print(f"Paths saved to: {PATHS_TXT}")
print("DONE")
