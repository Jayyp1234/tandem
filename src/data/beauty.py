"""Amazon Beauty preprocessing — direct file download from McAuley-Lab on HuggingFace.

The newer `datasets` library (>=4.0) dropped loading-script support. We use
`huggingface_hub` to download the All_Beauty review and metadata JSONL files
directly. The All_Beauty subset is compact (~700K reviews / ~110K items).

SASRec / BERT4Rec / P5 numbers from the 2018 release are cited from the
literature (literature_evidence.md) as published reference points; we do not
re-run those baselines on this 2023 release.

Run via:  make data
"""
from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

DATA_DIR = Path("data/beauty_5core")

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
SEED = 42
TEST_ITEMS_N = 200
MIN_HISTORY_LENGTH = 11   # 10 history + 1 target per persona


def _find_files() -> tuple[str, str]:
    """Locate the All_Beauty review + meta files in the McAuley-Lab repo."""
    files = list_repo_files(REPO_ID, repo_type="dataset")
    review_file = None
    meta_file = None
    for f in files:
        if "All_Beauty" not in f:
            continue
        if not (f.endswith(".jsonl") or f.endswith(".jsonl.gz")):
            continue
        is_meta = "meta" in f.lower()
        if is_meta:
            if meta_file is None or len(f) < len(meta_file):
                meta_file = f
        else:
            if review_file is None or len(f) < len(review_file):
                review_file = f
    if not review_file or not meta_file:
        candidates = [f for f in files if "All_Beauty" in f][:10]
        raise RuntimeError(
            f"Could not locate All_Beauty review/meta JSONL in {REPO_ID}. "
            f"Candidates seen: {candidates}"
        )
    return review_file, meta_file


def _iter_jsonl(local_path: str):
    """Yield parsed JSON records from a JSONL or JSONL.GZ file."""
    if local_path.endswith(".gz"):
        opener = lambda: gzip.open(local_path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(local_path, "r", encoding="utf-8")
    with opener() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def preprocess() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # ----- 1. Locate + download files ----------------------------------------
    print(f"  listing files in {REPO_ID}...")
    review_path, meta_path = _find_files()
    print(f"    review file: {review_path}")
    print(f"    meta file:   {meta_path}")

    print("  downloading from HuggingFace (cached after first run)...")
    reviews_local = hf_hub_download(REPO_ID, review_path, repo_type="dataset")
    meta_local = hf_hub_download(REPO_ID, meta_path, repo_type="dataset")

    # ----- 2. Parse reviews → user histories ---------------------------------
    print("  parsing reviews...")
    user_history: dict[str, list[dict]] = defaultdict(list)
    item_set: set[str] = set()
    n_reviews = 0
    for r in tqdm(_iter_jsonl(reviews_local), desc="reviews", unit="rev"):
        uid = r.get("user_id")
        iid = r.get("parent_asin") or r.get("asin")
        if not uid or not iid:
            continue
        user_history[uid].append({
            "item_id":     iid,
            "rating":      float(r.get("rating") or 0.0),
            "summary":     (r.get("title") or "").strip(),
            "review_text": (r.get("text") or "").strip(),
            "timestamp":   int(r.get("timestamp") or 0),
        })
        item_set.add(iid)
        n_reviews += 1
    print(f"    {n_reviews:,} reviews / {len(user_history):,} users / {len(item_set):,} items")

    rich_users = {
        uid: sorted(hs, key=lambda h: h["timestamp"])
        for uid, hs in user_history.items()
        if len(hs) >= MIN_HISTORY_LENGTH
    }
    print(f"    {len(rich_users):,} users with >= {MIN_HISTORY_LENGTH} interactions")

    # ----- 3. Parse item metadata --------------------------------------------
    print("  parsing item metadata...")
    items: dict[str, dict] = {}
    for m in tqdm(_iter_jsonl(meta_local), desc="meta", unit="item"):
        iid = m.get("parent_asin") or m.get("asin")
        if not iid or iid not in item_set:
            continue
        title = (m.get("title") or "").strip()
        store = (m.get("store") or "").strip()
        categories = m.get("categories") or []
        category = (
            "; ".join(str(c) for c in categories[:3])
            if isinstance(categories, list) else str(categories)
        )
        description = m.get("description") or []
        if isinstance(description, list):
            description = " ".join(str(d) for d in description)
        items[iid] = {
            "item_id":     iid,
            "title":       title,
            "brand":       store,
            "category":    category,
            "description": (description or "")[:1000],
            "price":       m.get("price", ""),
        }
    print(f"    {len(items):,} items with metadata")

    # Drop users whose history references items lacking metadata
    final_users: list[tuple[str, list[dict]]] = []
    for uid, hs in rich_users.items():
        hs_with_meta = [h for h in hs if h["item_id"] in items]
        if len(hs_with_meta) >= MIN_HISTORY_LENGTH:
            final_users.append((uid, hs_with_meta))
    print(f"    {len(final_users):,} users retained after metadata join")

    # ----- 4. Sample test items ---------------------------------------------
    valid_item_ids = sorted(items.keys())
    rng.shuffle(valid_item_ids)
    test_items = valid_item_ids[:TEST_ITEMS_N]

    # ----- 5. Write outputs -------------------------------------------------
    print("  writing JSONL outputs...")
    with (DATA_DIR / "users.jsonl").open("w", encoding="utf-8") as f:
        for uid, hs in final_users:
            f.write(json.dumps({"user_id": uid, "history": hs}, ensure_ascii=True) + "\n")
    with (DATA_DIR / "items.jsonl").open("w", encoding="utf-8") as f:
        for iid in valid_item_ids:
            f.write(json.dumps(items[iid], ensure_ascii=True) + "\n")
    with (DATA_DIR / "test_items_200.jsonl").open("w", encoding="utf-8") as f:
        for iid in test_items:
            f.write(json.dumps(items[iid], ensure_ascii=True) + "\n")

    (DATA_DIR / ".done").touch()
    print(
        f"  done: {len(final_users)} users, {len(items)} items, "
        f"{len(test_items)} test items"
    )


if __name__ == "__main__":
    preprocess()
