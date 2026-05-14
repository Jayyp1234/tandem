"""Naija-vs-English classifier for H6.

Per protocol_spec.md §4 H6: train a small TF-IDF + logistic-regression classifier
on Naija text (MasakhaNEWS-pcm, public, CC-BY) vs English news, then score
TANDEM's outputs. Cultural-overlay-on outputs should score significantly more
Naija-like than noise-on outputs (paired Wilcoxon).

Trains in seconds on CPU. Saves the fitted model to `results/naija_classifier.joblib`.

Run via:  python -m src.eval.classifier
"""
from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Public mirrors. Both are CC-BY (verified in phase3_cultural_resources.md §1.2).
PCM_URL = (
    "https://huggingface.co/datasets/masakhane/masakhanews/resolve/main/"
    "data/pcm/test.tsv"
)
EN_NEWS_URL = (
    "https://huggingface.co/datasets/SetFit/ag_news/resolve/main/test.jsonl"
)

CACHE_DIR = Path("data/raw")
MODEL_PATH = Path("results/naija_classifier.joblib")
SEED = 42


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"  downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _load_pcm() -> list[str]:
    """MasakhaNEWS-pcm test split. TSV with `label\\theadline\\ttext`."""
    path = _download(PCM_URL, CACHE_DIR / "masakhanews_pcm_test.tsv")
    docs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2].strip():
            docs.append(parts[2].strip())
    return docs


def _load_english() -> list[str]:
    """AG News test split (English news, public, similar register/length)."""
    path = _download(EN_NEWS_URL, CACHE_DIR / "ag_news_test.jsonl")
    docs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = rec.get("text", "").strip()
        if text:
            docs.append(text)
    return docs


def train(save_path: Path = MODEL_PATH) -> dict:
    naija = _load_pcm()
    english = _load_english()
    # Balance corpora so the classifier doesn't trivially exploit prior probabilities.
    n = min(len(naija), len(english))
    rng = np.random.default_rng(SEED)
    rng.shuffle(naija)
    rng.shuffle(english)
    naija, english = naija[:n], english[:n]
    print(f"  Naija docs: {len(naija)}, English docs: {len(english)}")

    X = naija + english
    y = [1] * len(naija) + [0] * len(english)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=20000,
            sublinear_tf=True, lowercase=True,
        )),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print("  classification report (English=0, Naija=1):")
    print(classification_report(y_test, y_pred))
    print(f"  AUC: {auc:.4f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "auc": float(auc)}, save_path)
    print(f"  saved to {save_path}")
    return {"auc": float(auc), "n_train": len(X_train), "n_test": len(X_test)}


def naija_score(text: str, model_path: Path = MODEL_PATH) -> float:
    """Return the trained classifier's predicted P(text is Naija) for a single review."""
    model = joblib.load(model_path)
    return float(model["pipeline"].predict_proba([text])[0, 1])


def naija_scores(texts: list[str], model_path: Path = MODEL_PATH) -> np.ndarray:
    """Batch version; same probability semantic."""
    model = joblib.load(model_path)
    return model["pipeline"].predict_proba(texts)[:, 1]


if __name__ == "__main__":
    train()
