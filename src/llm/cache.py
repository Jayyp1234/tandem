"""JSONL response cache for LLM calls.

Per research/experiment_matrix.md §0: cache key = SHA-256 of (prompt, model, seed,
temperature). Cache is shipped with the submission so reviewers reproduce numbers
without API access.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class CacheKey:
    """The canonical cache key. Hash determinism matters for reproducibility."""
    prompt: str
    model: str
    seed: int
    temperature: float

    def to_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass
class CacheEntry:
    """A single cached LLM call. Shape mirrors the schema in experiment_matrix.md §2."""
    key_hash: str
    prompt: str
    model: str
    seed: int
    temperature: float
    response_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    timestamp: str

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False) + "\n"

    @classmethod
    def from_jsonl(cls, line: str) -> "CacheEntry":
        return cls(**json.loads(line))


class JsonlCache:
    """Append-only JSONL cache. Reads load entire file into a hash-indexed dict.

    For a hackathon-scale cache (~30k entries, ~30MB), in-memory indexing is fine.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, CacheEntry] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = CacheEntry.from_jsonl(line)
                self._index[entry.key_hash] = entry

    def get(self, key: CacheKey) -> CacheEntry | None:
        return self._index.get(key.to_hash())

    def put(self, key: CacheKey, entry: CacheEntry) -> None:
        self._index[key.to_hash()] = entry
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry.to_jsonl())

    def __contains__(self, key: CacheKey) -> bool:
        return key.to_hash() in self._index

    def __len__(self) -> int:
        return len(self._index)

    def iter_entries(self) -> Iterator[CacheEntry]:
        yield from self._index.values()


def deterministic_seed(*parts: str | int) -> int:
    """Per experiment_matrix.md §0: seed deterministic per cell tuple.

    Used to derive the seed from (persona_id, item_id, condition, architecture).
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
