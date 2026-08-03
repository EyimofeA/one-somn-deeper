"""jsonl -> torch Dataset, plus the stratification keys evaluate.py needs."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokens import IGNORE_INDEX, PAD


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class DiagnosticDataset(Dataset):
    """Fixed-length rows (no padding needed within one task/scale, but the
    attention_mask is still produced so the model code doesn't special-case
    fully-populated sequences vs. shorter ones from a different config)."""

    def __init__(self, path: str | Path):
        self.rows = load_jsonl(Path(path))
        if not self.rows:
            raise ValueError(f"{path} is empty")
        self.max_len = max(len(r["input_ids"]) for r in self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        ids = row["input_ids"]
        labels = row["labels"]
        pad_n = self.max_len - len(ids)
        input_ids = ids + [PAD] * pad_n
        label_ids = labels + [IGNORE_INDEX] * pad_n
        attention_mask = [1] * len(ids) + [0] * pad_n
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "index": idx,
            "u": torch.tensor(row.get("u", -1), dtype=torch.long),
            "quotient": torch.tensor(row.get("quotient", -1), dtype=torch.long),
        }

    def meta(self, idx: int) -> dict:
        return self.rows[idx]


class ShuffledContextDataset(Dataset):
    """Attach a fixed, non-self source context for an initialization control."""

    def __init__(self, base: DiagnosticDataset, seed: int):
        self.base = base
        self.max_len = base.max_len
        generator = torch.Generator().manual_seed(seed)
        identity = torch.arange(len(base))
        while True:
            permutation = torch.randperm(len(base), generator=generator)
            if not (permutation == identity).any():
                break
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = dict(self.base[idx])
        context_item = self.base[int(self.permutation[idx])]
        item["init_input_ids"] = context_item["input_ids"]
        item["init_attention_mask"] = context_item["attention_mask"]
        return item

    def meta(self, idx: int) -> dict:
        return self.base.meta(idx)
