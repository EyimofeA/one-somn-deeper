# One Layer Deeper — Primary-sources handoff packet
Compiled 2026-07-22. Contains primary sources and raw logged metrics only.
Sources: [onelayerdeeper.ai](https://onelayerdeeper.ai), [onelayerdeeper.ai/problem](https://onelayerdeeper.ai/problem), [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper) (local clone `competition/`), `one-layer leaderboard` CLI, `solving/experiments/metrics/*.jsonl`.

## 1. Competition rules and Hard-tier recurrence warning

### 1.1 Rules text from upstream README (`competition/README.md`, section “Rules” through “Compute tiers”)

Verbatim copy of README headings `## Rules`, `### Submission contract`, and `### Compute tiers` (through the paragraph before `## CLI`). Outer fence uses four backticks so the nested submission-contract Python fence stays intact.

````markdown
## Rules

1. Submit exactly one UTF-8 file named `submission.py`. It exports one `benchmark.Submission` with model and optimizer factories and an optional training loss.
2. The submission must be self-contained. It may import the public `benchmark` API and pinned evaluator dependencies, but it may not depend on repository `model` or `optim` modules, extra files, package installation, or external services.
3. Participant code defines the model, optimizer bundle, optional learning-rate scheduler, optional loss, training and evaluation batch sizes, and maximum training steps. Recurrence, adaptive computation, and depth curricula are allowed.
4. The evaluator fixes data, sampling, the one-forward/one-backward loop, gradient clipping, optimizer cadence, seeds, deadline, final evaluation, and aggregation. Participants may choose the training and evaluation batch size and a lower maximum step count; evaluator ceilings still apply.
5. The model may contain at most 500,000,000 scalar parameters and persistent buffers. Shared state counts once; frozen state still counts.
6. Optimizer state, activations, and temporary workspace may use remaining VRAM. OOM or timeout fails the run.
7. Easy provides 60 H100 training seconds, Medium 600 seconds, and Hard 3,600 seconds. Model construction, submission import, and compilation consume the budget.
8. A custom training loss receives final logits, labels, and the model's auxiliary output and returns one differentiable finite scalar. The evaluator performs backward.
9. Each final checkpoint is evaluated once with a separate time budget equal to half its training allowance. The evaluator uses fixed loss and exact accuracy, and the score is mean exact accuracy across fixed evaluation splits and seeds.
10. Data inspection, task-specific solvers, custom training loops, participant-controlled backward passes, and manifest overrides are not allowed.
11. The metric recorded for a Hard run must not be exploited. Any attempt to exploit it will result in an immediate ban.

Depth is deliberately unconstrained. Fixed stacks, tied recurrence, iterative refinement, routing, adaptive halting, memory tokens, and parameter-free work are all valid if the model-state ceiling is respected. A deeper forward completes fewer optimizer updates under the same clock.

### Submission contract

The file is limited to 256 KiB. `build_model(spec)` receives `vocab_size`, `max_seq_len`, and `maximum_model_state_elements`. It returns a `torch.nn.Module` whose `config` exposes the first two matching fields. The model accepts evaluator tensor arguments and returns `(logits, auxiliary_value)`.

The evaluator calls `model.train()` for optimization and `model.eval()` for final evaluation. If the model should behave differently during evaluation, use PyTorch's inherited `self.training` flag inside `forward` (for example, `if self.training: ... else: ...`).

`build_optimizer(model, spec)` receives the per-seed time allowance and device type. It returns an `OptimizerBundle`; its optimizer must include every trainable parameter exactly once. An optional scheduler is stepped after every update.

```python
from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

def build_model(spec: ModelSpec):
    model = MyModel(spec)
    assert_model_state(model, spec)
    return model

def build_optimizer(model, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(MyOptimizer(model.parameters()))

SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=512,       # optional; training
    eval_batch_size=1024, # optional; evaluation
    max_steps=20_000,     # optional; cannot exceed the evaluator ceiling
)
```

If omitted, `batch_size` and `max_steps` use the evaluator manifest defaults.
Evaluation uses `eval_batch_size` when provided, then an explicit participant
`batch_size`, then the evaluator manifest's evaluation batch size, and finally
the manifest's training batch size. A participant `max_steps` can end training
early. The evaluator's wall-clock deadline and absolute step ceiling always remain
enforced. An optional scheduler returned in `OptimizerBundle` is stepped after
every completed optimizer update.

The website offers one basic, non-recurrent Transformer using `torch.optim.AdamW`. Its standalone `submission.py` lives under `submissions/baseline_adamw`.

### Compute tiers

The public Easy and Medium datasets provide separate prompt and output tensors.
The evaluator supplies a padding mask, not a causal mask, so models can attend
bidirectionally over the complete prompt in those practice tiers. Hard uses a
private hidden evaluator.

- **Easy:** datasets `e1`–`e5`, 60 training seconds, 60 accepted attempts per UTC day.
- **Medium:** datasets `m1`–`m5`, 600 training seconds, 6 accepted attempts per UTC day.
- **Hard:** dataset `h1`, 3,600 training seconds, 1 accepted attempt per UTC day.

Easy and Medium are practice tiers. The public leaderboard ranks only each participant's best successful Hard submission. Failed evaluations count after acceptance; authentication and validation rejections do not. Source and detailed results remain private.
````

### 1.2 Problem page text (https://onelayerdeeper.ai/problem) — Hard recurrence warning

Verbatim excerpt captured 2026-07-22 via page fetch:

> Hard is a hidden task evaluation and may change aspects of the recurrence itself; do not assume it is repeated squaring.

Additional problem-page text (same fetch):

```
Given a modulus N, a starting value x, and a step count T, predict the residue after squaring modulo N exactly T times.

Recurrence
x0 = x mod N
xt = xt−1² mod N
y = xT = x^(2^T) mod N

Hard equally averages exact accuracy on its hidden test, held-out-depth, and jointly held-out modulus/depth splits across seeds. Easy and Medium equally average test and their merged out-of-distribution split.
```


### 1.3 Organizer Q&A exchange (participant question + official response)

**Participant question** (paraphrase of submission-page warning; recorded 2026-07-22 for this packet):

> Hard task warning: Hard may change aspects of the recurrence itself; do not assume it is repeated squaring. this is mentioned on submission page does this mean, instead of X^2 mod N being the single step in serial computation, we should assume, it will be some nearby family? like affine transform of that, or cube or something else?

**Official response** (verbatim as provided for this packet; confirmation status noted by principal: “hard has not been confirmed”):

> yeah, some people have tried to guess that slightly new family, some approaches have worked, new ones have not lol

**Status:** The official line above acknowledges a “slightly new family” and that some guesses worked and some did not. Exact Hard recurrence is still not published. Principal note: Hard details have not been confirmed beyond this exchange and the public problem-page warning in §1.2.

## 2. Evaluator / harness interface sources

Paths relative to the upstream clone root (`competition/` locally).

### `benchmark/api.py`

```python
"""The complete public API available to an official submission."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class Scheduler(Protocol):
    def step(self) -> None: ...


@dataclass(frozen=True)
class ModelSpec:
    """Public shape and state limits for the current benchmark task."""

    vocab_size: int
    max_seq_len: int
    maximum_model_state_elements: int


def model_state_tensors(model: nn.Module):
    """Yield every distinct parameter and persistent buffer that counts."""

    seen: set[int] = set()
    for name, value in model.named_parameters():
        if id(value) not in seen:
            seen.add(id(value))
            yield f"parameter:{name}", value
    for module_name, child in model.named_modules():
        for name, value in child._buffers.items():
            if value is None or name in child._non_persistent_buffers_set:
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            full_name = f"{module_name}.{name}" if module_name else name
            yield f"buffer:{full_name}", value


def count_model_state_elements(model: nn.Module) -> int:
    """Count scalar elements in all inference-persistent model state."""

    return sum(value.numel() for _, value in model_state_tensors(model))


def assert_model_state(model: nn.Module, spec: ModelSpec) -> int:
    """Assert the public state budget and return the measured element count.

    Submissions should call this at the end of ``build_model`` for an immediate,
    readable failure.  The evaluator repeats the count independently after
    moving the model to the evaluation device.
    """

    elements = count_model_state_elements(model)
    if elements > spec.maximum_model_state_elements:
        raise AssertionError(
            f"model persistent state ({elements:,}) exceeds maximum "
            f"({spec.maximum_model_state_elements:,})"
        )
    return elements


@dataclass(frozen=True)
class OptimizerSpec:
    """Public, data-independent information available to an optimizer."""

    training_time_seconds: float
    device_type: str


@dataclass(frozen=True)
class OptimizerBundle:
    """Participant optimizer and optional participant-defined LR scheduler."""

    optimizer: torch.optim.Optimizer
    scheduler: Scheduler | None = None


@dataclass(frozen=True)
class Submission:
    """Participant-controlled components exported as ``SUBMISSION``."""

    build_model: Callable[[ModelSpec], nn.Module]
    build_optimizer: Callable[[nn.Module, OptimizerSpec], OptimizerBundle]
    training_loss: (
        Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor] | None
    ) = None
    batch_size: int | None = None
    max_steps: int | None = None
    eval_batch_size: int | None = None

```

### `benchmark/batches.py`

```python
"""Evaluator-owned batch normalization for supported benchmark datasets."""

from __future__ import annotations

import torch


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def prepare_batch(batch, device: torch.device):
    batch = _move_to_device(batch, device)
    input_ids = batch["input_ids"].long()
    targets = batch["labels"].long()
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = input_ids != 0
    return input_ids, targets, attention_mask.bool(), batch.get("target_positions")

```

### `submission_validation.py`

```python
"""Pure source-policy checks shared by the CLI, service, and evaluator."""

from __future__ import annotations

import ast


FORBIDDEN_SUBMISSION_IMPORTS = {"data", "model", "optim"}


def validate_submission_source(
    filename: str,
    source: str,
    max_bytes: int,
    *,
    required_filename: str | None = "submission.py",
) -> str:
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if required_filename is not None and basename != required_filename:
        raise ValueError(f"submission file must be named {required_filename}")
    if not basename.lower().endswith(".py"):
        raise ValueError("submit exactly one .py file")
    if not source.strip():
        raise ValueError("submission file is empty")
    if len(source.encode("utf-8")) > max_bytes:
        raise ValueError(f"submission exceeds the {max_bytes // 1024} KiB limit")
    try:
        tree = ast.parse(source, filename=basename)
    except SyntaxError as exc:
        raise ValueError(f"submission is not valid Python: {exc.msg}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported = [node.module]
        else:
            continue
        for name in imported:
            root = name.partition(".")[0]
            if root in FORBIDDEN_SUBMISSION_IMPORTS:
                raise ValueError(
                    f"submission must be self-contained and may not import {root}"
                )
    return basename

```

### `service/tiers.py`

```python
"""Server-owned compute tier and dataset catalog."""

from __future__ import annotations

from dataclasses import dataclass


RESULT_WAIT_GRACE_SECONDS = 60


@dataclass(frozen=True)
class DatasetOption:
    id: str
    label: str
    manifest_filename: str


@dataclass(frozen=True)
class ComputeTier:
    id: str
    label: str
    training_seconds: int
    daily_attempts: int
    evaluator_timeout_seconds: int
    datasets: tuple[DatasetOption, ...]

    @property
    def evaluation_seconds(self) -> int:
        return self.training_seconds // 2

    @property
    def run_deadline_seconds(self) -> int:
        return self.evaluator_timeout_seconds + RESULT_WAIT_GRACE_SECONDS


TIERS: tuple[ComputeTier, ...] = (
    ComputeTier(
        id="easy",
        label="Easy",
        training_seconds=60,
        daily_attempts=60,
        evaluator_timeout_seconds=390,
        datasets=(
            DatasetOption("e1", "E1 · Fixed N=323, T=1/2/3", "h100_easy_e1.json"),
            DatasetOption("e2", "E2 · Fixed N=899, T=1/2/4", "h100_easy_e2.json"),
            DatasetOption("e3", "E3 · 10–11 bit N, fixed T=2", "h100_easy_e3.json"),
            DatasetOption("e4", "E4 · 11–12 bit N, fixed T=2", "h100_easy_e4.json"),
            DatasetOption("e5", "E5 · 10–11 bit N, T=1/2/3", "h100_easy_e5.json"),
        ),
    ),
    ComputeTier(
        id="medium",
        label="Medium",
        training_seconds=600,
        daily_attempts=6,
        evaluator_timeout_seconds=1200,
        datasets=(
            DatasetOption("m1", "M1 · Fixed N=10,403, T=4/8/16", "h100_medium_m1.json"),
            DatasetOption("m2", "M2 · Fixed N=38,021, T=4/8/16", "h100_medium_m2.json"),
            DatasetOption("m3", "M3 · 11/13/15 bit N, fixed T=2", "h100_medium_m3.json"),
            DatasetOption("m4", "M4 · 14/18/22 bit N, fixed T=8", "h100_medium_m4.json"),
            DatasetOption("m5", "M5 · 12/14/16 bit N, T=2/4/8", "h100_medium_m5.json"),
        ),
    ),
    ComputeTier(
        id="hard",
        label="Hard",
        training_seconds=3600,
        daily_attempts=1,
        evaluator_timeout_seconds=6120,
        datasets=(
            DatasetOption(
                "h1",
                "H1 · Hidden evaluation",
                "h100_hard_h1.json",
            ),
        ),
    ),
)

TIER_BY_ID = {tier.id: tier for tier in TIERS}


def submission_manifest_timeouts() -> dict[str, int]:
    return {
        dataset.manifest_filename: tier.evaluator_timeout_seconds
        for tier in TIERS
        for dataset in tier.datasets
    }


def resolve_tier_dataset(tier_id: str, dataset_id: str | None) -> tuple[ComputeTier, DatasetOption]:
    tier = TIER_BY_ID.get(tier_id.strip().lower())
    if tier is None:
        raise ValueError(f"unknown tier {tier_id!r}; choose easy, medium, or hard")

    normalized_dataset = (dataset_id or "").strip().lower()
    if tier.id == "hard" and not normalized_dataset:
        normalized_dataset = "h1"
    for dataset in tier.datasets:
        if dataset.id == normalized_dataset:
            return tier, dataset
    choices = ", ".join(dataset.id for dataset in tier.datasets)
    raise ValueError(f"dataset for {tier.label} must be one of: {choices}")


def tier_public_payload(tier: ComputeTier) -> dict:
    return {
        "id": tier.id,
        "label": tier.label,
        "training_seconds": tier.training_seconds,
        "evaluation_seconds": tier.evaluation_seconds,
        "daily_attempts": tier.daily_attempts,
        "datasets": [
            {"id": dataset.id, "label": dataset.label}
            for dataset in tier.datasets
        ],
    }

```

### `scripts/generate_datasets.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Squaring-mod difficulty ladder.
#
# Every tier uses separate prompt and output tensors, so models can attend
# bidirectionally over the complete prompt. Easy and medium use prompt-level IID
# splits. Every (N, x, T) prompt is still unique; these tiers intentionally
# measure interpolation over problem families seen in training.
# Runtime and score calibration must be measured against this separate-output
# representation; results from the former causal-LM datasets are not comparable.

# ---------------------------------------------------------------------------
# Easy: five datasets using the one-minute training budget.
# ---------------------------------------------------------------------------

# E1: tiny fixed N with three ID depths.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123 \
  --fixed_p 17 --fixed_q 19 \
  --time_steps '[1,2,3]' --ood_time_steps '[6]' \
  --examples_per_setting 250 --ood_examples_per_setting 100 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E2: larger fixed N and geometric ID depths.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_899_t124 \
  --fixed_p 29 --fixed_q 31 \
  --time_steps '[1,2,4]' --ood_time_steps '[7]' \
  --examples_per_setting 800 --ood_examples_per_setting 300 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E3: sampled N at fixed T over two small, exactly auditable bit cells.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_t_b1011_t2 \
  --modulus_bits '[10,11]' --fixed_time_steps 2 \
  --ood_time_steps '[4]' \
  --examples_per_setting 2000 --ood_examples_per_setting 400 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E4: one bit harder than E3 with twice the per-cell row budget.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_t_b1112_t2 \
  --modulus_bits '[11,12]' --fixed_time_steps 2 \
  --ood_time_steps '[4]' \
  --examples_per_setting 4000 --ood_examples_per_setting 600 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E5: joint N/T conditioning at small scale.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_variable_b1011_t123 \
  --modulus_bits '[10,11]' \
  --time_steps '[1,2,3]' --ood_time_steps '[6]' \
  --examples_per_setting 1000 --ood_examples_per_setting 300 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# ---------------------------------------------------------------------------
# Medium: five datasets using the ten-minute training budget.
# ---------------------------------------------------------------------------

# M1: 14-bit fixed N with a geometric T schedule.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_n_10403_t4816 \
  --fixed_p 101 --fixed_q 103 \
  --time_steps '[4,8,16]' --ood_time_steps '[32]' \
  --examples_per_setting 10000 --ood_examples_per_setting 3000 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M2: 16-bit fixed N and a 95k-row complete dataset.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_n_38021_t4816 \
  --fixed_p 193 --fixed_q 197 \
  --time_steps '[4,8,16]' --ood_time_steps '[32]' \
  --examples_per_setting 30000 --ood_examples_per_setting 5000 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M3: sampled N, fixed T, spanning 11-15 bits.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_t_b111315_t2 \
  --modulus_bits '[11,13,15]' --fixed_time_steps 2 \
  --ood_time_steps '[4]' \
  --examples_per_setting 8000 --ood_examples_per_setting 1000 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M4: sampled N, fixed T, with larger 14-22 bit moduli.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_t_b141822_t8 \
  --modulus_bits '[14,18,22]' --fixed_time_steps 8 \
  --ood_time_steps '[16]' \
  --examples_per_setting 30000 --ood_examples_per_setting 3000 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M5: joint N/T conditioning across nine balanced ID cells.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_variable_b121416_t248 \
  --modulus_bits '[12,14,16]' \
  --time_steps '[2,4,8]' --ood_time_steps '[16]' \
  --examples_per_setting 10000 --ood_examples_per_setting 1000 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

```

### `data/squaring_mod.py` (TOKEN_IDS + collate; full file is 958 lines at this path)

```python
"""Tokenized repeated modular-squaring dataset generation and loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
import random
from pathlib import Path
from typing import Any

import torch

from .counting import (
    Record,
    TokenizedCountingDataset,
    collate_tokenized_counting,
    compute_split_counts,
    digit_token as counting_digit_token,
    load_counting_dataset_config,
    number_tokens as counting_number_tokens,
    write_dataset_config,
    write_split_files,
)


TOKEN_IDS: dict[str, int] = {
    "PAD": 0,
    "BOS": 1,
    "N": 2,
    "X": 3,
    "T": 4,
    "ANS": 5,
    "EOS": 6,
}
DIGIT_OFFSET = 7
NUM_DIGITS = 10
VOCAB_SIZE = DIGIT_OFFSET + NUM_DIGITS

# A deliberately small, deterministic suite used when DataConfig.data_root is
# unset. It is intended for end-to-end evaluator testing, not as the eventual
# scored squaring-mod benchmark.
SMOKE_FIXED_P = 11
SMOKE_FIXED_Q = 13
SMOKE_TIME_STEPS = (1, 2, 3)
SMOKE_OOD_TIME_STEPS = (4,)
SMOKE_EXAMPLES_PER_SETTING = 100
# N=143, x<=142, and one-digit T produce at most ten prompt tokens in the
# separate-input/output representation.
SMOKE_MAX_SEQ_LEN = 10
ID_SPLITS: tuple[str, ...] = ("train", "test")


class SquaringModTokenizedDataset(TokenizedCountingDataset):
    """JSONL-backed repeated modular-squaring dataset."""


def load_squaring_mod_dataset_config(root: str | Path) -> dict[str, Any]:
    return load_counting_dataset_config(root)


def collate_squaring_mod(batch: list[dict[str, Any]]) -> dict[str, Any]:
    uses_separate_output = [
        len(item["labels"]) != len(item["input_ids"])
        for item in batch
    ]
    if not any(uses_separate_output):
        return collate_tokenized_counting(batch, TOKEN_IDS["PAD"])
    if not all(uses_separate_output):
        raise ValueError("squaring_mod batch cannot mix causal_lm and separate_input_output rows")

    max_input_len = max(len(item["input_ids"]) for item in batch)
    max_target_len = max(len(item["labels"]) for item in batch)
    input_ids = torch.full(
        (len(batch), max_input_len), TOKEN_IDS["PAD"], dtype=torch.long
    )
    labels = torch.full((len(batch), max_target_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_input_len), dtype=torch.bool)
    target_positions = torch.full(
        (len(batch), max_target_len), -1, dtype=torch.long
    )

    for row, item in enumerate(batch):
        item_input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        item_labels = torch.tensor(item["labels"], dtype=torch.long)
        input_len = item_input_ids.numel()
        target_len = item_labels.numel()
        if target_len > input_len:
            raise ValueError("squaring_mod output cannot be longer than its input")
        input_ids[row, :input_len] = item_input_ids
        labels[row, :target_len] = item_labels
        attention_mask[row, :input_len] = True
        target_positions[row, :target_len] = torch.arange(
            input_len - target_len, input_len, dtype=torch.long
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "target_positions": target_positions,
    }


@dataclass(frozen=True)
class SquaringModGenerationConfig:
    output_dir: str
    modulus_bits: list[int] = field(default_factory=lambda: [32])
    fixed_p: int | None = None
    fixed_q: int | None = None
    time_steps: list[int] = field(default_factory=lambda: [16])
    fixed_time_steps: int | None = None
    examples_per_setting: int = 100
    seed: int = 45
    train_fraction: float = 0.8
    test_fraction: float = 0.2
    ood_time_steps: list[int] = field(default_factory=list)
    ood_examples_per_setting: int | None = None
    generator_family: str = "rsa_repeated_squaring"
    separate_input_output: bool = False
    split_group: str = "prompt"
```

### `benchmark/runner.py` (full file 666 lines). Excerpts: scoring splits, batch resolution, forward/loss, train loop deadline.

```python
"""Evaluator-owned runner for the One Layer Deeper competition."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data import (
    infer_max_seq_len,
    infer_vocab_size,
    make_dataloaders,
)
from .api import ModelSpec, OptimizerBundle, OptimizerSpec, Submission
from .batches import prepare_batch
from .manifest import BenchmarkManifest, load_manifest
from .metrics import MetricRecorder
from .validation import (
    assert_state_versions_unchanged,
    capture_state_versions,
    lint_submission_source,
    validate_model_state,
    validate_optimizer,
    validate_submission,
)


EVALUATION_TIME_FRACTION = 0.5
SCORING_SPLIT_PRIORITY = ("test", "ood", "ood_t", "ood_n_t")
NON_SCORING_SPLITS = frozenset(("train", "eval"))


def _scoring_split_names(dataloaders) -> tuple[str, ...]:
    """Return deterministic scored splits for final measurement."""

    available = set(dataloaders) - NON_SCORING_SPLITS
    prioritized = [name for name in SCORING_SPLIT_PRIORITY if name in available]
    remaining = sorted(available - set(prioritized))
    return tuple((*prioritized, *remaining))


# ...
def _resolve_batch_sizes(
    submission: Submission,
    manifest: BenchmarkManifest,
) -> tuple[int, int]:
    batch_size = submission.batch_size or manifest.data.batch_size
    eval_batch_size = (
        submission.eval_batch_size
        or submission.batch_size
        or manifest.data.eval_batch_size
        or manifest.data.batch_size
    )
    return batch_size, eval_batch_size

# ...
def _loss_and_accuracy(
    model: nn.Module,
    batch,
    manifest: BenchmarkManifest,
    device: torch.device,
    *,
    training_loss=None,
) -> tuple[torch.Tensor, float, int, int]:
    input_ids, targets, attention_mask, target_positions = prepare_batch(
        batch,
        device,
    )

    with _autocast(manifest, device):
        logits, auxiliary = model(
            input_ids,
            attention_mask=attention_mask,
        )
        if (
            logits.ndim != 3
            or logits.shape[:2] != input_ids.shape
            or logits.shape[-1] != model.config.vocab_size
        ):
            raise ValueError(
                "language-model logits must have shape "
                "(batch, sequence, vocab_size)"
            )
        if target_positions is None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    "causal language-model targets must match the input shape"
                )
            token_logits = logits[:, :-1, :].float()
            token_targets = targets[:, 1:]
        else:
            if target_positions.shape != targets.shape:
                raise ValueError(
                    "target_positions must have the same shape as targets"
                )
            valid_positions = target_positions[targets != -100]
            if (
                (valid_positions < 0).any().item()
                or (valid_positions >= input_ids.shape[1]).any().item()
            ):
                raise ValueError("target position is outside the input sequence")
            batch_indices = torch.arange(logits.shape[0], device=device)[:, None]
            token_logits = logits[
                batch_indices,
                target_positions.clamp_min(0),
            ].float()
            token_targets = targets

        valid = token_targets != -100
        if not valid.any().item():
            raise ValueError("batch contains no valid language-model targets")
        loss_logits = token_logits[valid]
        loss_labels = token_targets[valid]
        if training_loss is None:
            loss = F.cross_entropy(loss_logits, loss_labels)
        else:
            loss = training_loss(loss_logits, loss_labels, auxiliary)

        token_predictions = token_logits.argmax(dim=-1)
        rows_with_targets = valid.any(dim=1)
        exact_rows = (
            (token_predictions == token_targets) | ~valid
        ).all(dim=1)[rows_with_targets]
        example_count = int(rows_with_targets.sum().item())
        loss_weight = int(valid.sum().item())

        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise TypeError("training_loss must return one scalar tensor")
        if loss.device != device:
            raise ValueError(f"training_loss must return a tensor on {device}")
        if training_loss is not None and not loss.requires_grad:
            raise ValueError("training_loss result must be differentiable")

    exact_accuracy = exact_rows.float().mean().item()
    return loss, exact_accuracy, example_count, loss_weight


def _train(
    *,
    raw_model: nn.Module,
    train_model: nn.Module,
    training_loss,
    bundle: OptimizerBundle,
    dataloader,
    manifest: BenchmarkManifest,
    device: torch.device,
    started_at: float,
    deadline: float,
    budget_seconds: float,
    max_steps: int,
    seed: int,
    metric_recorder: MetricRecorder | None = None,
) -> tuple[float | None, int, float, int]:
    optimizer = bundle.optimizer
    raw_model.train()
    validate_optimizer(bundle, raw_model, device)
    iterator = iter(dataloader)
    final_loss = None
    final_accuracy = None
    completed_steps = 0
    last_metric_step = 0
    optimizer_state_elements = 0

    for step in range(1, max_steps + 1):
        if time.monotonic() >= deadline:
            break
        validate_model_state(raw_model, manifest.model_state, device)
        batch, iterator = _next_batch(iterator, dataloader)
        optimizer.zero_grad(set_to_none=True)
        loss, accuracy, _, _ = _loss_and_accuracy(
            train_model,
            batch,
            manifest,
            device,
            training_loss=training_loss,
        )
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError(f"non-finite training loss at step {step}")
        loss.backward()
        if manifest.runtime.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), manifest.runtime.grad_clip
            )
        optimizer.step()
        if bundle.scheduler is not None:
            bundle.scheduler.step()

```

### `benchmark/validation.py`

```python
"""Structural checks for official submissions and their runtime state."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch import nn

from .api import OptimizerBundle, Submission, model_state_tensors
from .manifest import ModelStateSpec
from submission_validation import validate_submission_source


def lint_submission_source(path: Path) -> None:
    """Apply the shared source policy before importing an evaluator temp file."""

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("submission.py must be UTF-8") from exc
    validate_submission_source(path.name, source, 256 * 1024, required_filename=None)


def validate_submission(submission: Submission) -> None:
    if not callable(submission.build_model) or not callable(submission.build_optimizer):
        raise TypeError("submission factories must be callable")
    if submission.training_loss is not None and not callable(submission.training_loss):
        raise TypeError("submission training_loss must be callable when provided")
    for name in ("batch_size", "eval_batch_size", "max_steps"):
        value = getattr(submission, name)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"submission {name} must be a positive integer when provided")


def validate_model_state(
    model: nn.Module,
    state_spec: ModelStateSpec,
    device: torch.device,
) -> int:
    state = list(model_state_tensors(model))
    if not state:
        raise ValueError("model must contain persistent parameter or buffer state")
    wrong_device = [name for name, value in state if value.device != device]
    if wrong_device:
        raise ValueError(f"model state is not on {device}: {wrong_device[:5]}")
    elements = sum(value.numel() for _, value in state)
    if elements > state_spec.maximum_elements:
        raise ValueError(
            f"model persistent state ({elements:,}) exceeds maximum ({state_spec.maximum_elements:,})"
        )
    return elements


def validate_optimizer(
    bundle: OptimizerBundle,
    model: nn.Module,
    device: torch.device,
) -> int:
    if not isinstance(bundle, OptimizerBundle):
        raise TypeError("build_optimizer must return benchmark.OptimizerBundle")
    optimizer = bundle.optimizer
    for method_name in ("zero_grad", "step", "state_dict"):
        if not callable(getattr(optimizer, method_name, None)):
            raise TypeError(
                "OptimizerBundle.optimizer must implement zero_grad, step, and state_dict"
            )
    if not isinstance(getattr(optimizer, "param_groups", None), list):
        raise TypeError("OptimizerBundle.optimizer must expose a param_groups list")
    if bundle.scheduler is not None and not callable(
        getattr(bundle.scheduler, "step", None)
    ):
        raise TypeError("OptimizerBundle.scheduler must expose step()")

    expected = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    actual = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    expected_ids = Counter(id(parameter) for parameter in expected)
    actual_ids = Counter(id(parameter) for parameter in actual)
    if expected_ids != actual_ids:
        raise ValueError(
            "optimizer must contain every trainable model parameter exactly once"
        )

    child_optimizers = getattr(optimizer, "optimizers", [optimizer])
    state_elements = 0
    for child_optimizer in child_optimizers:
        child_state = getattr(child_optimizer, "state", {})
        for state in child_state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    if value.device != device:
                        raise ValueError(f"optimizer state tensor is not on {device}")
                    state_elements += value.numel()
    return state_elements


def capture_state_versions(model: nn.Module) -> dict[int, int]:
    return {id(value): value._version for _, value in model_state_tensors(model)}


def assert_state_versions_unchanged(model: nn.Module, before: dict[int, int]) -> None:
    after = capture_state_versions(model)
    if after != before:
        raise ValueError(
            "model parameters or persistent buffers changed during evaluation"
        )

```

### Example Easy e1 manifest `benchmark/manifests/h100_easy_e1.json`

```json
{
  "name": "squaring-mod-easy-e1",
  "data": {
    "kind": "squaring_mod",
    "data_root": "data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123",
    "batch_size": 512,
    "eval_batch_size": 512,
    "shuffle_train": true,
    "shuffle_eval": false,
    "num_workers": 2,
    "pin_memory": true,
    "drop_last": true,
    "seed": 45
  },
  "runtime": {
    "device": "cuda:0",
    "dtype": "bfloat16",
    "amp": true,
    "compile": false,
    "total_training_time_seconds": 60,
    "max_steps": 1000000,
    "seeds": [
      74
    ],
    "grad_clip": 1,
    "log_every": 100
  },
  "model_state": {
    "maximum_elements": 500000000
  }
}

```

### Example Medium m5 manifest `benchmark/manifests/h100_medium_m5.json`

```json
{
  "name": "squaring-mod-medium-m5",
  "data": {
    "kind": "squaring_mod",
    "data_root": "data/generated/squaring_mod_new11_medium_bidirectional_variable_b121416_t248",
    "batch_size": 512,
    "eval_batch_size": 512,
    "shuffle_train": true,
    "shuffle_eval": false,
    "num_workers": 2,
    "pin_memory": true,
    "drop_last": true,
    "seed": 45
  },
  "runtime": {
    "device": "cuda:0",
    "dtype": "bfloat16",
    "amp": true,
    "compile": false,
    "total_training_time_seconds": 600,
    "max_steps": 1000000,
    "seeds": [
      74
    ],
    "grad_clip": 1,
    "log_every": 100
  },
  "model_state": {
    "maximum_elements": 500000000
  }
}

```

### Forward depth may be input-dependent?

Primary sources:
- README Rules item 3: “Recurrence, adaptive computation, and depth curricula are allowed.”
- README Submission contract: “If the model should behave differently during evaluation, use PyTorch's inherited `self.training` flag inside `forward`.”
- Runner calls `model(input_ids, attention_mask=attention_mask)` once per train/eval step (`benchmark/runner.py` `_loss_and_accuracy`). Whether depth inside that forward depends on tokens (e.g. on T) is not further constrained in the pasted API beyond Rules 3 and 10.
- Whether branching depth on decoded T is permitted beyond Rule 10: **UNDOCUMENTED** in README/problem page (no explicit yes/no beyond Rule 10’s ban on task-specific solvers).

## 3. Tier table (verbatim labels + generator script + split semantics)

### 3.1 From `service/tiers.py` DatasetOption labels

| Tier | training_seconds | daily_attempts | evaluation_seconds (= training//2) | dataset id | label |
|------|------------------|----------------|--------------------------------------|------------|-------|
| easy | 60 | 60 | 30 | e1 | E1 · Fixed N=323, T=1/2/3 |
| easy | 60 | 60 | 30 | e2 | E2 · Fixed N=899, T=1/2/4 |
| easy | 60 | 60 | 30 | e3 | E3 · 10–11 bit N, fixed T=2 |
| easy | 60 | 60 | 30 | e4 | E4 · 11–12 bit N, fixed T=2 |
| easy | 60 | 60 | 30 | e5 | E5 · 10–11 bit N, T=1/2/3 |
| medium | 600 | 6 | 300 | m1 | M1 · Fixed N=10,403, T=4/8/16 |
| medium | 600 | 6 | 300 | m2 | M2 · Fixed N=38,021, T=4/8/16 |
| medium | 600 | 6 | 300 | m3 | M3 · 11/13/15 bit N, fixed T=2 |
| medium | 600 | 6 | 300 | m4 | M4 · 14/18/22 bit N, fixed T=8 |
| medium | 600 | 6 | 300 | m5 | M5 · 12/14/16 bit N, T=2/4/8 |
| hard | 3600 | 1 | 1800 | h1 | H1 · Hidden evaluation |

### 3.2 Generator parameters from `scripts/generate_datasets.sh` (Easy/Medium public data)

| ID | fixed N / bits | ID time_steps | ood_time_steps | examples_per_setting | ood_examples_per_setting | train_fraction | test_fraction |
|----|----------------|---------------|----------------|----------------------|--------------------------|----------------|---------------|
| e1 | N=323 (p=17,q=19) | [1,2,3] | [6] | 250 | 100 | 0.8 | 0.2 |
| e2 | N=899 (p=29,q=31) | [1,2,4] | [7] | 800 | 300 | 0.8 | 0.2 |
| e3 | bits [10,11] | fixed T=2 | [4] | 2000 | 400 | 0.8 | 0.2 |
| e4 | bits [11,12] | fixed T=2 | [4] | 4000 | 600 | 0.8 | 0.2 |
| e5 | bits [10,11] | [1,2,3] | [6] | 1000 | 300 | 0.8 | 0.2 |
| m1 | N=10403 (p=101,q=103) | [4,8,16] | [32] | 10000 | 3000 | 0.9 | 0.1 |
| m2 | N=38021 (p=193,q=197) | [4,8,16] | [32] | 30000 | 5000 | 0.9 | 0.1 |
| m3 | bits [11,13,15] | fixed T=2 | [4] | 8000 | 1000 | 0.9 | 0.1 |
| m4 | bits [14,18,22] | fixed T=8 | [16] | 30000 | 3000 | 0.9 | 0.1 |
| m5 | bits [12,14,16] | [2,4,8] | [16] | 10000 | 1000 | 0.9 | 0.1 |

### 3.3 Split semantics (documented vs UNDOCUMENTED)

From `scripts/generate_datasets.sh` header (verbatim):

> Every tier uses separate prompt and output tensors, so models can attend bidirectionally over the complete prompt. Easy and medium use prompt-level IID splits. Every (N, x, T) prompt is still unique; these tiers intentionally measure interpolation over problem families seen in training.

From problem page (verbatim):

> Hard equally averages exact accuracy on its hidden test, held-out-depth, and jointly held-out modulus/depth splits across seeds. Easy and Medium equally average test and their merged out-of-distribution split.

| Question | Status |
|----------|--------|
| Easy/Medium `test` vs `train` | Documented as prompt-level IID split (`split_group prompt`); fractions in generator table. |
| Easy/Medium `ood` | Documented via `ood_time_steps` in generator (held-out **T** values listed above). |
| Held-out N on Easy/Medium `ood`: same-range-unseen vs larger bits? | **UNDOCUMENTED** as a separate N-OOD split for Easy/Medium public generators (OOD knobs in the public script are `ood_time_steps`). |
| Hard held-out N / T details | Problem page names hidden test, held-out-depth, jointly held-out modulus/depth; bit ranges and T values **UNDOCUMENTED** publicly. |
| Hard recurrence definition | Problem page: may change aspects of the recurrence; exact form **UNDOCUMENTED**. |

## 4. Example rows (serialized tokens)

TOKEN_IDS (`data/squaring_mod.py`): PAD=0 BOS=1 N=2 X=3 T=4 ANS=5 EOS=6; digits 0–9 → ids 7–16.

### 4.1 Problem-page conceptual serialization

```
N77X2T4ANS9
```

### 4.2 Local e1-like sample rows (see note in each block)

#### split=train modulus=323 x=140 T=1 y=220
- source: `solving/experiments/data_samples/e1_like_n323_t123/train.jsonl`
- note: Locally generated with fixed_p=17 fixed_q=19 time_steps [1,2,3]; NOT the official hosted e1 row count (official uses examples_per_setting=250). Tokenization matches competition TOKEN_IDS.
- input_ids: `[2, 10, 9, 10, 3, 8, 11, 7, 4, 8]`
- input_tokens: `N 3 2 3 X 1 4 0 T 1`
- labels: `[9, 9, 7]`
- label_tokens: `2 2 0`

#### split=train modulus=323 x=214 T=1 y=253
- source: `solving/experiments/data_samples/e1_like_n323_t123/train.jsonl`
- note: Locally generated with fixed_p=17 fixed_q=19 time_steps [1,2,3]; NOT the official hosted e1 row count (official uses examples_per_setting=250). Tokenization matches competition TOKEN_IDS.
- input_ids: `[2, 10, 9, 10, 3, 9, 8, 11, 4, 8]`
- input_tokens: `N 3 2 3 X 2 1 4 T 1`
- labels: `[9, 12, 10]`
- label_tokens: `2 5 3`

#### split=train modulus=323 x=250 T=1 y=161
- source: `solving/experiments/data_samples/e1_like_n323_t123/train.jsonl`
- note: Locally generated with fixed_p=17 fixed_q=19 time_steps [1,2,3]; NOT the official hosted e1 row count (official uses examples_per_setting=250). Tokenization matches competition TOKEN_IDS.
- input_ids: `[2, 10, 9, 10, 3, 9, 12, 7, 4, 8]`
- input_tokens: `N 3 2 3 X 2 5 0 T 1`
- labels: `[8, 13, 8]`
- label_tokens: `1 6 1`

#### split=train modulus=323 x=132 T=1 y=305
- source: `solving/experiments/data_samples/e1_like_n323_t123/train.jsonl`
- note: Locally generated with fixed_p=17 fixed_q=19 time_steps [1,2,3]; NOT the official hosted e1 row count (official uses examples_per_setting=250). Tokenization matches competition TOKEN_IDS.
- input_ids: `[2, 10, 9, 10, 3, 8, 10, 9, 4, 8]`
- input_tokens: `N 3 2 3 X 1 3 2 T 1`
- labels: `[10, 7, 12]`
- label_tokens: `3 0 5`

#### split=test modulus=323 x=6 T=1 y=36
- source: `solving/experiments/data_samples/e1_like_n323_t123/test.jsonl`
- note: Locally generated with fixed_p=17 fixed_q=19 time_steps [1,2,3]; NOT the official hosted e1 row count (official uses examples_per_setting=250). Tokenization matches competition TOKEN_IDS.
- input_ids: `[2, 10, 9, 10, 3, 13, 4, 8]`
- input_tokens: `N 3 2 3 X 6 T 1`
- labels: `[10, 13]`
- label_tokens: `3 6`

#### split=test modulus=323 x=59 T=1 y=251
- source: `solving/experiments/data_samples/e1_like_n323_t123/test.jsonl`
- note: Locally generated with fixed_p=17 fixed_q=19 time_steps [1,2,3]; NOT the official hosted e1 row count (official uses examples_per_setting=250). Tokenization matches competition TOKEN_IDS.
- input_ids: `[2, 10, 9, 10, 3, 12, 16, 4, 8]`
- input_tokens: `N 3 2 3 X 5 9 T 1`
- labels: `[9, 12, 8]`
- label_tokens: `2 5 1`

#### split=test modulus=323 x=146 T=1 y=321
- source: `solving/experiments/data_samples/e1_like_n323_t123/test.jsonl`
- note: Locally generated with fixed_p=17 fixed_q=19 time_steps [1,2,3]; NOT the official hosted e1 row count (official uses examples_per_setting=250). Tokenization matches competition TOKEN_IDS.
- input_ids: `[2, 10, 9, 10, 3, 8, 11, 13, 4, 8]`
- input_tokens: `N 3 2 3 X 1 4 6 T 1`
- labels: `[10, 9, 8]`
- label_tokens: `3 2 1`

#### split=ood modulus=323 x=199 T=4 y=137
- source: `solving/experiments/data_samples/e1_like_n323_t123/ood.jsonl`
- note: Locally generated with fixed_p=17 fixed_q=19 time_steps [1,2,3]; NOT the official hosted e1 row count (official uses examples_per_setting=250). Tokenization matches competition TOKEN_IDS.
- input_ids: `[2, 10, 9, 10, 3, 8, 16, 16, 4, 11]`
- input_tokens: `N 3 2 3 X 1 9 9 T 4`
- labels: `[8, 10, 14]`
- label_tokens: `1 3 7`

#### split=ood modulus=323 x=275 T=4 y=137
- source: `solving/experiments/data_samples/e1_like_n323_t123/ood.jsonl`
- note: Locally generated with fixed_p=17 fixed_q=19 time_steps [1,2,3]; NOT the official hosted e1 row count (official uses examples_per_setting=250). Tokenization matches competition TOKEN_IDS.
- input_ids: `[2, 10, 9, 10, 3, 9, 14, 12, 4, 11]`
- input_tokens: `N 3 2 3 X 2 7 5 T 4`
- labels: `[8, 10, 14]`
- label_tokens: `1 3 7`

#### split=ood modulus=323 x=318 T=4 y=35
- source: `solving/experiments/data_samples/e1_like_n323_t123/ood.jsonl`
- note: Locally generated with fixed_p=17 fixed_q=19 time_steps [1,2,3]; NOT the official hosted e1 row count (official uses examples_per_setting=250). Tokenization matches competition TOKEN_IDS.
- input_ids: `[2, 10, 9, 10, 3, 10, 8, 15, 4, 11]`
- input_tokens: `N 3 2 3 X 3 1 8 T 4`
- labels: `[10, 12]`
- label_tokens: `3 5`

### 4.3 Official hosted Easy e2–e5 / Medium / Hard JSONL rows

**Unavailable in the public clone.** Official `data/generated/…` trees are produced by `scripts/generate_datasets.sh` on the evaluator / after local generation; they are not shipped. Hard rows are not public.

## 5. Results table (JSONL + joined configs + recovered/UNRECOVERED)

Exact accuracies are fractions in [0,1] (JSONL native) or converted from percent statements in logs/NOTES. `train_em_last` is the last `type=train` exact_accuracy when present.

**Status column**
- `JSONL` — numbers from `solving/experiments/metrics/<run>.jsonl`
- `RECOVERED` — no JSONL; numbers copied verbatim from cited log/NOTE (converted %→fraction where needed)
- `UNRECOVERED` — card exists or is referenced, but mean/test/ood not found in JSONL or logs

**Config columns** joined from `solving/experiments/<date>_<card>/{submission.py,config.json,NOTE.md}` (not present in evaluator JSONL). Embedding tags are parsed from code (docstrings/comments stripped), not from prose.

| run_name | card | dataset | status | metrics_source | experiment_dir | d_model | num_heads | layers_or_K | embedding_scheme | optimizer | scheduler | batch_size | params | config_change | completed_steps | training_seconds | mean_exact_accuracy | train_em_last | test_em | ood_em | ood_t_em | ood_n_em | ood_n_t_em | recovery_note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b0_e1 | b0_transformer | e1 | JSONL | `solving/experiments/metrics/b0_e1.jsonl` | solving/experiments/2026-07-21_b0_transformer | 128 | 4 | — | token+abs_pos | AdamW(lr=1e-3,wd=0.1) | — | — | — | baseline TF | 261 | 60.1 | 0.01 | 0.08 | 0.02 | 0 | — | — | — | — |
| b0_max_e1 | b0_transformer_max | e1 | JSONL | `solving/experiments/metrics/b0_max_e1.jsonl` | solving/experiments/2026-07-21_b0_transformer_max | 64 | 4 | — | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | d=64 + sched + bs | 557 | 60.1 | 0.013 | 0.055 | 0.027 | 0 | — | — | — | — |
| b1_e1 | b1_mlp | e1 | JSONL | `solving/experiments/metrics/b1_e1.jsonl` | solving/experiments/2026-07-21_b1_mlp | 128 | — | — | token+abs_pos | AdamW(lr=1e-3,wd=0.1) | — | — | — | MLP bag | 287 | 60.1 | 0.003 | 0.037 | 0.007 | 0 | — | — | — | — |
| b1_max_e1 | b1_mlp_max | e1 | JSONL | `solving/experiments/metrics/b1_max_e1.jsonl` | solving/experiments/2026-07-21_b1_mlp_max | 64 | — | — | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | max recipe | 585 | 60.1 | 0.01 | 0.023 | 0.02 | 0 | — | — | — | — |
| b2_e1 | b2_rnn | e1 | JSONL | `solving/experiments/metrics/b2_e1.jsonl` | solving/experiments/2026-07-21_b2_rnn | 128 | — | — | token+abs_pos | AdamW(lr=1e-3,wd=0.1) | — | — | — | BiGRU | 258 | 60.1 | 0.007 | 0.211 | 0.013 | 0 | — | — | — | — |
| b2_max_e1 | b2_rnn_max | e1 | JSONL | `solving/experiments/metrics/b2_max_e1.jsonl` | solving/experiments/2026-07-21_b2_rnn_max | 64 | — | — | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | max recipe | 555 | 60.1 | 0.01 | 0.188 | 0.02 | 0 | — | — | — | — |
| depth_d32_act_e1 | depth_d32_act | e1 | JSONL | `solving/experiments/metrics/depth_d32_act_e1.jsonl` | solving/experiments/2026-07-21_depth_d32_act | 32 | 4 | K=8;soft_ACT | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | soft ACT | 397 | 60.1 | 0.038 | 0.082 | 0.027 | 0.05 | — | — | — | — |
| depth_d32_act_e5 | depth_d32_act | e5 | JSONL | `solving/experiments/metrics/depth_d32_act_e5.jsonl` | solving/experiments/2026-07-21_depth_d32_act | 32 | 4 | K=8;soft_ACT | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | soft ACT | 1798 | 60 | 0.008 | 0.008 | 0.007 | 0.008 | — | — | — | — |
| depth_d32_k2_e1 | depth_d32_k2 | e1 | JSONL | `solving/experiments/metrics/depth_d32_k2_e1.jsonl` | solving/experiments/2026-07-21_depth_d32_k2 | 32 | 4 | K=2 | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | K=2 | 383 | 60.1 | 0.062 | 0.062 | 0.033 | 0.09 | — | — | — | — |
| depth_d32_k2_e5 | depth_d32_k2 | e5 | JSONL | `solving/experiments/metrics/depth_d32_k2_e5.jsonl` | solving/experiments/2026-07-21_depth_d32_k2 | 32 | 4 | K=2 | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | K=2 | 2503 | 60 | 0.005 | 0.027 | 0.003 | 0.007 | — | — | — | — |
| depth_d32_k2_ut_e1 | depth_d32_k2_ut | e1 | JSONL | `solving/experiments/metrics/depth_d32_k2_ut_e1.jsonl` | solving/experiments/2026-07-21_depth_d32_k2_ut | 32 | 4 | K=2 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | UT depth emb K=2 | 393 | 60.1 | 0.065 | 0.047 | 0.04 | 0.09 | — | — | — | — |
| depth_d32_k2_ut_e5 | depth_d32_k2_ut | e5 | JSONL | `solving/experiments/metrics/depth_d32_k2_ut_e5.jsonl` | solving/experiments/2026-07-21_depth_d32_k2_ut | 32 | 4 | K=2 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | UT depth emb K=2 | 2359 | 60.2 | 0.007 | 0.02 | 0.008 | 0.005 | — | — | — | — |
| depth_d32_k2_ut_evalk4_e1 | depth_d32_k2_ut_evalk4 | e1 | JSONL | `solving/experiments/metrics/depth_d32_k2_ut_evalk4_e1.jsonl` | solving/experiments/2026-07-21_depth_d32_k2_ut_evalk4 | 32 | 4 | K=4;train2/eval4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | train K2 eval K4 | 609 | 60.1 | 0.068 | 0.043 | 0.047 | 0.09 | — | — | — | — |
| depth_d32_k2_ut_evalk4_e1_r2 | depth_d32_k2_ut_evalk4 | e1 | JSONL | `solving/experiments/metrics/depth_d32_k2_ut_evalk4_e1_r2.jsonl` | solving/experiments/2026-07-21_depth_d32_k2_ut_evalk4 | 32 | 4 | K=4;train2/eval4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | train K2 eval K4 | 407 | 60.2 | 0.068 | 0.074 | 0.047 | 0.09 | — | — | — | — |
| depth_d32_k2_ut_evalk4_e1_r3 | depth_d32_k2_ut_evalk4 | e1 | JSONL | `solving/experiments/metrics/depth_d32_k2_ut_evalk4_e1_r3.jsonl` | solving/experiments/2026-07-21_depth_d32_k2_ut_evalk4 | 32 | 4 | K=4;train2/eval4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | train K2 eval K4 | 409 | 60.2 | 0.068 | 0.031 | 0.047 | 0.09 | — | — | — | — |
| depth_d32_k2_ut_evalk4_e5 | depth_d32_k2_ut_evalk4 | e5 | JSONL | `solving/experiments/metrics/depth_d32_k2_ut_evalk4_e5.jsonl` | solving/experiments/2026-07-21_depth_d32_k2_ut_evalk4 | 32 | 4 | K=4;train2/eval4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | train K2 eval K4 | 3583 | 60 | 0.004 | 0.031 | 0.007 | 0.002 | — | — | — | — |
| depth_d32_k2_ut_evalk4_m5 | depth_d32_k2_ut_evalk4 | m5 | JSONL | `solving/experiments/metrics/depth_d32_k2_ut_evalk4_m5.jsonl` | solving/experiments/2026-07-21_depth_d32_k2_ut_evalk4 | 32 | 4 | K=4;train2/eval4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | train K2 eval K4 | 62770 | 600 | 0.001 | 0.004 | 0.001 | 0.001 | — | — | — | — |
| depth_d32_k3_e1 | depth_d32_k3 | e1 | JSONL | `solving/experiments/metrics/depth_d32_k3_e1.jsonl` | solving/experiments/2026-07-21_depth_d32_k3 | 32 | 4 | K=3 | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | K=3 | 407 | 60.2 | 0.05 | 0.07 | 0.02 | 0.08 | — | — | — | — |
| depth_d32_k3_e5 | depth_d32_k3 | e5 | JSONL | `solving/experiments/metrics/depth_d32_k3_e5.jsonl` | solving/experiments/2026-07-21_depth_d32_k3 | 32 | 4 | K=3 | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | K=3 | 2341 | 60.1 | 0.004 | 0.016 | 0.007 | 0 | — | — | — | — |
| depth_d32_k4_e1 | depth_d32_k4 | e1 | JSONL | `solving/experiments/metrics/depth_d32_k4_e1.jsonl` | solving/experiments/2026-07-21_depth_d32_k4 | 32 | 4 | K=4 | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | shared block ×4 | 471 | 60 | 0.055 | 0.07 | 0.02 | 0.09 | — | — | — | — |
| depth_d32_k4_e5 | depth_d32_k4 | e5 | JSONL | `solving/experiments/metrics/depth_d32_k4_e5.jsonl` | solving/experiments/2026-07-21_depth_d32_k4 | 32 | 4 | K=4 | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | shared block ×4 | 2527 | 60 | 0.008 | 0.023 | 0.011 | 0.005 | — | — | — | — |
| depth_d32_k4_ut_e1 | depth_d32_k4_ut | e1 | JSONL | `solving/experiments/metrics/depth_d32_k4_ut_e1.jsonl` | solving/experiments/2026-07-21_depth_d32_k4_ut | 32 | 4 | K=4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | UT K=4 | 413 | 60.2 | 0.047 | 0.082 | 0.013 | 0.08 | — | — | — | — |
| depth_d32_k4_ut_e5 | depth_d32_k4_ut | e5 | JSONL | `solving/experiments/metrics/depth_d32_k4_ut_e5.jsonl` | solving/experiments/2026-07-21_depth_d32_k4_ut | 32 | 4 | K=4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | UT K=4 | 2275 | 60 | 0.01 | 0.027 | 0.008 | 0.012 | — | — | — | — |
| depth_d32_k4_ut_m1 | depth_d32_k4_ut | m1 | JSONL | `solving/experiments/metrics/depth_d32_k4_ut_m1.jsonl` | solving/experiments/2026-07-21_depth_d32_k4_ut | 32 | 4 | K=4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | UT K=4 | 44993 | 600 | 0.001 | 0.004 | 0.001 | 0 | — | — | — | — |
| depth_d32_k4_ut_m5 | depth_d32_k4_ut | m5 | JSONL | `solving/experiments/metrics/depth_d32_k4_ut_m5.jsonl` | solving/experiments/2026-07-21_depth_d32_k4_ut | 32 | 4 | K=4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | UT K=4 | 51049 | 600 | 0.001 | 0.004 | 0.001 | 0.001 | — | — | — | — |
| depth_d32_k4_ut_optsched_m5 | depth_d32_k4_ut_optsched | m5 | JSONL | `solving/experiments/metrics/depth_d32_k4_ut_optsched_m5.jsonl` | solving/experiments/2026-07-21_depth_d32_k4_ut_optsched | 32 | 4 | K=4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | wallclock_LambdaLR | 256 | — | clamped cosine | 70007 | 600 | 0.002 | 0.004 | 0.001 | 0.002 | — | — | — | — |
| depth_d32_k6_e1 | depth_d32_k6 | e1 | JSONL | `solving/experiments/metrics/depth_d32_k6_e1.jsonl` | solving/experiments/2026-07-21_depth_d32_k6 | 32 | 4 | K=6 | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | K=6 | 411 | 60.2 | 0.045 | 0.062 | 0.02 | 0.07 | — | — | — | — |
| depth_d32_k8_e1 | depth_d32_k8 | e1 | JSONL | `solving/experiments/metrics/depth_d32_k8_e1.jsonl` | solving/experiments/2026-07-21_depth_d32_k8 | 32 | 4 | K=8 | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | K=8 | 413 | 60 | 0.027 | 0.082 | 0.033 | 0.02 | — | — | — | — |
| depth_d32_midloop_k4_e1 | depth_d32_midloop_k4 | e1 | JSONL | `solving/experiments/metrics/depth_d32_midloop_k4_e1.jsonl` | solving/experiments/2026-07-21_depth_d32_midloop_k4 | 32 | 4 | K=4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | middle-only depth | 567 | 60 | 0.008 | 0.309 | 0.007 | 0.01 | — | — | — | — |
| depth_d32_midloop_k4_e5 | depth_d32_midloop_k4 | e5 | JSONL | `solving/experiments/metrics/depth_d32_midloop_k4_e5.jsonl` | solving/experiments/2026-07-21_depth_d32_midloop_k4 | 32 | 4 | K=4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | middle-only depth | 2817 | 60 | 0.008 | 0.074 | 0.009 | 0.007 | — | — | — | — |
| depth_k4_e1 | depth_looped_k4 | e1 | JSONL | `solving/experiments/metrics/depth_k4_e1.jsonl` | solving/experiments/2026-07-21_depth_looped_k4 | 64 | 4 | K=4 | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | K=4 loops | 489 | 60.1 | 0.018 | 0.137 | 0.007 | 0.03 | — | — | — | — |
| depth_k8_e1 | depth_looped_k8 | e1 | JSONL | `solving/experiments/metrics/depth_k8_e1.jsonl` | solving/experiments/2026-07-21_depth_looped_k8 | 64 | 4 | K=8 | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | K=8 | 491 | 60 | 0.017 | 0.34 | 0.013 | 0.02 | — | — | — | — |
| ncond_d32_k4_e1 | depth_d32_k4_ncond | e1 | JSONL | `solving/experiments/metrics/ncond_d32_k4_e1.jsonl` | solving/experiments/2026-07-21_depth_d32_k4_ncond | 32 | 4 | K=4 | token+abs_pos+N-FiLM | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | N-cond FiLM | 407 | 60.1 | 0.058 | 0.059 | 0.027 | 0.09 | — | — | — | — |
| ncond_d32_k4_e5 | depth_d32_k4_ncond | e5 | JSONL | `solving/experiments/metrics/ncond_d32_k4_e5.jsonl` | solving/experiments/2026-07-21_depth_d32_k4_ncond | 32 | 4 | K=4 | token+abs_pos+N-FiLM | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | N-cond FiLM | 2215 | 60.2 | 0.003 | 0.023 | 0.003 | 0.003 | — | — | — | — |
| scale_d128_e1 | scale_tf_d128 | e1 | JSONL | `solving/experiments/metrics/scale_d128_e1.jsonl` | solving/experiments/2026-07-21_scale_tf_d128 | 128 | 4 | — | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | d=128 | 503 | 60.1 | 0.018 | 0.07 | 0.027 | 0.01 | — | — | — | — |
| scale_d32_e1 | scale_tf_d32 | e1 | JSONL | `solving/experiments/metrics/scale_d32_e1.jsonl` | solving/experiments/2026-07-21_scale_tf_d32 | 32 | 4 | — | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | d=32 | 539 | 60.1 | 0.027 | 0.039 | 0.013 | 0.04 | — | — | — | — |
| scale_d96_e1 | scale_tf_d96 | e1 | JSONL | `solving/experiments/metrics/scale_d96_e1.jsonl` | solving/experiments/2026-07-21_scale_tf_d96 | 96 | 4 | — | token+abs_pos | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | d=96 | 541 | 60.2 | 0.015 | 0.039 | 0.02 | 0.01 | — | — | — | — |
| claude_abacus_e1 | claude_abacus_e1 | e1 | RECOVERED | `solving/experiments/*_claude_abacus_e1/NOTE.md` | solving/experiments/2026-07-22_claude_abacus_e1 | 32 | 4 | L=4 | token+Abacus+Abacus_place | AdamW(lr=3e-3,wd=0.1) | wallclock_LambdaLR | 256 | 53184 | Abacus place-value embedding alone (end-anchored, MSD-corrected), no RoPE, no absolute position, no depth embedding — vs anchor claude_std_rope_e1 | 1381 | 60 | 0.03665 | 0.449 | 0.0133 | 0.06 | — | — | — | metrics not saved |
| claude_evalk4_zeroinit_e1 | claude_evalk4_zeroinit | e1 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_evalk4_zeroinit | 32 | 4 | K=4;train2/eval4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | zeroinit | — | — | 0.0233 | 0.09 | 0.027 | 0.02 | — | — | — | — |
| claude_fire_e1 | claude_fire_e1 | e1 | RECOVERED | `solving/experiments/*_claude_fire_e1/NOTE.md` | solving/experiments/2026-07-22_claude_fire_e1 | 32 | 4 | L=4 | token+FIRE | AdamW(lr=3e-3,wd=0.1) | wallclock_LambdaLR | 256 | 52390 | FIRE relative attention bias alone (bidirectional-adapted, signed distance), no RoPE, no absolute position, no depth embedding — vs anchor claude_std_rope_e1 | 1251 | 60 | 0.01835 | 1 | 0.0067 | 0.03 | — | — | — | metrics not saved |
| claude_fireabacus_e1 | claude_fireabacus_e1 | e1 | RECOVERED | `solving/experiments/*_claude_fireabacus_e1/NOTE.md` | solving/experiments/2026-07-22_claude_fireabacus_e1 | 32 | 4 | L=4 | token+FIRE+Abacus+Abacus_place | AdamW(lr=3e-3,wd=0.1) | wallclock_LambdaLR | 256 | 54438 | Abacus + FIRE combined (both end-anchored/bidirectional-adapted as in the two parent cards), no RoPE, no absolute position, no depth embedding — vs anchor claude_std_rope_e1 | 1225 | 60 | 0.01335 | 1 | 0.0067 | 0.02 | — | — | — | metrics not saved |
| claude_hard_h1 | claude_hard_h1 | h1 | RECOVERED | `solving/RESEARCH_LOG.md` (2026-07-21 Claude session) + `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_hard_h1 | 2048 | 16 | K=4 | token+abs_pos+place+field/segment | AdamW(lr=None,wd=0.1) | wallclock_LambdaLR | 256 | 50500000 | d=2048 K=4 | 190017 | 3600 | 0.0003 | 1 | 0 | — | 0 | — | 0 | LB 0.03%; split exacts 0.0000% |
| claude_pv_ansplace_e1 | claude_pv_ansplace | e1 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_ansplace | 32 | 4 | K=4 | token+abs_pos+place+field/segment+depth+ans_place | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | ansplace | — | — | 0.02 | — | 0.02 | 0.02 | — | — | — | — |
| claude_pv_d128_e1 | claude_pv_d128 | e1 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_d128 | 128 | 8 | K=4 | token+abs_pos+place+field/segment+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | 210000 | d=128 | 394 | — | 0.02 | — | 0.02 | 0.02 | — | — | — | — |
| claude_pv_d128_k8_e1 | claude_pv_d128_k8 | e1 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_d128_k8 | 128 | 8 | K=8 | token+abs_pos+place+field/segment+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | d128 k8 | — | — | — | 0.031 | 0.047 | — | — | — | — | mean_exact not stated; test+train only |
| claude_pv_evalk4_e1 | claude_pv_evalk4 | e1 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_evalk4 | 32 | 4 | K=4;train2/eval4 | token+abs_pos+place+field/segment+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | pv+evalk4 | — | — | 0.0217 | — | 0.033 | 0.01 | — | — | — | — |
| claude_pv_fast_e3 | claude_pv_fast | e3 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_fast | 32 | 4 | K=4 | token+abs_pos+place+field/segment+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | fast | — | — | 0.0075 | — | — | — | — | — | — | — |
| claude_pv_fast_m1 | claude_pv_fast | m1 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_fast | 32 | 4 | K=4 | token+abs_pos+place+field/segment+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | fast | — | — | 0.00083 | — | — | — | — | — | — | majority-class prior vicinity |
| claude_pv_k4_ut_e1 | claude_pv_k4_ut | e1 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_k4_ut | 32 | 4 | K=4 | token+abs_pos+place+field/segment+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | place-value emb | — | — | 0.0583 | — | 0.027 | 0.09 | — | — | — | — |
| claude_pv_k4_ut_e3_r1 | claude_pv_k4_ut | e3 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_k4_ut | 32 | 4 | K=4 | token+abs_pos+place+field/segment+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | place-value emb | 1505 | — | 0.0131 | — | 0.013 | 0.014 | — | — | — | — |
| claude_pv_k4_ut_e3_r2 | claude_pv_k4_ut | e3 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_k4_ut | 32 | 4 | K=4 | token+abs_pos+place+field/segment+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | place-value emb | 2041 | — | 0.0069 | — | 0.005 | 0.009 | — | — | — | — |
| claude_pv_noabspos_e1 | claude_pv_noabspos | e1 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_noabspos | 32 | 4 | K=4 | token+place+field/segment+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | noabspos | — | — | 0.0383 | — | 0.007 | 0.07 | — | — | — | — |
| claude_pv_tadapt_e1 | claude_pv_tadapt | e1 | RECOVERED | `learnings/concepts/17-recurrence-generalisation.md` + NOTE | solving/experiments/2026-07-21_claude_pv_tadapt | 32 | 4 | loops=T | token+abs_pos+place+field/segment | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | T-adapt depth | — | — | 0.02 | — | — | — | — | — | — | ood collapse cited as 2% |
| claude_pv_tcoupled_e1 | claude_pv_tcoupled | e1 | RECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_tcoupled | 32 | 4 | loops=T+1 | token+abs_pos+place+field/segment+depth | AdamW(lr=3e-3,wd=0.1) | CosineAnnealingLR | 256 | — | T-coupled | — | — | 0.02 | — | 0.02 | 0.02 | — | — | — | — |
| claude_std_rope_e1 | claude_std_rope_e1 | e1 | RECOVERED | `solving/experiments/*_claude_std_rope_e1/NOTE.md` | solving/experiments/2026-07-22_claude_std_rope_e1 | 32 | 4 | L=4 | token+RoPE | AdamW(lr=3e-3,wd=0.1) | wallclock_LambdaLR | 256 | — | standard (non-recurrent, 4 independent layers) Transformer, plain token embedding only, RoPE instead of learned absolute position + depth embedding — vs anchor depth_d32_k4_ut (weight-tied K=4 loop, absolute + depth embeddings) | 1353 | 60 | 0.04835 | 1 | 0.0267 | 0.07 | — | — | — | mean=avg(test,ood); metrics not saved |
| claude_pv_fast_tsched_UNSPEC | claude_pv_fast_tsched | — | UNRECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_fast_tsched | 32 | 4 | K=4 | token+abs_pos+place+field/segment+depth | AdamW(lr=3e-3,wd=0.1) | wallclock_LambdaLR | 256 | — | tsched | — | — | — | — | — | — | — | — | — | wall-clock schedule verified; scored mean not recorded in log/NOTE |
| claude_pv_tadapt_tsched_UNSPEC | claude_pv_tadapt_tsched | — | UNRECOVERED | `learnings/concepts/16-representation-vs-throughput.md` | solving/experiments/2026-07-21_claude_pv_tadapt_tsched | 32 | 4 | loops=T | token+abs_pos+place+field/segment | AdamW(lr=3e-3,wd=0.1) | wallclock_LambdaLR | 256 | — | tadapt+tsched | — | — | — | — | — | — | — | — | — | wall-clock schedule verified; scored mean not recorded in log/NOTE |
| claude_scale_d2048_e5 | claude_scale_d2048 | e5 | UNRECOVERED | `solving/RESEARCH_LOG.md` (2026-07-21 Claude session) | solving/experiments/2026-07-21_claude_scale_d2048 | 2048 | 16 | — | token+abs_pos+place+field/segment | AdamW(lr=None,wd=0.1) | wallclock_LambdaLR | 256 | — | d=2048 | 1765 | — | — | — | — | — | — | — | — | steps only; mean/test/ood not in log |
| claude_scale_d4096_e5 | claude_scale_d4096 | e5 | UNRECOVERED | `solving/RESEARCH_LOG.md` (2026-07-21 Claude session) | solving/experiments/2026-07-21_claude_scale_d4096 | 4096 | 32 | — | token+abs_pos+place+field/segment | AdamW(lr=None,wd=0.1) | wallclock_LambdaLR | 256 | — | d=4096 | 1005 | — | — | — | — | — | — | — | — | steps only; mean/test/ood not in log |
| claude_scale_d512_e5 | claude_scale_d512 | e5 | UNRECOVERED | `solving/RESEARCH_LOG.md` (2026-07-21 Claude session) | solving/experiments/2026-07-21_claude_scale_d512 | 512 | 8 | — | token+abs_pos+place+field/segment | AdamW(lr=None,wd=0.1) | wallclock_LambdaLR | 256 | — | d=512 | 1981 | — | — | — | — | — | — | — | — | steps only; mean/test/ood not in log |
| hard_sample_v0_NO_RUN | hard_sample_v0 | — | UNRECOVERED | no metrics JSONL; not in RESEARCH_LOG/NOTE scored lines | solving/experiments/2026-07-22_hard_sample_v0 | 128 | 4 | K=8 | token+abs_pos+place+field/segment+depth | AdamW(lr=1e-3,wd=1.0) | wallclock_LambdaLR | 256 | — | Hard-oriented sample: d=128 K=8 UT + place/field + input inject + clamped cosine + wd=1.0 | — | — | — | — | — | — | — | — | — | card exists on disk; no scored numbers found |
| t1only_probe_rope_NO_RUN | t1only_probe_rope | — | UNRECOVERED | no metrics JSONL; not in RESEARCH_LOG/NOTE scored lines | solving/experiments/2026-07-22_t1only_probe_rope | 32 | 4 | L=4 | token+RoPE | AdamW(lr=3e-3,wd=0.1) | wallclock_LambdaLR | 256 | 51136 | no architecture change vs claude_std_rope_e1 anchor; new manifest — e5 train/test filtered to time_steps==1 only, N varying (1600/400 rows), no ood split | — | — | — | — | — | — | — | — | — | card exists on disk; no scored numbers found |
| t1only_probe_ut_k4_NO_RUN | t1only_probe_ut_k4 | — | UNRECOVERED | no metrics JSONL; not in RESEARCH_LOG/NOTE scored lines | solving/experiments/2026-07-22_t1only_probe_ut_k4 | 32 | 4 | K=4 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | wallclock_LambdaLR | 256 | 13728 | swap 4 untied RoPE layers for weight-tied UT K=4 loop (depth_d32_k4_ut, unmodified), same T=1-only probe manifest as t1only_probe_rope | — | — | — | — | — | — | — | — | — | card exists on disk; no scored numbers found |
| t1only_probe_ut_k8_NO_RUN | t1only_probe_ut_k8 | — | UNRECOVERED | no metrics JSONL; not in RESEARCH_LOG/NOTE scored lines | solving/experiments/2026-07-22_t1only_probe_ut_k8 | 32 | 4 | K=8 | token+abs_pos+depth | AdamW(lr=3e-3,wd=0.1) | wallclock_LambdaLR | 256 | 13856 | swap 4 untied RoPE layers for weight-tied UT K=8 loop (depth_d32_k4_ut with num_loops=8), same T=1-only probe manifest as t1only_probe_rope | — | — | — | — | — | — | — | — | — | card exists on disk; no scored numbers found |

### 5.1 Hard H1 detail (same card as `claude_hard_h1` row above)

From `learnings/concepts/16-representation-vs-throughput.md` / RESEARCH_LOG factual lines:

| field | value |
|-------|-------|
| card | `claude_hard_h1` |
| d_model / params / K | 2048 / 50.5M / K=4 |
| completed_steps | 190017 |
| train exact (end) | 1.0 |
| test / ood_t / ood_n_t exact | 0.0 / 0.0 / 0.0 |
| eval losses | test 15.836; ood_t 16.170; ood_n_t 16.387 |
| leaderboard | 0.03% (#11 at time of log) |

### 5.2 Config join method

Per-run arch/dims/embedding/optimizer are **not** fields in the evaluator JSONL. This packet joins them from each experiment dir. Embedding scheme is a parsed presence list (token / abs_pos / place / field/segment / depth / RoPE / FIRE / Abacus / N-FiLM / ans_place), not a full hyperparameter dump — read `submission.py` for exact formulas.

Params marked `—` were not stated in NOTE/config and could not be counted here (no local torch). Known overrides: `claude_hard_h1`=50500000; `claude_pv_d128`=210000 (from concept 16); Jul-22 Abacus/FIRE NOTES.

### 5.3 Loss / curve artifact paths

Directory: `solving/experiments/figures/`

Directory: `solving/experiments/figures/`

- `solving/experiments/figures/fig_act_train_curves.png`
- `solving/experiments/figures/fig_act_vs_k4_e1_e5.png`
- `solving/experiments/figures/fig_all_eval_loss.png`
- `solving/experiments/figures/fig_all_gen_gap.png`
- `solving/experiments/figures/fig_all_loss_start_end.png`
- `solving/experiments/figures/fig_all_score_ladder.png`
- `solving/experiments/figures/fig_all_throughput.png`
- `solving/experiments/figures/fig_all_train_exact.png`
- `solving/experiments/figures/fig_all_train_loss.png`
- `solving/experiments/figures/fig_all_train_loss_zoom.png`
- `solving/experiments/figures/fig_baseline_ladder_e1.png`
- `solving/experiments/figures/fig_baseline_train_curves_e1.png`
- `solving/experiments/figures/fig_baseline_train_loss_e1.png`
- `solving/experiments/figures/fig_combo_d32_k4_e1.png`
- `solving/experiments/figures/fig_combo_d32_k4_train_e1.png`
- `solving/experiments/figures/fig_d32_k4_e1_vs_e5.png`
- `solving/experiments/figures/fig_d32_k4_e1_vs_e5_curves.png`
- `solving/experiments/figures/fig_d32_k_e1_vs_e5.png`
- `solving/experiments/figures/fig_d32_k_sweep_curves_e1.png`
- `solving/experiments/figures/fig_d32_k_sweep_e1.png`
- `solving/experiments/figures/fig_depth_ablation_e1.png`
- `solving/experiments/figures/fig_depth_train_exact_e1.png`
- `solving/experiments/figures/fig_max_ladder_e1.png`
- `solving/experiments/figures/fig_max_vs_v1_e1.png`
- `solving/experiments/figures/fig_midloop_vs_ut_e1_e5.png`
- `solving/experiments/figures/fig_ncond_train_curves.png`
- `solving/experiments/figures/fig_ncond_vs_base_e1_e5.png`
- `solving/experiments/figures/fig_scaling_params_vs_score.png`
- `solving/experiments/figures/fig_scaling_steps_vs_score.png`
- `solving/experiments/figures/fig_scaling_width_e1.png`
- `solving/experiments/figures/fig_scaling_width_steps_e1.png`
- `solving/experiments/figures/fig_ut_vs_plain_e1_e5.png`

Index: `solving/experiments/figures/PLOTS_INDEX.md`

Raw JSONL curves: each `solving/experiments/metrics/<run>.jsonl` contains `type=training` rows with `loss` / `exact_accuracy` / `step`.

## 6. Submission quotas and evaluation hardware

From README Compute tiers + `service/tiers.py`:

| Tier | Accepted attempts / UTC day | Training seconds | Eval seconds |
|------|----------------------------|------------------|--------------|
| Easy | 60 | 60 | 30 |
| Medium | 6 | 600 | 300 |
| Hard | 1 | 3600 | 1800 |

README: “Failed evaluations count after acceptance; authentication and validation rejections do not.”

Hardware (README / site): H100 training-time budget; site tagline “One H100.” Manifests use `"device": "cuda:0"`, `dtype` bfloat16.
Exact GPU SKU string beyond “H100”: **UNDOCUMENTED** in README beyond H100 naming.

## 7. Compute available to us (workspace ops note)

From `solving/experiments/OPS.md` (2026-07-22):

- Prime Intellect L40S instance used for local zero-quota runs; SSH `ubuntu@204.52.24.142` (ephemeral while instance up); alias `oneL40`.
- Stack note on that box: torch 2.12.1+cu126, NVIDIA L40S.
- User-stated budget in chat: ~$1/hr credits; H100s available if needed.
- Competition hosted scoring: H100 via `one-layer submit` (does not spend Prime credits).

## 8. Leaderboard state (one line + raw dump)

**One line (2026-07-22, `one-layer leaderboard` + site):** Top Hard exact accuracy **0.40%** (az); site lists **26 ranked** participants on Hard; score range on the printed board **0.40% … 0.00%**; Easy/Medium are private practice (site).

Raw CLI output (truncated to top 20 as returned):

```
Hard leaderboard · best successful score per participant
  #     score  participant / file
  1     0.40%  az / submission.py
  2     0.23%  Frosty40 / submission.py
  3     0.07%  rumi / submission.py
  4     0.06%  vatsalnar123 / submission.py
  5     0.05%  byebyescaling / submission.py
  6     0.05%  Edward Ngo / submission.py
  7     0.05%  1rreverent / submission.py
  8     0.05%  tokenbender / submission.py
  9     0.05%  Shikhar Gupta / submission.py
 10     0.04%  0xDEADD06 / submission.py
 11     0.04%  Amandeep Singh / submission.py
 12     0.04%  priormancer / submission.py
 13     0.04%  armoredmeatball / submission.py
 14     0.04%  trolldemort9 / submission.py
 15     0.03%  Kashif / submission.py
 16     0.03%  Vikas Mishra / submission.py
 17     0.03%  Dhruv Rawat / submission.py
 18     0.03%  Aakanksh Zarapkar / submission.py
 19     0.03%  Nikhil Barhate / submission.py
 20     0.03%  mof / submission.py
```

Site fetch 2026-07-22 also listed ranks 21–26 at 0.02% or 0.00% (Asher Labovich … Umesh Yadav).

## Appendix A — Full `benchmark/runner.py`

```python
"""Evaluator-owned runner for the One Layer Deeper competition."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data import (
    infer_max_seq_len,
    infer_vocab_size,
    make_dataloaders,
)
from .api import ModelSpec, OptimizerBundle, OptimizerSpec, Submission
from .batches import prepare_batch
from .manifest import BenchmarkManifest, load_manifest
from .metrics import MetricRecorder
from .validation import (
    assert_state_versions_unchanged,
    capture_state_versions,
    lint_submission_source,
    validate_model_state,
    validate_optimizer,
    validate_submission,
)


EVALUATION_TIME_FRACTION = 0.5
SCORING_SPLIT_PRIORITY = ("test", "ood", "ood_t", "ood_n_t")
NON_SCORING_SPLITS = frozenset(("train", "eval"))


def _scoring_split_names(dataloaders) -> tuple[str, ...]:
    """Return deterministic scored splits for final measurement."""

    available = set(dataloaders) - NON_SCORING_SPLITS
    prioritized = [name for name in SCORING_SPLIT_PRIORITY if name in available]
    remaining = sorted(available - set(prioritized))
    return tuple((*prioritized, *remaining))


def _deny_dataset_file_access(data_root: str | Path) -> None:
    """Prevent uploaded code from reopening evaluator-owned dataset files."""

    protected_root = Path(data_root).resolve()

    def audit(event: str, args: tuple) -> None:
        if event != "open" or not args:
            return
        candidate = args[0]
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(candidate)).resolve()
        if path == protected_root or protected_root in path.parents:
            raise PermissionError("submission may not access evaluator-owned dataset files")

    sys.addaudithook(audit)


def _configure_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _with_batch_size(
    dataloaders,
    manifest: BenchmarkManifest,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
):
    """Rebatch already-loaded datasets without reopening evaluator data files."""

    resized = {}
    for split_name, original in dataloaders.items():
        is_train = split_name == "train"
        generator = (
            torch.Generator(device="cpu").manual_seed(seed) if is_train else None
        )
        loader = DataLoader(
            original.dataset,
            batch_size=batch_size if is_train else eval_batch_size,
            shuffle=(
                manifest.data.shuffle_train
                if is_train
                else manifest.data.shuffle_eval
            ),
            collate_fn=original.collate_fn,
            num_workers=manifest.data.num_workers,
            pin_memory=original.pin_memory,
            drop_last=manifest.data.drop_last if is_train else False,
            generator=generator,
        )
        if is_train and len(loader) == 0:
            raise ValueError(
                f"submission batch_size={batch_size} produces no complete training batches"
            )
        resized[split_name] = loader
    return resized


def _resolve_batch_sizes(
    submission: Submission,
    manifest: BenchmarkManifest,
) -> tuple[int, int]:
    batch_size = submission.batch_size or manifest.data.batch_size
    eval_batch_size = (
        submission.eval_batch_size
        or submission.batch_size
        or manifest.data.eval_batch_size
        or manifest.data.batch_size
    )
    return batch_size, eval_batch_size


def _resolve_device(manifest: BenchmarkManifest) -> torch.device:
    device = torch.device(manifest.runtime.device)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("manifest requires CUDA, but CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "official execution requires exactly one visible CUDA device; "
            f"found {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(device)
    return device


def _make_model_spec(manifest: BenchmarkManifest) -> ModelSpec:
    return ModelSpec(
        vocab_size=infer_vocab_size(manifest.data),
        max_seq_len=infer_max_seq_len(manifest.data),
        maximum_model_state_elements=manifest.model_state.maximum_elements,
    )


def _validate_model_interface(model: nn.Module, spec: ModelSpec) -> None:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("model must expose a config object")
    expected = {
        "vocab_size": spec.vocab_size,
        "max_seq_len": spec.max_seq_len,
    }
    for field, value in expected.items():
        if getattr(config, field, None) != value:
            raise ValueError(f"model config {field} must equal {value}")


def _autocast(manifest: BenchmarkManifest, device: torch.device):
    if not manifest.runtime.amp:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=getattr(torch, manifest.runtime.dtype),
    )


def _compile_model(model: nn.Module, manifest: BenchmarkManifest) -> nn.Module:
    return torch.compile(model, dynamic=True) if manifest.runtime.compile else model


def _next_batch(iterator, dataloader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def _loss_and_accuracy(
    model: nn.Module,
    batch,
    manifest: BenchmarkManifest,
    device: torch.device,
    *,
    training_loss=None,
) -> tuple[torch.Tensor, float, int, int]:
    input_ids, targets, attention_mask, target_positions = prepare_batch(
        batch,
        device,
    )

    with _autocast(manifest, device):
        logits, auxiliary = model(
            input_ids,
            attention_mask=attention_mask,
        )
        if (
            logits.ndim != 3
            or logits.shape[:2] != input_ids.shape
            or logits.shape[-1] != model.config.vocab_size
        ):
            raise ValueError(
                "language-model logits must have shape "
                "(batch, sequence, vocab_size)"
            )
        if target_positions is None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    "causal language-model targets must match the input shape"
                )
            token_logits = logits[:, :-1, :].float()
            token_targets = targets[:, 1:]
        else:
            if target_positions.shape != targets.shape:
                raise ValueError(
                    "target_positions must have the same shape as targets"
                )
            valid_positions = target_positions[targets != -100]
            if (
                (valid_positions < 0).any().item()
                or (valid_positions >= input_ids.shape[1]).any().item()
            ):
                raise ValueError("target position is outside the input sequence")
            batch_indices = torch.arange(logits.shape[0], device=device)[:, None]
            token_logits = logits[
                batch_indices,
                target_positions.clamp_min(0),
            ].float()
            token_targets = targets

        valid = token_targets != -100
        if not valid.any().item():
            raise ValueError("batch contains no valid language-model targets")
        loss_logits = token_logits[valid]
        loss_labels = token_targets[valid]
        if training_loss is None:
            loss = F.cross_entropy(loss_logits, loss_labels)
        else:
            loss = training_loss(loss_logits, loss_labels, auxiliary)

        token_predictions = token_logits.argmax(dim=-1)
        rows_with_targets = valid.any(dim=1)
        exact_rows = (
            (token_predictions == token_targets) | ~valid
        ).all(dim=1)[rows_with_targets]
        example_count = int(rows_with_targets.sum().item())
        loss_weight = int(valid.sum().item())

        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise TypeError("training_loss must return one scalar tensor")
        if loss.device != device:
            raise ValueError(f"training_loss must return a tensor on {device}")
        if training_loss is not None and not loss.requires_grad:
            raise ValueError("training_loss result must be differentiable")

    exact_accuracy = exact_rows.float().mean().item()
    return loss, exact_accuracy, example_count, loss_weight


def _train(
    *,
    raw_model: nn.Module,
    train_model: nn.Module,
    training_loss,
    bundle: OptimizerBundle,
    dataloader,
    manifest: BenchmarkManifest,
    device: torch.device,
    started_at: float,
    deadline: float,
    budget_seconds: float,
    max_steps: int,
    seed: int,
    metric_recorder: MetricRecorder | None = None,
) -> tuple[float | None, int, float, int]:
    optimizer = bundle.optimizer
    raw_model.train()
    validate_optimizer(bundle, raw_model, device)
    iterator = iter(dataloader)
    final_loss = None
    final_accuracy = None
    completed_steps = 0
    last_metric_step = 0
    optimizer_state_elements = 0

    for step in range(1, max_steps + 1):
        if time.monotonic() >= deadline:
            break
        validate_model_state(raw_model, manifest.model_state, device)
        batch, iterator = _next_batch(iterator, dataloader)
        optimizer.zero_grad(set_to_none=True)
        loss, accuracy, _, _ = _loss_and_accuracy(
            train_model,
            batch,
            manifest,
            device,
            training_loss=training_loss,
        )
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError(f"non-finite training loss at step {step}")
        loss.backward()
        if manifest.runtime.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), manifest.runtime.grad_clip
            )
        optimizer.step()
        if bundle.scheduler is not None:
            bundle.scheduler.step()

        final_loss = float(loss.item())
        final_accuracy = accuracy
        completed_steps = step
        if step == 1:
            optimizer_state_elements = validate_optimizer(bundle, raw_model, device)
        if step == 1 or step % manifest.runtime.log_every == 0:
            elapsed = time.monotonic() - started_at
            print(
                f"step={step} loss={final_loss:.6f} accuracy={accuracy:.6f} "
                f"elapsed={elapsed:.1f}s budget={budget_seconds:.1f}s",
                flush=True,
            )
            if metric_recorder is not None:
                metric_recorder.record_training(
                    seed=seed,
                    step=step,
                    elapsed_seconds=elapsed,
                    loss=final_loss,
                    exact_accuracy=accuracy,
                )
                last_metric_step = step

    elapsed = time.monotonic() - started_at
    validate_model_state(raw_model, manifest.model_state, device)
    if (
        metric_recorder is not None
        and completed_steps > 0
        and completed_steps != last_metric_step
    ):
        metric_recorder.record_training(
            seed=seed,
            step=completed_steps,
            elapsed_seconds=elapsed,
            loss=final_loss,
            exact_accuracy=final_accuracy,
        )
    return final_loss, completed_steps, elapsed, optimizer_state_elements


def _evaluate(
    model: nn.Module,
    dataloader,
    manifest: BenchmarkManifest,
    device: torch.device,
    *,
    deadline: float,
    budget_seconds: float,
) -> dict[str, float]:
    model.eval()
    versions = capture_state_versions(model)
    loss_sum = 0.0
    correct_sum = 0.0
    example_count = 0
    loss_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"evaluation exhausted its {budget_seconds:.1f}s time budget"
                )
            loss, accuracy, batch_examples, batch_loss_weight = _loss_and_accuracy(
                model, batch, manifest, device
            )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"evaluation exhausted its {budget_seconds:.1f}s time budget"
                )
            loss_sum += float(loss.item()) * batch_loss_weight
            correct_sum += accuracy * batch_examples
            example_count += batch_examples
            loss_count += batch_loss_weight
    if time.monotonic() >= deadline:
        raise TimeoutError(
            f"evaluation exhausted its {budget_seconds:.1f}s time budget"
        )
    assert_state_versions_unchanged(model, versions)
    model.train()
    if example_count == 0 or loss_count == 0:
        raise ValueError("evaluation split contains no labels")
    accuracy = correct_sum / example_count
    return {"loss": loss_sum / loss_count, "exact_accuracy": accuracy}


def _run_seed(
    submission: Submission,
    manifest: BenchmarkManifest,
    model_spec: ModelSpec,
    device: torch.device,
    seed: int,
    budget_seconds: float,
    submission_load_seconds: float,
    dataloaders=None,
    metric_recorder: MetricRecorder | None = None,
) -> dict:
    _configure_seed(seed, device)
    batch_size, eval_batch_size = _resolve_batch_sizes(submission, manifest)
    if dataloaders is None:
        dataloaders = make_dataloaders(
            replace(
                manifest.data,
                seed=seed,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
            ),
            device=device,
        )
    elif (
        batch_size != manifest.data.batch_size
        or eval_batch_size
        != (manifest.data.eval_batch_size or manifest.data.batch_size)
    ):
        dataloaders = _with_batch_size(
            dataloaders,
            manifest,
            batch_size,
            eval_batch_size,
            seed,
        )
    max_steps = min(
        manifest.runtime.max_steps,
        submission.max_steps or manifest.runtime.max_steps,
    )

    started_at = time.monotonic() - submission_load_seconds
    deadline = started_at + budget_seconds
    if time.monotonic() >= deadline:
        raise TimeoutError("submission import exhausted the training-time budget")
    model = submission.build_model(model_spec)
    if not isinstance(model, nn.Module):
        raise TypeError("build_model must return torch.nn.Module")
    model_dtype = (
        torch.float32
        if manifest.runtime.amp
        else getattr(
            torch,
            manifest.runtime.dtype,
        )
    )
    model = model.to(device=device, dtype=model_dtype)
    _validate_model_interface(model, model_spec)
    state_elements = validate_model_state(model, manifest.model_state, device)

    bundle = submission.build_optimizer(
        model,
        OptimizerSpec(
            training_time_seconds=budget_seconds,
            device_type=device.type,
        ),
    )
    validate_optimizer(bundle, model, device)
    train_model = _compile_model(model, manifest)
    final_loss, steps, training_seconds, optimizer_state_elements = _train(
        raw_model=model,
        train_model=train_model,
        training_loss=submission.training_loss,
        bundle=bundle,
        dataloader=dataloaders["train"],
        manifest=manifest,
        device=device,
        started_at=started_at,
        deadline=deadline,
        budget_seconds=budget_seconds,
        max_steps=max_steps,
        seed=seed,
        metric_recorder=metric_recorder,
    )

    evaluation = {}
    evaluation_budget_seconds = budget_seconds * EVALUATION_TIME_FRACTION
    evaluation_started_at = time.monotonic()
    evaluation_deadline = evaluation_started_at + evaluation_budget_seconds
    for split_name in _scoring_split_names(dataloaders):
        dataloader = dataloaders[split_name]
        metrics = _evaluate(
            model,
            dataloader,
            manifest,
            device,
            deadline=evaluation_deadline,
            budget_seconds=evaluation_budget_seconds,
        )
        evaluation[split_name] = metrics
        print(
            f"seed={seed} split={split_name} loss={metrics['loss']:.6f} "
            f"exact_accuracy={metrics['exact_accuracy']:.6f}",
            flush=True,
        )
        if metric_recorder is not None:
            metric_recorder.record_evaluation(
                seed=seed,
                split=split_name,
                loss=metrics["loss"],
                exact_accuracy=metrics["exact_accuracy"],
            )
    evaluation_seconds = time.monotonic() - evaluation_started_at

    return {
        "seed": seed,
        "model_state_elements": state_elements,
        "optimizer_state_elements_after_first_step": optimizer_state_elements,
        "final_train_loss": final_loss,
        "completed_training_steps": steps,
        "training_batch_size": batch_size,
        "evaluation_batch_size": eval_batch_size,
        "max_training_steps": max_steps,
        "training_seconds": training_seconds,
        "evaluation_budget_seconds": evaluation_budget_seconds,
        "evaluation_seconds": evaluation_seconds,
        "evaluation": evaluation,
    }


def _load_submission_file(path: str | Path) -> Submission:
    submission_path = Path(path).resolve()
    if submission_path.suffix != ".py" or not submission_path.is_file():
        raise ValueError("submission must be one existing .py file")
    if submission_path.stat().st_size > 256 * 1024:
        raise ValueError("submission file exceeds the 256 KiB limit")
    lint_submission_source(submission_path)
    module_spec = importlib.util.spec_from_file_location(
        f"uploaded_submission_{submission_path.stat().st_mtime_ns}",
        submission_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load submission from {submission_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    submission = getattr(module, "SUBMISSION", None)
    if not isinstance(submission, Submission):
        raise TypeError("submission.py must export benchmark.Submission as SUBMISSION")
    return submission


def run_submission_file(
    submission_path: str | Path,
    manifest_path: str | Path,
    *,
    include_structured_metrics: bool = False,
) -> dict:
    manifest = load_manifest(manifest_path)
    device = _resolve_device(manifest)
    model_spec = _make_model_spec(manifest)
    preloaded_dataloaders = {}
    if manifest.data.data_root is not None:
        preloaded_dataloaders = {
            seed: make_dataloaders(
                replace(manifest.data, seed=seed),
                device=device,
            )
            for seed in manifest.runtime.seeds
        }
        _deny_dataset_file_access(manifest.data.data_root)
    submission_load_started = time.monotonic()
    submission = _load_submission_file(submission_path)
    validate_submission(submission)
    submission_load_seconds = time.monotonic() - submission_load_started
    budget_per_seed = manifest.runtime.total_training_time_seconds / len(
        manifest.runtime.seeds
    )
    evaluation_budget_per_seed = budget_per_seed * EVALUATION_TIME_FRACTION
    metric_recorder = MetricRecorder() if include_structured_metrics else None
    batch_size, eval_batch_size = _resolve_batch_sizes(submission, manifest)

    print(
        json.dumps(
            {
                "manifest": manifest.name,
                "model_spec": asdict(model_spec),
                "training_batch_size": batch_size,
                "evaluation_batch_size": eval_batch_size,
                "max_training_steps": min(
                    manifest.runtime.max_steps,
                    submission.max_steps or manifest.runtime.max_steps,
                ),
                "total_training_time_seconds": manifest.runtime.total_training_time_seconds,
                "training_time_seconds_per_seed": budget_per_seed,
                "evaluation_time_seconds_per_seed": evaluation_budget_per_seed,
                "seeds": manifest.runtime.seeds,
            },
            indent=2,
        ),
        flush=True,
    )

    seed_results = [
        _run_seed(
            submission,
            manifest,
            model_spec,
            device,
            seed,
            budget_per_seed,
            submission_load_seconds / len(manifest.runtime.seeds),
            preloaded_dataloaders.get(seed),
            metric_recorder,
        )
        for seed in manifest.runtime.seeds
    ]
    measurements = [
        metrics
        for seed_result in seed_results
        for metrics in seed_result["evaluation"].values()
    ]
    result = {
        "manifest": manifest.name,
        "score": {
            "primary_metric": "mean_exact_accuracy",
            "mean_exact_accuracy": statistics.fmean(
                metrics["exact_accuracy"] for metrics in measurements
            ),
            "mean_loss": statistics.fmean(metrics["loss"] for metrics in measurements),
            "num_measurements": len(measurements),
        },
        "seeds": seed_results,
    }
    if metric_recorder is not None:
        metric_recorder.record_summary(
            completed_steps=sum(
                seed_result["completed_training_steps"]
                for seed_result in seed_results
            ),
            training_seconds=sum(
                seed_result["training_seconds"] for seed_result in seed_results
            ),
            mean_exact_accuracy=result["score"]["mean_exact_accuracy"],
        )
        result["structured_metrics"] = metric_recorder.snapshot()
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--submission-file", required=True)
    parser.add_argument("--include-structured-metrics", action="store_true")
    args = parser.parse_args()
    run_submission_file(
        args.submission_file,
        args.manifest,
        include_structured_metrics=args.include_structured_metrics,
    )


if __name__ == "__main__":
    cli()

```

## Appendix B — Full `benchmark/manifest.py`

```python
"""Validated, evaluator-owned benchmark manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from data import DataConfig


@dataclass(frozen=True)
class RuntimeSpec:
    device: str
    dtype: str
    amp: bool
    compile: bool
    total_training_time_seconds: float
    max_steps: int
    seeds: tuple[int, ...]
    grad_clip: float | None
    log_every: int


@dataclass(frozen=True)
class ModelStateSpec:
    maximum_elements: int


@dataclass(frozen=True)
class BenchmarkManifest:
    name: str
    data: DataConfig
    runtime: RuntimeSpec
    model_state: ModelStateSpec


def _require_keys(value: dict[str, Any], expected: set[str], *, where: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        pieces = []
        if missing:
            pieces.append(f"missing={sorted(missing)}")
        if extra:
            pieces.append(f"unknown={sorted(extra)}")
        raise ValueError(f"invalid {where}: {', '.join(pieces)}")


def load_manifest(path: str | Path) -> BenchmarkManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    _require_keys(
        payload,
        {
            "name",
            "data",
            "runtime",
            "model_state",
        },
        where="manifest",
    )

    runtime_payload = payload["runtime"]
    _require_keys(
        runtime_payload,
        {
            "device",
            "dtype",
            "amp",
            "compile",
            "total_training_time_seconds",
            "max_steps",
            "seeds",
            "grad_clip",
            "log_every",
        },
        where="runtime",
    )
    runtime = RuntimeSpec(
        device=str(runtime_payload["device"]),
        dtype=str(runtime_payload["dtype"]),
        amp=bool(runtime_payload["amp"]),
        compile=bool(runtime_payload["compile"]),
        total_training_time_seconds=float(
            runtime_payload["total_training_time_seconds"]
        ),
        max_steps=int(runtime_payload["max_steps"]),
        seeds=tuple(int(seed) for seed in runtime_payload["seeds"]),
        grad_clip=(
            None
            if runtime_payload["grad_clip"] is None
            else float(runtime_payload["grad_clip"])
        ),
        log_every=int(runtime_payload["log_every"]),
    )
    if runtime.dtype not in {"float32", "bfloat16"}:
        raise ValueError("runtime.dtype must be float32 or bfloat16")
    if (
        runtime.total_training_time_seconds <= 0
        or runtime.max_steps < 1
        or runtime.log_every < 1
    ):
        raise ValueError(
            "total_training_time_seconds, max_steps, and log_every must be positive"
        )
    if not runtime.seeds or len(set(runtime.seeds)) != len(runtime.seeds):
        raise ValueError("runtime.seeds must be a non-empty list of unique integers")
    if runtime.grad_clip is not None and runtime.grad_clip <= 0:
        raise ValueError("runtime.grad_clip must be positive when provided")

    state_payload = payload["model_state"]
    _require_keys(
        state_payload,
        {"maximum_elements"},
        where="model_state",
    )
    model_state = ModelStateSpec(
        maximum_elements=int(state_payload["maximum_elements"])
    )
    if model_state.maximum_elements < 1:
        raise ValueError("model state maximum must be positive")

    data = DataConfig(**payload["data"])

    return BenchmarkManifest(
        name=str(payload["name"]),
        data=data,
        runtime=runtime,
        model_state=model_state,
    )

```

## Appendix C — Full `data/squaring_mod.py`

```python
"""Tokenized repeated modular-squaring dataset generation and loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
import random
from pathlib import Path
from typing import Any

import torch

from .counting import (
    Record,
    TokenizedCountingDataset,
    collate_tokenized_counting,
    compute_split_counts,
    digit_token as counting_digit_token,
    load_counting_dataset_config,
    number_tokens as counting_number_tokens,
    write_dataset_config,
    write_split_files,
)


TOKEN_IDS: dict[str, int] = {
    "PAD": 0,
    "BOS": 1,
    "N": 2,
    "X": 3,
    "T": 4,
    "ANS": 5,
    "EOS": 6,
}
DIGIT_OFFSET = 7
NUM_DIGITS = 10
VOCAB_SIZE = DIGIT_OFFSET + NUM_DIGITS

# A deliberately small, deterministic suite used when DataConfig.data_root is
# unset. It is intended for end-to-end evaluator testing, not as the eventual
# scored squaring-mod benchmark.
SMOKE_FIXED_P = 11
SMOKE_FIXED_Q = 13
SMOKE_TIME_STEPS = (1, 2, 3)
SMOKE_OOD_TIME_STEPS = (4,)
SMOKE_EXAMPLES_PER_SETTING = 100
# N=143, x<=142, and one-digit T produce at most ten prompt tokens in the
# separate-input/output representation.
SMOKE_MAX_SEQ_LEN = 10
ID_SPLITS: tuple[str, ...] = ("train", "test")


class SquaringModTokenizedDataset(TokenizedCountingDataset):
    """JSONL-backed repeated modular-squaring dataset."""


def load_squaring_mod_dataset_config(root: str | Path) -> dict[str, Any]:
    return load_counting_dataset_config(root)


def collate_squaring_mod(batch: list[dict[str, Any]]) -> dict[str, Any]:
    uses_separate_output = [
        len(item["labels"]) != len(item["input_ids"])
        for item in batch
    ]
    if not any(uses_separate_output):
        return collate_tokenized_counting(batch, TOKEN_IDS["PAD"])
    if not all(uses_separate_output):
        raise ValueError("squaring_mod batch cannot mix causal_lm and separate_input_output rows")

    max_input_len = max(len(item["input_ids"]) for item in batch)
    max_target_len = max(len(item["labels"]) for item in batch)
    input_ids = torch.full(
        (len(batch), max_input_len), TOKEN_IDS["PAD"], dtype=torch.long
    )
    labels = torch.full((len(batch), max_target_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_input_len), dtype=torch.bool)
    target_positions = torch.full(
        (len(batch), max_target_len), -1, dtype=torch.long
    )

    for row, item in enumerate(batch):
        item_input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        item_labels = torch.tensor(item["labels"], dtype=torch.long)
        input_len = item_input_ids.numel()
        target_len = item_labels.numel()
        if target_len > input_len:
            raise ValueError("squaring_mod output cannot be longer than its input")
        input_ids[row, :input_len] = item_input_ids
        labels[row, :target_len] = item_labels
        attention_mask[row, :input_len] = True
        target_positions[row, :target_len] = torch.arange(
            input_len - target_len, input_len, dtype=torch.long
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "target_positions": target_positions,
    }


@dataclass(frozen=True)
class SquaringModGenerationConfig:
    output_dir: str
    modulus_bits: list[int] = field(default_factory=lambda: [32])
    fixed_p: int | None = None
    fixed_q: int | None = None
    time_steps: list[int] = field(default_factory=lambda: [16])
    fixed_time_steps: int | None = None
    examples_per_setting: int = 100
    seed: int = 45
    train_fraction: float = 0.8
    test_fraction: float = 0.2
    ood_time_steps: list[int] = field(default_factory=list)
    ood_examples_per_setting: int | None = None
    generator_family: str = "rsa_repeated_squaring"
    separate_input_output: bool = False
    split_group: str = "prompt"
    factor_modulus: int | None = None
    factor_remainder: int | None = None
    separate_ood_splits: bool = False

    def __post_init__(self) -> None:
        fixed_values = (self.fixed_p, self.fixed_q)
        if self.split_group not in ("prompt", "x", "modulus"):
            raise ValueError("split_group must be one of: prompt, x, modulus")
        if self.split_group == "x" and self.fixed_p is None:
            raise ValueError("split_group=x requires fixed_p and fixed_q")
        if self.split_group == "modulus" and self.fixed_p is not None:
            raise ValueError("split_group=modulus requires sampled moduli")
        factor_values = (self.factor_modulus, self.factor_remainder)
        if any(value is None for value in factor_values) and any(
            value is not None for value in factor_values
        ):
            raise ValueError("factor_modulus and factor_remainder must be provided together")
        if self.factor_modulus is not None:
            if self.fixed_p is not None:
                raise ValueError("factor congruence constraints require sampled moduli")
            if self.factor_modulus < 2:
                raise ValueError("factor_modulus must be at least 2")
            if not 0 <= self.factor_remainder < self.factor_modulus:
                raise ValueError("factor_remainder must be in [0, factor_modulus)")
        if self.separate_ood_splits and self.split_group == "prompt":
            raise ValueError("separate_ood_splits requires split_group=x or split_group=modulus")
        if any(value is None for value in fixed_values) and any(value is not None for value in fixed_values):
            raise ValueError("fixed_p and fixed_q must be provided together")
        if self.fixed_p is None:
            if not self.modulus_bits:
                raise ValueError("modulus_bits must contain at least one value when fixed_p/fixed_q are not set")
            if any(value < 4 for value in self.modulus_bits):
                raise ValueError("all modulus_bits values must be at least 4")
        else:
            if self.fixed_p == self.fixed_q:
                raise ValueError("fixed_p and fixed_q must be distinct")
            if not is_probable_prime(self.fixed_p) or not is_probable_prime(self.fixed_q):
                raise ValueError("fixed_p and fixed_q must be prime")
        if self.fixed_time_steps is None:
            if not self.time_steps:
                raise ValueError("time_steps must contain at least one value when fixed_time_steps is not set")
            if any(value < 0 for value in self.time_steps):
                raise ValueError("all time_steps values must be non-negative")
        elif self.fixed_time_steps < 0:
            raise ValueError("fixed_time_steps must be non-negative")
        if any(value < 0 for value in self.ood_time_steps):
            raise ValueError("all ood_time_steps values must be non-negative")
        if self.examples_per_setting < 1:
            raise ValueError("examples_per_setting must be positive")
        if self.ood_examples_per_setting is not None and self.ood_examples_per_setting < 1:
            raise ValueError("ood_examples_per_setting must be positive when provided")
        split_total = self.train_fraction + self.test_fraction
        if not math.isclose(split_total, 1.0):
            raise ValueError("train_fraction + test_fraction must equal 1")
        if min(self.train_fraction, self.test_fraction) < 0:
            raise ValueError("split fractions must be non-negative")
        in_distribution_time_steps = (
            [self.fixed_time_steps]
            if self.fixed_time_steps is not None
            else self.time_steps
        )
        if len(set(in_distribution_time_steps)) != len(in_distribution_time_steps):
            raise ValueError("time_steps must not contain duplicates")
        if len(set(self.ood_time_steps)) != len(self.ood_time_steps):
            raise ValueError("ood_time_steps must not contain duplicates")
        if set(in_distribution_time_steps) & set(self.ood_time_steps):
            raise ValueError("ood_time_steps must not overlap training time steps")
        if self.fixed_p is None:
            for modulus_bits in self.modulus_bits:
                prompt_capacity = _exact_sampled_modulus_prompt_capacity(
                    modulus_bits,
                    factor_modulus=self.factor_modulus,
                    factor_remainder=self.factor_remainder,
                )
                if prompt_capacity is not None and self.examples_per_setting > prompt_capacity:
                    raise ValueError(
                        "examples_per_setting exceeds the number of unique sampled-modulus prompts "
                        f"for modulus_bits={modulus_bits} (capacity={prompt_capacity})"
                    )
                if (
                    prompt_capacity is not None
                    and self.ood_time_steps
                    and self.effective_ood_examples_per_setting > prompt_capacity
                ):
                    raise ValueError(
                        "ood_examples_per_setting exceeds the number of unique sampled-modulus prompts "
                        f"for modulus_bits={modulus_bits} (capacity={prompt_capacity})"
                    )
        if self.fixed_p is not None and self.fixed_q is not None:
            prompt_capacity = (self.fixed_p - 1) * (self.fixed_q - 1)
            if self.examples_per_setting > prompt_capacity:
                raise ValueError(
                    "examples_per_setting exceeds the number of unique x values "
                    "for the fixed modulus"
                )
            if self.ood_time_steps and self.effective_ood_examples_per_setting > prompt_capacity:
                raise ValueError(
                    "ood_examples_per_setting exceeds the number of unique x values "
                    "for the fixed modulus"
                )

    @property
    def effective_ood_examples_per_setting(self) -> int:
        return self.ood_examples_per_setting or self.examples_per_setting


def generate_squaring_mod_smoke_dataset(
    output_dir: str | Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Generate the built-in tiny evaluator smoke dataset."""

    return generate_squaring_mod_dataset(
        SquaringModGenerationConfig(
            output_dir=str(output_dir),
            fixed_p=SMOKE_FIXED_P,
            fixed_q=SMOKE_FIXED_Q,
            time_steps=list(SMOKE_TIME_STEPS),
            examples_per_setting=SMOKE_EXAMPLES_PER_SETTING,
            seed=seed,
            train_fraction=0.8,
            test_fraction=0.2,
            ood_time_steps=list(SMOKE_OOD_TIME_STEPS),
            ood_examples_per_setting=SMOKE_EXAMPLES_PER_SETTING,
            separate_input_output=True,
        )
    )


def _exact_sampled_modulus_prompt_capacity(
    modulus_bits: int,
    *,
    factor_modulus: int | None = None,
    factor_remainder: int | None = None,
) -> int | None:
    """Return exact unit-prompt capacity when the factor ranges are tractable."""
    factor_pairs = _enumerate_sampled_factor_pairs(
        modulus_bits,
        factor_modulus=factor_modulus,
        factor_remainder=factor_remainder,
    )
    if factor_pairs is None:
        return None
    return sum((p - 1) * (q - 1) for p, q in factor_pairs)


def _enumerate_sampled_factor_pairs(
    modulus_bits: int,
    *,
    factor_modulus: int | None = None,
    factor_remainder: int | None = None,
) -> list[tuple[int, int]] | None:
    p_bits = modulus_bits // 2
    q_bits = modulus_bits - p_bits
    if max(p_bits, q_bits) > 10:
        return None

    def eligible(value: int) -> bool:
        if not is_probable_prime(value):
            return False
        if factor_modulus is None:
            return True
        return value % factor_modulus == factor_remainder

    p_candidates = [
        value
        for value in range(1 << (p_bits - 1), 1 << p_bits)
        if eligible(value)
    ]
    q_candidates = [
        value
        for value in range(1 << (q_bits - 1), 1 << q_bits)
        if eligible(value)
    ]
    factor_pairs = {
        p * q: (p, q)
        for p in p_candidates
        for q in q_candidates
        if p != q and (p * q).bit_length() == modulus_bits
    }
    return list(factor_pairs.values())


def digit_token(digit: int) -> int:
    return counting_digit_token(digit, digit_offset=DIGIT_OFFSET)


def number_tokens(value: int) -> list[int]:
    return counting_number_tokens(value, digit_offset=DIGIT_OFFSET)


def trapdoor_squaring_mod(x: int, time_steps: int, p: int, q: int) -> int:
    if time_steps < 0:
        raise ValueError("time_steps must be non-negative")
    modulus = p * q
    phi = (p - 1) * (q - 1)
    exponent = pow(2, time_steps, phi)
    return pow(x, exponent, modulus)


def tokenize_squaring_mod_with_result(
    modulus: int,
    x: int,
    time_steps: int,
    result: int,
    *,
    separate_input_output: bool = False,
) -> tuple[list[int], list[int]]:
    input_ids = [TOKEN_IDS["N"]]
    input_ids.extend(number_tokens(modulus))
    input_ids.append(TOKEN_IDS["X"])
    input_ids.extend(number_tokens(x))
    input_ids.append(TOKEN_IDS["T"])
    input_ids.extend(number_tokens(time_steps))

    result_tokens = number_tokens(result)
    if separate_input_output:
        return input_ids, result_tokens

    input_ids.insert(0, TOKEN_IDS["BOS"])
    input_ids.append(TOKEN_IDS["ANS"])
    input_ids.extend(result_tokens)
    input_ids.append(TOKEN_IDS["EOS"])

    labels = [-100] * len(input_ids)
    answer_start = len(input_ids) - len(result_tokens) - 1
    for offset, token in enumerate(result_tokens):
        labels[answer_start + offset] = token
    labels[-1] = TOKEN_IDS["EOS"]
    return input_ids, labels


def generate_squaring_mod_dataset(config: SquaringModGenerationConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    if config.split_group == "x":
        records = _generate_x_grouped_records(config=config, rng=rng)
    elif config.split_group == "modulus":
        records = _generate_modulus_grouped_records(config=config, rng=rng)
    else:
        records = _generate_prompt_grouped_records(config=config, rng=rng)

    write_split_files(output_dir, records)
    dataset_config = _dataset_config(config, records)
    write_dataset_config(output_dir, dataset_config)
    return dataset_config


def _id_split_counts(config: SquaringModGenerationConfig) -> dict[str, int]:
    return compute_split_counts(
        config.examples_per_setting,
        {
            "train": config.train_fraction,
            "test": config.test_fraction,
        },
    )


def _generate_prompt_grouped_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
) -> list[Record]:
    records: list[Record] = []
    seen_prompts: set[tuple[int, int, int]] = set()
    for modulus_bits in _modulus_settings(config):
        for time_steps in _time_settings(config):
            records.extend(
                _generate_setting_records(
                    config=config,
                    rng=rng,
                    modulus_bits=modulus_bits,
                    time_steps=time_steps,
                    start_index=len(records),
                    seen_prompts=seen_prompts,
                )
            )
    for modulus_bits in _modulus_settings(config):
        for time_steps in config.ood_time_steps:
            records.extend(
                _generate_ood_records(
                    config=config,
                    rng=rng,
                    modulus_bits=modulus_bits,
                    time_steps=time_steps,
                    start_index=len(records),
                    seen_prompts=seen_prompts,
                )
            )
    return records


def _generate_x_grouped_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
) -> list[Record]:
    if config.fixed_p is None or config.fixed_q is None:
        raise ValueError("split_group=x requires fixed factors")
    p, q = config.fixed_p, config.fixed_q
    modulus = p * q
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    rng.shuffle(units)
    if config.examples_per_setting > len(units):
        raise ValueError("fixed modulus lacks enough unique x groups")

    split_counts = _id_split_counts(config)
    id_x = units[: config.examples_per_setting]
    x_by_split: dict[str, list[int]] = {}
    start = 0
    for split in ID_SPLITS:
        stop = start + split_counts[split]
        x_by_split[split] = id_x[start:stop]
        start = stop

    records: list[Record] = []
    for time_steps in _time_settings(config):
        for split in ID_SPLITS:
            for x in x_by_split[split]:
                records.append(
                    _build_record(
                        config=config,
                        p=p,
                        q=q,
                        x=x,
                        time_steps=time_steps,
                        split=split,
                        index=len(records),
                        modulus_bits=None,
                    )
                )

    ood_count = config.effective_ood_examples_per_setting
    unused_x = units[config.examples_per_setting :]
    for time_steps in config.ood_time_steps:
        if config.separate_ood_splits:
            if ood_count > len(x_by_split["train"]):
                raise ValueError("ood_t_seen_x exceeds the number of training x groups")
            if ood_count > len(unused_x):
                raise ValueError("ood_t_unseen_x exceeds the unused fixed-modulus x capacity")
            ood_groups = (
                ("ood_t_seen_x", rng.sample(x_by_split["train"], ood_count)),
                ("ood_t_unseen_x", rng.sample(unused_x, ood_count)),
            )
        else:
            ood_groups = (("ood", rng.sample(units, ood_count)),)
        for split, x_values in ood_groups:
            for x in x_values:
                records.append(
                    _build_record(
                        config=config,
                        p=p,
                        q=q,
                        x=x,
                        time_steps=time_steps,
                        split=split,
                        index=len(records),
                        modulus_bits=None,
                    )
                )
    return records


def _generate_modulus_grouped_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
) -> list[Record]:
    split_counts = _id_split_counts(config)
    factor_pools_by_bits: dict[int, dict[str, list[tuple[int, int]]]] = {}
    for modulus_bits in config.modulus_bits:
        factor_pairs = _enumerate_sampled_factor_pairs(
            modulus_bits,
            factor_modulus=config.factor_modulus,
            factor_remainder=config.factor_remainder,
        )
        if factor_pairs is None:
            raise ValueError(
                "split_group=modulus requires exactly enumerable factor ranges "
                f"for modulus_bits={modulus_bits}"
            )
        factor_pools_by_bits[modulus_bits] = _partition_factor_pairs(
            factor_pairs=factor_pairs,
            split_counts=split_counts,
            config=config,
            rng=rng,
            modulus_bits=modulus_bits,
        )

    records: list[Record] = []
    seen_prompts: set[tuple[int, int, int]] = set()
    for modulus_bits in config.modulus_bits:
        factor_pools = factor_pools_by_bits[modulus_bits]
        for time_steps in _time_settings(config):
            for split in ID_SPLITS:
                records.extend(
                    _generate_records_from_factor_pool(
                        config=config,
                        rng=rng,
                        factor_pool=factor_pools[split],
                        count=split_counts[split],
                        modulus_bits=modulus_bits,
                        time_steps=time_steps,
                        split=split,
                        start_index=len(records),
                        seen_prompts=seen_prompts,
                    )
                )

    ood_count = config.effective_ood_examples_per_setting
    for modulus_bits in config.modulus_bits:
        factor_pools = factor_pools_by_bits[modulus_bits]
        heldout_pool = factor_pools["test"]
        for time_steps in config.ood_time_steps:
            if config.separate_ood_splits:
                ood_pools = (
                    ("ood_t", factor_pools["train"]),
                    ("ood_n_t", heldout_pool),
                )
            else:
                ood_pools = (("ood", sum(factor_pools.values(), [])),)
            for split, factor_pool in ood_pools:
                records.extend(
                    _generate_records_from_factor_pool(
                        config=config,
                        rng=rng,
                        factor_pool=factor_pool,
                        count=ood_count,
                        modulus_bits=modulus_bits,
                        time_steps=time_steps,
                        split=split,
                        start_index=len(records),
                        seen_prompts=seen_prompts,
                    )
                )
    return records


def _partition_factor_pairs(
    *,
    factor_pairs: list[tuple[int, int]],
    split_counts: dict[str, int],
    config: SquaringModGenerationConfig,
    rng: random.Random,
    modulus_bits: int,
) -> dict[str, list[tuple[int, int]]]:
    modulus_counts = compute_split_counts(
        len(factor_pairs),
        {
            "train": config.train_fraction,
            "test": config.test_fraction,
        },
    )
    for _ in range(10_000):
        shuffled = list(factor_pairs)
        rng.shuffle(shuffled)
        pools: dict[str, list[tuple[int, int]]] = {}
        start = 0
        for split in ID_SPLITS:
            stop = start + modulus_counts[split]
            pools[split] = shuffled[start:stop]
            start = stop
        if all(
            _factor_pool_capacity(pools[split]) >= split_counts[split]
            for split in ID_SPLITS
        ):
            return pools
    raise ValueError(
        "could not partition modulus identities with enough prompt capacity "
        f"for modulus_bits={modulus_bits}; increase the bit size or reduce rows"
    )


def _factor_pool_capacity(factor_pool: list[tuple[int, int]]) -> int:
    return sum((p - 1) * (q - 1) for p, q in factor_pool)


def _generate_records_from_factor_pool(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    factor_pool: list[tuple[int, int]],
    count: int,
    modulus_bits: int,
    time_steps: int,
    split: str,
    start_index: int,
    seen_prompts: set[tuple[int, int, int]],
) -> list[Record]:
    if not factor_pool:
        raise ValueError(f"empty factor pool for split={split}, modulus_bits={modulus_bits}")
    if count > _factor_pool_capacity(factor_pool):
        raise ValueError(
            f"split={split}, modulus_bits={modulus_bits} lacks unique prompt capacity"
        )

    weights = [(p - 1) * (q - 1) for p, q in factor_pool]
    records: list[Record] = []
    for offset in range(count):
        for _ in range(10_000):
            p, q = rng.choices(factor_pool, weights=weights, k=1)[0]
            modulus = p * q
            x = _sample_unit(modulus=modulus, rng=rng)
            prompt = (modulus, x, time_steps)
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                break
        else:
            raise ValueError(
                "could not sample a unique grouped-modulus prompt after 10,000 attempts"
            )
        records.append(
            _build_record(
                config=config,
                p=p,
                q=q,
                x=x,
                time_steps=time_steps,
                split=split,
                index=start_index + offset,
                modulus_bits=modulus_bits,
            )
        )
    return records


def _generate_setting_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    modulus_bits: int | None,
    time_steps: int,
    start_index: int,
    seen_prompts: set[tuple[int, int, int]],
) -> list[Record]:
    split_counts = compute_split_counts(
        config.examples_per_setting,
        {
            "train": config.train_fraction,
            "test": config.test_fraction,
        },
    )
    records: list[Record] = []
    for split in ID_SPLITS:
        for _ in range(split_counts[split]):
            records.append(
                _generate_record(
                    config=config,
                    rng=rng,
                    modulus_bits=modulus_bits,
                    time_steps=time_steps,
                    split=split,
                    index=start_index + len(records),
                    seen_prompts=seen_prompts,
                )
            )
    return records


def _generate_ood_records(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    modulus_bits: int | None,
    time_steps: int,
    start_index: int,
    seen_prompts: set[tuple[int, int, int]],
) -> list[Record]:
    return [
        _generate_record(
            config=config,
            rng=rng,
            modulus_bits=modulus_bits,
            time_steps=time_steps,
            split="ood",
            index=start_index + offset,
            seen_prompts=seen_prompts,
        )
        for offset in range(config.effective_ood_examples_per_setting)
    ]


def _generate_record(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    modulus_bits: int | None,
    time_steps: int,
    split: str,
    index: int,
    seen_prompts: set[tuple[int, int, int]],
) -> Record:
    for _ in range(10_000):
        p, q = _sample_or_fixed_factors(
            config=config,
            rng=rng,
            modulus_bits=modulus_bits,
        )
        modulus = p * q
        x = _sample_unit(modulus=modulus, rng=rng)
        prompt = (modulus, x, time_steps)
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            break
    else:
        raise ValueError(
            "could not generate a unique (modulus, x, time_steps) prompt "
            "after 10,000 attempts"
        )
    return _build_record(
        config=config,
        p=p,
        q=q,
        x=x,
        time_steps=time_steps,
        split=split,
        index=index,
        modulus_bits=modulus_bits,
    )


def _build_record(
    *,
    config: SquaringModGenerationConfig,
    p: int,
    q: int,
    x: int,
    time_steps: int,
    split: str,
    index: int,
    modulus_bits: int | None,
) -> Record:
    modulus = p * q
    result = trapdoor_squaring_mod(x, time_steps, p, q)
    input_ids, labels = tokenize_squaring_mod_with_result(
        modulus,
        x,
        time_steps,
        result,
        separate_input_output=config.separate_input_output,
    )
    bit_label = "fixed" if modulus_bits is None else str(modulus_bits)
    return {
        "instance_id": f"squaring_mod_b{bit_label}_t{time_steps}_s{config.seed}_{index:08d}",
        "seed": config.seed,
        "modulus": modulus,
        "modulus_bits": modulus.bit_length(),
        "configured_modulus_bits": modulus_bits,
        "x": x,
        "time_steps": time_steps,
        "result": result,
        "generator_family": config.generator_family,
        "label_exact": True,
        "label_method": "trapdoor_phi",
        "split": split,
        "input_ids": input_ids,
        "labels": labels,
    }


def _sample_or_fixed_factors(
    *,
    config: SquaringModGenerationConfig,
    rng: random.Random,
    modulus_bits: int | None,
) -> tuple[int, int]:
    if config.fixed_p is not None and config.fixed_q is not None:
        return config.fixed_p, config.fixed_q
    if modulus_bits is None:
        raise ValueError("modulus_bits is required when fixed_p/fixed_q are not set")
    return _sample_rsa_factors(
        modulus_bits=modulus_bits,
        rng=rng,
        factor_modulus=config.factor_modulus,
        factor_remainder=config.factor_remainder,
    )


def _sample_rsa_factors(
    *,
    modulus_bits: int,
    rng: random.Random,
    factor_modulus: int | None = None,
    factor_remainder: int | None = None,
) -> tuple[int, int]:
    p_bits = modulus_bits // 2
    q_bits = modulus_bits - p_bits
    for _ in range(10_000):
        p = _sample_prime(bits=p_bits, rng=rng)
        q = _sample_prime(bits=q_bits, rng=rng)
        factors_match = (
            factor_modulus is None
            or (p % factor_modulus == factor_remainder and q % factor_modulus == factor_remainder)
        )
        if p != q and (p * q).bit_length() == modulus_bits and factors_match:
            return p, q
    raise ValueError(f"could not sample a {modulus_bits}-bit RSA modulus")


def _sample_prime(*, bits: int, rng: random.Random) -> int:
    if bits < 2:
        raise ValueError("prime bit length must be at least 2")
    for _ in range(10_000):
        value = rng.getrandbits(bits)
        value |= 1
        value |= 1 << (bits - 1)
        if is_probable_prime(value):
            return value
    raise ValueError(f"could not sample a {bits}-bit prime")


def _sample_unit(*, modulus: int, rng: random.Random) -> int:
    while True:
        x = rng.randrange(1, modulus)
        if math.gcd(x, modulus) == 1:
            return x


def is_probable_prime(value: int) -> bool:
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if value in small_primes:
        return True
    if any(value % prime == 0 for prime in small_primes):
        return False

    d = value - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for base in _miller_rabin_bases(value):
        if base % value == 0:
            continue
        witness = pow(base, d, value)
        if witness in (1, value - 1):
            continue
        for _ in range(s - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _miller_rabin_bases(value: int) -> tuple[int, ...]:
    if value < 2_152_302_898_747:
        return (2, 3, 5, 7, 11)
    if value < 3_474_749_660_383:
        return (2, 3, 5, 7, 11, 13)
    if value < 341_550_071_728_321:
        return (2, 3, 5, 7, 11, 13, 17)
    return (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _modulus_settings(config: SquaringModGenerationConfig) -> list[int | None]:
    if config.fixed_p is not None and config.fixed_q is not None:
        return [None]
    return config.modulus_bits


def _time_settings(config: SquaringModGenerationConfig) -> list[int]:
    if config.fixed_time_steps is not None:
        return [config.fixed_time_steps]
    return config.time_steps


def _dataset_config(config: SquaringModGenerationConfig, records: list[Record]) -> dict[str, Any]:
    split_names = sorted({str(record["split"]) for record in records})
    split_counts = {
        split: sum(1 for record in records if record["split"] == split)
        for split in split_names
    }
    return {
        "dataset_kind": "squaring_mod",
        "generator_config": _public_generator_config(config),
        "split_group": config.split_group,
        "factor_congruence": (
            None
            if config.factor_modulus is None
            else {
                "modulus": config.factor_modulus,
                "remainder": config.factor_remainder,
            }
        ),
        "token_ids": TOKEN_IDS | {"DIGIT_OFFSET": DIGIT_OFFSET},
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": max(len(record["input_ids"]) for record in records),
        "max_modulus_bits": max(record["modulus_bits"] for record in records),
        "max_time_steps": max(record["time_steps"] for record in records),
        "num_examples": len(records),
        "split_counts": split_counts,
        "data_format": (
            "separate_input_output" if config.separate_input_output else "causal_lm"
        ),
        "label_format": (
            "tail_aligned_decimal_residue"
            if config.separate_input_output
            else "next_token_decimal_residue"
        ),
        "label_method": "trapdoor_phi",
    }


def _public_generator_config(config: SquaringModGenerationConfig) -> dict[str, Any]:
    config_dict = asdict(config)
    if config.fixed_p is not None and config.fixed_q is not None:
        config_dict["fixed_modulus"] = config.fixed_p * config.fixed_q
    config_dict["fixed_p"] = None if config.fixed_p is None else "<redacted>"
    config_dict["fixed_q"] = None if config.fixed_q is None else "<redacted>"
    return config_dict


def cli() -> None:
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_class_arguments(SquaringModGenerationConfig)
    parsed = parser.instantiate(parser.parse_args())
    if isinstance(parsed, SquaringModGenerationConfig):
        config = parsed
    else:
        config = SquaringModGenerationConfig(
            **{field.name: getattr(parsed, field.name) for field in fields(SquaringModGenerationConfig)}
        )
    dataset_config = generate_squaring_mod_dataset(config)
    print(f"wrote {dataset_config['num_examples']} examples to {config.output_dir}")


if __name__ == "__main__":
    cli()

```

## Appendix D — Upstream baseline `submissions/baseline_adamw/submission.py`

```python
"""Basic single-pass Transformer with PyTorch AdamW."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)


D_MODEL = 128
NUM_HEADS = 4


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=x.device, dtype=torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.block(x, attention_mask)
        return self.head(self.final_norm(x)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer)

```
