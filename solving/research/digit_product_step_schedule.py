"""Local-only fixed-step scheduler wrapper for the digit-product baseline."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch
from torch import nn

from benchmark import OptimizerBundle, OptimizerSpec, Submission


BASELINE_PATH = Path(__file__).with_name("gate1_digit_product_transformer.py")
spec = importlib.util.spec_from_file_location("digit_product_baseline", BASELINE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASELINE_PATH}")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

WARMUP_STEPS = 400
TOTAL_STEPS = 1_000


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    del spec
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    def factor(step: int) -> float:
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = min((step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS), 1.0)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return OptimizerBundle(
        optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, factor),
    )


SUBMISSION = Submission(
    build_model=baseline.build_model,
    build_optimizer=build_optimizer,
    batch_size=256,
    eval_batch_size=512,
)
