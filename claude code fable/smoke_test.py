"""Replicates the evaluator contract from PRIMARY_SOURCES.md against submission.py.

Checks: (1) source policy (verbatim validate_submission_source), (2) size,
(3) collate/target_positions semantics (verbatim generator+collate), (4) the
_loss_and_accuracy shape/loss contract, (5) train steps w/ grad clip + scheduler,
(6) eval purity via capture_state_versions, (7) bf16 autocast forward,
(8) register mask covers scored positions, (9) eval determinism,
(10) short CPU learnability probe on N=323 T in {1,2,3}.
"""
from __future__ import annotations

import ast
import math
import random
import sys
import time
import types

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- fake benchmark API (verbatim semantics from packet Appendix / api.py)
benchmark = types.ModuleType("benchmark")

from dataclasses import dataclass
from typing import Protocol, Callable
from torch import nn


class Scheduler(Protocol):
    def step(self) -> None: ...


@dataclass(frozen=True)
class ModelSpec:
    vocab_size: int
    max_seq_len: int
    maximum_model_state_elements: int


def model_state_tensors(model: nn.Module):
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
    return sum(value.numel() for _, value in model_state_tensors(model))


def assert_model_state(model: nn.Module, spec: ModelSpec) -> int:
    elements = count_model_state_elements(model)
    if elements > spec.maximum_model_state_elements:
        raise AssertionError("state over budget")
    return elements


@dataclass(frozen=True)
class OptimizerSpec:
    training_time_seconds: float
    device_type: str


@dataclass(frozen=True)
class OptimizerBundle:
    optimizer: torch.optim.Optimizer
    scheduler: Scheduler | None = None


@dataclass(frozen=True)
class Submission:
    build_model: Callable
    build_optimizer: Callable
    training_loss: Callable | None = None
    batch_size: int | None = None
    max_steps: int | None = None
    eval_batch_size: int | None = None


for _n, _v in dict(
    ModelSpec=ModelSpec, OptimizerBundle=OptimizerBundle,
    OptimizerSpec=OptimizerSpec, Submission=Submission,
    assert_model_state=assert_model_state,
    model_state_tensors=model_state_tensors,
    count_model_state_elements=count_model_state_elements,
).items():
    setattr(benchmark, _n, _v)
sys.modules["benchmark"] = benchmark

# ---------------------------------------------------------------- source policy (verbatim, packet submission_validation.py)
FORBIDDEN_SUBMISSION_IMPORTS = {"data", "model", "optim"}


def validate_submission_source(filename, source, max_bytes, *, required_filename="submission.py"):
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if required_filename is not None and basename != required_filename:
        raise ValueError("bad name")
    if len(source.encode("utf-8")) > max_bytes:
        raise ValueError("too big")
    tree = ast.parse(source, filename=basename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported = [node.module]
        else:
            continue
        for name in imported:
            if name.partition(".")[0] in FORBIDDEN_SUBMISSION_IMPORTS:
                raise ValueError(f"forbidden import {name}")
    return basename


src = open("/home/claude/submission.py").read()
validate_submission_source("submission.py", src, 256 * 1024)
print(f"[1] source policy OK, size={len(src.encode())/1024:.1f} KiB")

# ---------------------------------------------------------------- generator (verbatim math from packet Appendix C)
TOKEN_IDS = {"PAD": 0, "BOS": 1, "N": 2, "X": 3, "T": 4, "ANS": 5, "EOS": 6}
DIGIT_OFFSET = 7


def number_tokens(v):
    return [DIGIT_OFFSET + int(c) for c in str(v)]


def trapdoor(x, t, p, q):
    return pow(x, pow(2, t, (p - 1) * (q - 1)), p * q)


def tokenize(modulus, x, t):
    ids = [TOKEN_IDS["N"], *number_tokens(modulus), TOKEN_IDS["X"],
           *number_tokens(x), TOKEN_IDS["T"], *number_tokens(t)]
    return ids, number_tokens(trapdoor(x, t, 17, 19) if modulus == 323 else 0)


def collate(batch):
    max_in = max(len(b["input_ids"]) for b in batch)
    max_tg = max(len(b["labels"]) for b in batch)
    input_ids = torch.full((len(batch), max_in), TOKEN_IDS["PAD"], dtype=torch.long)
    labels = torch.full((len(batch), max_tg), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_in), dtype=torch.bool)
    target_positions = torch.full((len(batch), max_tg), -1, dtype=torch.long)
    for row, item in enumerate(batch):
        ii = torch.tensor(item["input_ids"]); ll = torch.tensor(item["labels"])
        li, lt = ii.numel(), ll.numel()
        assert lt <= li
        input_ids[row, :li] = ii
        labels[row, :lt] = ll
        attention_mask[row, :li] = True
        target_positions[row, :lt] = torch.arange(li - lt, li)
    return dict(input_ids=input_ids, labels=labels,
                attention_mask=attention_mask, target_positions=target_positions)


rng = random.Random(45)
units = [x for x in range(1, 323) if math.gcd(x, 323) == 1]
rows = []
for t in (1, 2, 3):
    for x in rng.sample(units, 200):
        ii, ll = tokenize(323, x, t)
        rows.append({"input_ids": ii, "labels": ll, "x": x, "t": t})
# depth-extrapolation rows (T=16 and T=0) to exercise the loop path
edge = []
for t in (0, 16):
    for x in rng.sample(units, 8):
        y = trapdoor(x, t, 17, 19)
        ii = [TOKEN_IDS["N"], *number_tokens(323), TOKEN_IDS["X"],
              *number_tokens(x), TOKEN_IDS["T"], *number_tokens(t)]
        edge.append({"input_ids": ii, "labels": number_tokens(y)})
print(f"[2] generated {len(rows)} train rows + {len(edge)} edge rows")

# ---------------------------------------------------------------- harness _loss_and_accuracy (verbatim semantics)
def loss_and_accuracy(model, batch, training_loss=None, autocast=False):
    input_ids = batch["input_ids"]; targets = batch["labels"]
    attention_mask = batch["attention_mask"]; target_positions = batch["target_positions"]
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else torch.no_grad() if False else None
    import contextlib
    cm = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    with cm:
        logits, auxiliary = model(input_ids, attention_mask=attention_mask)
        assert logits.ndim == 3 and logits.shape[:2] == input_ids.shape \
            and logits.shape[-1] == model.config.vocab_size, "logits shape contract"
        valid_positions = target_positions[targets != -100]
        assert not (valid_positions < 0).any() and not (valid_positions >= input_ids.shape[1]).any()
        bi = torch.arange(logits.shape[0])[:, None]
        token_logits = logits[bi, target_positions.clamp_min(0)].float()
        token_targets = targets
        valid = token_targets != -100
        loss_logits = token_logits[valid]; loss_labels = token_targets[valid]
        if training_loss is None:
            loss = F.cross_entropy(loss_logits, loss_labels)
        else:
            loss = training_loss(loss_logits, loss_labels, auxiliary)
            assert loss.requires_grad, "training_loss must be differentiable"
        preds = token_logits.argmax(dim=-1)
        rows_with = valid.any(dim=1)
        exact = ((preds == token_targets) | ~valid).all(dim=1)[rows_with]
        assert torch.is_tensor(loss) and loss.ndim == 0
    return loss, exact.float().mean().item()


import importlib.util
mspec = importlib.util.spec_from_file_location("subm", "/home/claude/submission.py")
subm = importlib.util.module_from_spec(mspec)
mspec.loader.exec_module(subm)
SUB = subm.SUBMISSION

max_len = max(len(r["input_ids"]) for r in rows + edge)
spec = ModelSpec(vocab_size=17, max_seq_len=max_len, maximum_model_state_elements=500_000_000)
model = SUB.build_model(spec)
n_params = count_model_state_elements(model)
print(f"[3] built model: {n_params:,} state elements (cap 5e8)")

# register mask must cover every scored position
batch = collate(rows[:64] + edge)
field, place, tval, reg, _ = model._parse(batch["input_ids"], batch["attention_mask"])
tp = batch["target_positions"]; lb = batch["labels"]
for r in range(tp.shape[0]):
    for j in range(tp.shape[1]):
        if lb[r, j] != -100:
            assert reg[r, tp[r, j]], f"register misses scored slot row {r}"
exp_t = torch.tensor([x["t"] for x in rows[:64]] + [0]*8 + [16]*8)
assert (tval == exp_t).all(), f"T parse mismatch: {tval[:10]} vs {exp_t[:10]}"
print("[4] register covers all scored positions; T parsed exactly (incl. T=0, T=16)")

bundle = SUB.build_optimizer(model, OptimizerSpec(training_time_seconds=60.0, device_type="cpu"))
opt, sched = bundle.optimizer, bundle.scheduler

# eval purity + determinism before training
model.eval()
versions = {id(v): v._version for _, v in model_state_tensors(model)}
with torch.no_grad():
    l1, _ = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    l2, _ = model(batch["input_ids"], attention_mask=batch["attention_mask"])
assert {id(v): v._version for _, v in model_state_tensors(model)} == versions, "eval mutated state"
assert torch.equal(l1, l2), "eval nondeterministic"
print("[5] eval purity + determinism OK")

# bf16 autocast forward + loss finite
model.train()
loss, acc = loss_and_accuracy(model, batch, subm.training_loss, autocast=True)
assert torch.isfinite(loss), "non-finite loss under bf16 autocast"
print(f"[6] bf16 autocast train pass OK: loss={loss.item():.4f}")

# training loop replica: grad clip 1.0, scheduler step, N steps on T=1-3 data
t0 = time.time()
random.seed(0); torch.manual_seed(0)
steps = 220
bs = 96
for step in range(1, steps + 1):
    sample = random.sample(rows, bs)
    b = collate(sample)
    opt.zero_grad(set_to_none=True)
    loss, acc = loss_and_accuracy(model, b, subm.training_loss)
    assert torch.isfinite(loss), f"non-finite loss step {step}"
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 40 == 0 or step == 1:
        print(f"    step={step} loss={loss.item():.4f} train_em={acc:.3f} "
              f"lr={opt.param_groups[0]['lr']:.2e} prog={model.progress:.2f}")
print(f"[7] {steps} contract-train steps in {time.time()-t0:.1f}s, final loss finite")

# eval after training: purity again + edge (T=16 loops path) works
model.eval()
versions = {id(v): v._version for _, v in model_state_tensors(model)}
eb = collate(edge)
with torch.no_grad():
    loss_e, acc_e = loss_and_accuracy(model, eb, None)
assert {id(v): v._version for _, v in model_state_tensors(model)} == versions
print(f"[8] post-train eval OK (T=0/T=16 rows): loss={loss_e.item():.3f} em={acc_e:.3f}")

tb = collate(random.sample(rows, 128))
with torch.no_grad():
    loss_t, acc_t = loss_and_accuracy(model, tb, None)
print(f"[9] in-distribution eval after {steps} CPU steps: loss={loss_t.item():.3f} em={acc_t:.3f}")
print("ALL CONTRACT CHECKS PASSED")
