"""Diagnostic-only intermediate-state supervision for the final-label VDF card.

Arithmetic is used only to construct research labels. This is not a legal
competition training program and cannot be submitted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from benchmark import ModelSpec


def source_module(path):
    spec = importlib.util.spec_from_file_location("vdf_card", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def decimal(tokens):
    value = 0
    for token in tokens:
        value = value * 10 + token - 7
    return value


def rows(path):
    output = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        prompt, labels = record["input_ids"], record["labels"]
        x_mark, t_mark = prompt.index(3), prompt.index(4)
        modulus, state, depth = decimal(prompt[1:x_mark]), decimal(prompt[x_mark + 1:t_mark]), decimal(prompt[t_mark + 1:])
        trace = []
        for _ in range(depth):
            state = state * state % modulus
            trace.append([7 + int(digit) for digit in f"{state:0{len(labels)}d}"])
        output.append((prompt, labels, trace, depth))
    return output


def batch(items, device):
    sequence, answer, depth = max(len(item[0]) for item in items), max(len(item[1]) for item in items), max(item[3] for item in items)
    ids = torch.zeros(len(items), sequence, dtype=torch.long, device=device)
    mask = torch.zeros(len(items), sequence, dtype=torch.bool, device=device)
    labels = torch.full((len(items), answer), -100, dtype=torch.long, device=device)
    positions = torch.full((len(items), answer), -1, dtype=torch.long, device=device)
    trace = torch.full((len(items), depth, answer), -100, dtype=torch.long, device=device)
    steps = torch.tensor([item[3] for item in items], device=device)
    for row, (prompt, target, targets, _) in enumerate(items):
        ids[row, :len(prompt)] = torch.tensor(prompt, device=device)
        mask[row, :len(prompt)] = True
        labels[row, :len(target)] = torch.tensor(target, device=device)
        positions[row, :len(target)] = torch.arange(len(prompt) - len(target), len(prompt), device=device)
        for step, target_state in enumerate(targets):
            trace[row, step, :len(target_state)] = torch.tensor(target_state, device=device)
    return ids, mask, labels, positions, trace, steps


def forward_trace(model, source, ids, mask):
    """The imported VDF forward, with readouts retained after each tied F step."""
    size, length = ids.shape
    field, place, steps, register = source.prompt_layout(ids, mask)
    base = model.token(ids) + model.position(torch.arange(length, device=ids.device)) + model.field(field) + model.place(place)
    state = torch.zeros(size, length, model.config.vocab_size, device=ids.device, dtype=base.dtype)
    hidden, outputs = base, []
    for depth in range(int(steps.max().item())):
        index = torch.nonzero(steps > depth, as_tuple=True)[0]
        active_base, active_state = base[index], state[index]
        active_mask, active_register = mask[index], register[index]
        squared = model.square(active_base + model.register_projection(active_state), active_mask)
        reduced = model.reduce(active_base + squared, active_mask)
        logits = model.head(model.norm(reduced[active_register]))
        soft = logits.softmax(-1)
        hard = F.one_hot(logits.argmax(-1), model.config.vocab_size).to(soft.dtype)
        next_state = hard + soft - soft.detach() if model.training else hard
        rows, columns = active_register.nonzero(as_tuple=True)
        state = state.index_copy(0, index, active_state.index_put((rows, columns), next_state))
        hidden = hidden.index_copy(0, index, reduced)
        outputs.append(model.head(model.norm(hidden)))
    return outputs[-1], outputs


def gathered(logits, positions):
    return logits[torch.arange(logits.shape[0], device=logits.device)[:, None], positions]


@torch.no_grad()
def score(model, source, items, device):
    model.eval(); by_t = {}
    for start in range(0, len(items), 512):
        ids, mask, labels, positions, _, steps = batch(items[start:start + 512], device)
        logits, _ = forward_trace(model, source, ids, mask)
        correct = (gathered(logits, positions).argmax(-1) == labels).all(-1)
        for depth, value in zip(steps.tolist(), correct.tolist()): by_t.setdefault(depth, []).append(value)
    return {str(depth): sum(values) / len(values) for depth, values in sorted(by_t.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--trace-weight", type=float, default=1.0)
    args = parser.parse_args()
    torch.manual_seed(74); device = torch.device("cuda")
    source = source_module(args.submission)
    train, test, ood = (rows(args.data_root / f"{split}.jsonl") for split in ("train", "test", "ood"))
    max_length = max(len(row[0]) for row in train + test + ood)
    model = source.VDFModel(ModelSpec(17, max_length, 500_000_000)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .95), weight_decay=.05)
    started, step, history = time.monotonic(), 0, []
    while time.monotonic() - started < args.seconds:
        offset = (step * 512) % len(train); items = (train + train)[offset:offset + 512]
        ids, mask, labels, positions, trace, steps = batch(items, device)
        model.train(); final_logits, all_logits = forward_trace(model, source, ids, mask)
        final_loss = F.cross_entropy(gathered(final_logits, positions).reshape(-1, 17), labels.reshape(-1))
        trace_loss = torch.zeros((), device=device)
        for depth, logits in enumerate(all_logits):
            active = steps > depth
            trace_loss += F.cross_entropy(gathered(logits[active], positions[active]).reshape(-1, 17), trace[active, depth].reshape(-1))
        loss = final_loss + args.trace_weight * trace_loss / len(all_logits)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); step += 1
        if step == 1 or step % 100 == 0:
            history.append({"step": step, "seconds": round(time.monotonic() - started, 1), "loss": float(loss), "final_loss": float(final_loss), "trace_loss": float(trace_loss / len(all_logits)), "train_final_exact": float((gathered(final_logits, positions).argmax(-1) == labels).all(-1).float().mean())})
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"classification": "DIAGNOSTIC ONLY — generated intermediate labels", "steps": step, "seconds": time.monotonic() - started, "trace_weight": args.trace_weight, "train_curve": history, "test_final_exact_by_T": score(model, source, test, device), "ood_final_exact_by_T": score(model, source, ood, device)}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
