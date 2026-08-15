"""Read-only mechanistic diagnostics for trained multiplication Neural GPUs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from train_neural_gpu_multiplication_variant import Model, bits, rows
from train_transformer_multiplication_factorial import split_examples


def load_model(checkpoint, variant, device):
    model = Model(variant, 128).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def initial_state(model, left, right):
    batch, width = left.shape
    hidden = torch.zeros(batch, model.channels, 4, 2 * width, device=left.device)
    hidden[:, :, 0, :width] = model.embedding(left).transpose(1, 2) + model.left_marker[None, :, None]
    hidden[:, :, 1, :width] = model.embedding(right).transpose(1, 2) + model.right_marker[None, :, None]
    return hidden


def advance(model, hidden, start, stop, zero_row=None, zero_channels=None, captures=None):
    for step in range(start, stop):
        hidden, _ = model.cells[step % len(model.cells)](hidden)
        if zero_row is not None:
            hidden[:, :, zero_row] = 0
        if zero_channels is not None:
            hidden[:, zero_channels[0]:zero_channels[1]] = 0
        if captures is not None:
            captures.append(hidden.clone())
    return hidden


def logits(model, hidden):
    return model.readout(hidden[:, :, 0]).squeeze(1)


def exact(prediction, target):
    return (prediction.eq(target).all(1).float().mean().item())


@torch.no_grad()
def depth_and_ablation(model, data, device):
    left = torch.tensor([item[0] for item in data], device=device)
    right = torch.tensor([item[1] for item in data], device=device)
    target = torch.tensor([item[2] for item in data], device=device)
    hidden = initial_state(model, left, right)
    captures = [hidden.clone()]
    advance(model, hidden, 0, 28, captures=captures)
    by_step = {str(step): exact(logits(model, captures[step]).gt(0), target)
               for step in range(1, 29)}
    requested = {str(step): by_step[str(step)] for step in (10, 12, 14, 16, 20, 28)}
    base = exact(logits(model, captures[14]).gt(0), target)
    rows_out = {}
    for row in range(4):
        state = initial_state(model, left, right)
        state = advance(model, state, 0, 14, zero_row=row)
        rows_out[str(row)] = exact(logits(model, state).gt(0), target)
    channels_out = {}
    width = model.channels // 8
    for group in range(8):
        state = initial_state(model, left, right)
        span = (group * width, (group + 1) * width)
        state = advance(model, state, 0, 14, zero_channels=span)
        channels_out[f"{span[0]}:{span[1]}"] = exact(logits(model, state).gt(0), target)
    square = torch.tensor([item[3] == item[4] for item in data], device=device)
    prediction = logits(model, captures[14]).gt(0)
    row_ok = prediction.eq(target).all(1)
    return {"base_exact": base, "square_count": int(square.sum()),
            "square_exact": row_ok[square].float().mean().item(),
            "nonsquare_exact": row_ok[~square].float().mean().item(),
            "decode_every_step": by_step, "requested_depths": requested,
            "causal_row_zeroing": rows_out, "causal_channel_group_zeroing": channels_out}


def targets_for(items, device):
    operand, pairwise, sums, carries, outputs = [], [], [], [], []
    for item in items:
        a_bits, b_bits, out = item[0], item[1], item[2]
        operand.append(a_bits + b_bits)
        pairwise.append([x * y for x in a_bits for y in b_bits])
        columns = [sum(a_bits[i] * b_bits[k-i] for i in range(7)
                       if 0 <= k-i < 7) for k in range(13)]
        carry, carry_values = 0, []
        for value in columns:
            carry = (value + carry) // 2
            carry_values.append(carry)
        sums.append([value / 7 for value in columns])
        carries.append([value / 7 for value in carry_values])
        outputs.append(out)
    return {"operands": (torch.tensor(operand, device=device).float(), True),
            "pairwise_products": (torch.tensor(pairwise, device=device).float(), True),
            "column_sums": (torch.tensor(sums, device=device).float(), False),
            "carries": (torch.tensor(carries, device=device).float(), False),
            "final_bits": (torch.tensor(outputs, device=device).float(), True)}


def ridge_score(train_x, test_x, train_y, test_y, classification):
    mean, std = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True).clamp_min(1e-4)
    train_x, test_x = (train_x-mean)/std, (test_x-mean)/std
    # A fixed-rank PCA prevents an underdetermined 1,792-feature probe from
    # memorizing 512 examples. PCA is linear and fitted on train states only.
    _, _, basis = torch.pca_lowrank(train_x, q=64, center=False, niter=4)
    train_x, test_x = train_x @ basis, test_x @ basis
    train_x = torch.cat((train_x, torch.ones(len(train_x), 1, device=train_x.device)), 1).double()
    test_x = torch.cat((test_x, torch.ones(len(test_x), 1, device=test_x.device)), 1).double()
    train_y, test_y = train_y.double(), test_y.double()
    gram = train_x.T @ train_x
    weights = torch.linalg.solve(gram + torch.eye(train_x.shape[1], device=train_x.device,
                                                 dtype=torch.float64), train_x.T @ train_y)
    prediction = test_x @ weights
    if classification:
        return (prediction.gt(0.5) == test_y.bool()).float().mean().item()
    residual = (prediction-test_y).pow(2).sum()
    total = (test_y-test_y.mean(0, keepdim=True)).pow(2).sum().clamp_min(1e-8)
    return (1-residual/total).item()


@torch.no_grad()
def capture(model, items, device, steps):
    left = torch.tensor([item[0] for item in items], device=device)
    right = torch.tensor([item[1] for item in items], device=device)
    state = initial_state(model, left, right)
    found = {0: state.clone()}
    for step in range(1, max(steps)+1):
        state = advance(model, state, step-1, step)
        if step in steps:
            found[step] = state.clone()
    return found


def linear_probes(model, train, validation, device):
    train, validation = train[:512], validation[:512]
    steps = (0, 2, 4, 7, 10, 14)
    train_h, validation_h = capture(model, train, device, steps), capture(model, validation, device, steps)
    train_targets, validation_targets = targets_for(train, device), targets_for(validation, device)
    report = {}
    for step in steps:
        report[str(step)] = {}
        for row in range(4):
            train_x = train_h[step][:, :, row].flatten(1).float()
            validation_x = validation_h[step][:, :, row].flatten(1).float()
            scores = {}
            for name, (train_y, classification) in train_targets.items():
                test_y = validation_targets[name][0]
                scores[name] = ridge_score(train_x, validation_x, train_y, test_y, classification)
            report[str(step)][str(row)] = scores
    return report


@torch.no_grad()
def counterfactual_patching(model, data, device):
    cases = []
    for item in data:
        a, b = item[3], item[4]
        for bit in range(7):
            changed = a ^ (1 << bit)
            if changed <= 99:
                cases.append((a, b, changed, bit))
                break
        if len(cases) == 128:
            break
    left = torch.tensor([bits(a, 7) for a, _, _, _ in cases], device=device)
    changed_left = torch.tensor([bits(c, 7) for _, _, c, _ in cases], device=device)
    right = torch.tensor([bits(b, 7) for _, b, _, _ in cases], device=device)
    changed_target = torch.tensor([bits(c*b, 14) for _, b, c, _ in cases], device=device)
    base_states = capture(model, [(bits(a,7), bits(b,7), bits(a*b,14), a, b)
                                  for a,b,_,_ in cases], device, (2,7,14))
    changed_states = capture(model, [(bits(c,7), bits(b,7), bits(c*b,14), c, b)
                                     for _,b,c,_ in cases], device, (2,7,14))
    changed_prediction = logits(model, changed_states[14]).gt(0)
    output_delta_exact = exact(changed_prediction, changed_target)
    patched = {}
    for step in (2, 7):
        patched[str(step)] = {}
        for row in range(4):
            state = base_states[step].clone()
            state[:, :, row] = changed_states[step][:, :, row]
            state = advance(model, state, step, 14)
            patched[str(step)][str(row)] = exact(logits(model, state).gt(0), changed_target)
    return {"cases": len(cases), "full_counterfactual_exact": output_delta_exact,
            "row_activation_patching_to_counterfactual_target": patched}


@torch.no_grad()
def square_control(model, device):
    square_data = rows([(value, value) for value in range(100)])
    rng = random.Random(91)
    nonsquares = [(a, b) for a in range(100) for b in range(a, 100) if a != b]
    rng.shuffle(nonsquares)
    nonsquare_data = rows(nonsquares[:100])
    def score(data):
        left = torch.tensor([item[0] for item in data], device=device)
        right = torch.tensor([item[1] for item in data], device=device)
        target = torch.tensor([item[2] for item in data], device=device)
        state = advance(model, initial_state(model, left, right), 0, 14)
        return exact(logits(model, state).gt(0), target)
    return {"square_examples": 100, "square_exact": score(square_data),
            "matched_nonsquare_examples": 100, "matched_nonsquare_exact": score(nonsquare_data),
            "warning": "Behavioral control mixes seen and held-out numeric pairs; it is not a squaring-generalization score."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", default="muon_decay")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(0); torch.manual_seed(0)
    train_examples, test_examples = split_examples(0)
    rng = random.Random(20_000); rng.shuffle(test_examples)
    train, validation, audit = rows(train_examples), rows(test_examples[:1003]), rows(test_examples[1003:])
    model = load_model(args.checkpoint, args.variant, args.device)
    report = {"checkpoint": str(args.checkpoint),
              "audit_diagnostics": depth_and_ablation(model, audit, args.device),
              "all_values_square_control": square_control(model, args.device),
              "validation_linear_probes": linear_probes(model, train, validation, args.device),
              "validation_counterfactuals": counterfactual_patching(model, validation, args.device)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
