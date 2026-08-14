"""Two-digit multiplication Transformer: digit order x leading-zero factorial.

This is a fixed-width interpolation diagnostic, not a length-generalization
claim. Unordered operand pairs are assigned wholly to train or test so the
commuted form of a held-out pair cannot leak into training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DIGITS = 10
PAD, BOS, EOS, MUL = 10, 11, 12, 13
VOCAB = 14
MAX_OPERAND_DIGITS = 2
MAX_PRODUCT_DIGITS = 4


def decimal_digits(value: int, order: str, width: int | None = None) -> list[int]:
    text = str(value) if width is None else f"{value:0{width}d}"
    values = [int(character) for character in text]
    return values[::-1] if order == "lsd" else values


def encode_example(a: int, b: int, order: str, padded: bool):
    operand_width = MAX_OPERAND_DIGITS if padded else None
    product_width = MAX_PRODUCT_DIGITS if padded else None
    source = decimal_digits(a, order, operand_width) + [MUL] + decimal_digits(b, order, operand_width)
    target = decimal_digits(a * b, order, product_width)
    return source, target


def split_examples(seed: int):
    groups = [(a, b) for a in range(100) for b in range(a, 100)]
    rng = random.Random(seed)
    rng.shuffle(groups)
    test_groups = set(groups[: round(0.2 * len(groups))])
    train, test = [], []
    for a in range(100):
        for b in range(100):
            group = (min(a, b), max(a, b))
            (test if group in test_groups else train).append((a, b))
    return train, test


class MultiplicationTransformer(nn.Module):
    def __init__(self, d_model=64, heads=4, layers=2, ff=128, dropout=0.0):
        super().__init__()
        self.token = nn.Embedding(VOCAB, d_model)
        self.source_position = nn.Embedding(5, d_model)
        self.target_position = nn.Embedding(5, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, heads, ff, dropout, batch_first=True, norm_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, heads, ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, VOCAB)

    def forward(self, source, decoder_input, source_padding, target_padding):
        source_positions = torch.arange(source.shape[1], device=source.device)
        target_positions = torch.arange(decoder_input.shape[1], device=source.device)
        source_state = self.token(source) + self.source_position(source_positions)[None]
        target_state = self.token(decoder_input) + self.target_position(target_positions)[None]
        memory = self.encoder(source_state, src_key_padding_mask=source_padding)
        causal = nn.Transformer.generate_square_subsequent_mask(
            decoder_input.shape[1], device=source.device
        )
        state = self.decoder(
            target_state,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=target_padding,
            memory_key_padding_mask=source_padding,
        )
        return self.output(self.norm(state))


def collate(examples, order, padded, device):
    encoded = [(a, b, *encode_example(a, b, order, padded)) for a, b in examples]
    source_length = max(len(row[2]) for row in encoded)
    target_length = max(len(row[3]) for row in encoded) + 1
    source = torch.full((len(encoded), source_length), PAD, dtype=torch.long, device=device)
    decoder_input = torch.full((len(encoded), target_length), PAD, dtype=torch.long, device=device)
    labels = torch.full((len(encoded), target_length), -100, dtype=torch.long, device=device)
    for index, (_, _, source_tokens, target_tokens) in enumerate(encoded):
        source[index, : len(source_tokens)] = torch.tensor(source_tokens, device=device)
        decoder_tokens = [BOS] + target_tokens
        label_tokens = target_tokens + [EOS]
        decoder_input[index, : len(decoder_tokens)] = torch.tensor(decoder_tokens, device=device)
        labels[index, : len(label_tokens)] = torch.tensor(label_tokens, device=device)
    return source, decoder_input, labels, source.eq(PAD), decoder_input.eq(PAD)


@torch.no_grad()
def generate(model, examples, order, padded, device):
    source, _, _, source_padding, _ = collate(examples, order, padded, device)
    generated = torch.full((len(examples), 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(len(examples), dtype=torch.bool, device=device)
    for _ in range(MAX_PRODUCT_DIGITS + 1):
        logits = model(
            source,
            generated,
            source_padding,
            generated.eq(PAD),
        )
        token = logits[:, -1].argmax(-1)
        generated = torch.cat((generated, token[:, None]), dim=1)
        finished |= token.eq(EOS)
        if finished.all():
            break
    return generated[:, 1:].cpu().tolist()


def parse_prediction(tokens: list[int], order: str):
    if EOS not in tokens:
        return None, None, "missing_eos"
    digits = tokens[: tokens.index(EOS)]
    if not digits or any(token >= DIGITS for token in digits):
        return None, None, "non_digit"
    natural = digits[::-1] if order == "lsd" else digits
    text = "".join(map(str, natural))
    return int(text), text, None


def digit_length(value: int) -> int:
    return len(str(value))


def carry_count(a: int, b: int) -> int:
    left = decimal_digits(a, "lsd", MAX_OPERAND_DIGITS)
    right = decimal_digits(b, "lsd", MAX_OPERAND_DIGITS)
    carry = count = 0
    for column in range(2 * MAX_OPERAND_DIGITS):
        total = carry
        for i, left_digit in enumerate(left):
            j = column - i
            if 0 <= j < len(right):
                total += left_digit * right[j]
        carry = total // 10
        count += int(carry > 0)
    return count


def bucket_report(counts):
    return {
        str(key): {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1]}
        for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
    }


@torch.no_grad()
def evaluate(model, examples, order, padded, device, batch_size=512):
    predictions = {}
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        rows = generate(model, chunk, order, padded, device)
        predictions.update({pair: row for pair, row in zip(chunk, rows)})

    exact = length_correct = invalid = 0
    operand_buckets = defaultdict(lambda: [0, 0])
    product_buckets = defaultdict(lambda: [0, 0])
    carry_buckets = defaultdict(lambda: [0, 0])
    digit_correct = [0] * MAX_PRODUCT_DIGITS
    digit_total = [0] * MAX_PRODUCT_DIGITS
    commutative_correct = commutative_total = 0
    for a, b in examples:
        truth = a * b
        value, text, error = parse_prediction(predictions[(a, b)], order)
        correct = value == truth
        exact += int(correct)
        invalid += int(error is not None)
        if text is not None:
            expected_length = MAX_PRODUCT_DIGITS if padded else digit_length(truth)
            length_correct += int(len(text) == expected_length)
        operand_key = f"{digit_length(a)}x{digit_length(b)}"
        product_key = digit_length(truth)
        carry_key = carry_count(a, b)
        for buckets, key in (
            (operand_buckets, operand_key),
            (product_buckets, product_key),
            (carry_buckets, carry_key),
        ):
            buckets[key][0] += int(correct)
            buckets[key][1] += 1
        true_lsd = decimal_digits(truth, "lsd")
        predicted_lsd = decimal_digits(value, "lsd") if value is not None else []
        for position, true_digit in enumerate(true_lsd):
            digit_total[position] += 1
            digit_correct[position] += int(
                position < len(predicted_lsd) and predicted_lsd[position] == true_digit
            )
        if (b, a) in predictions and a <= b:
            other_value, _, _ = parse_prediction(predictions[(b, a)], order)
            commutative_correct += int(value is not None and value == other_value)
            commutative_total += 1

    return {
        "exact_numerical_accuracy": exact / len(examples),
        "output_length_accuracy": length_correct / len(examples),
        "invalid_output_rate": invalid / len(examples),
        "digit_accuracy_lsd_first": [
            {"position": position, "correct": digit_correct[position], "total": digit_total[position],
             "accuracy": digit_correct[position] / digit_total[position] if digit_total[position] else None}
            for position in range(MAX_PRODUCT_DIGITS)
        ],
        "by_operand_digit_lengths": bucket_report(operand_buckets),
        "by_product_digit_length": bucket_report(product_buckets),
        "by_carry_columns": bucket_report(carry_buckets),
        "commutativity_consistency": commutative_correct / commutative_total,
        "commutativity_pairs": commutative_total,
        "examples": len(examples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--order", choices=("msd", "lsd"), required=True)
    parser.add_argument("--padded", action="store_true")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, test = split_examples(args.seed)
    device = torch.device(args.device)
    model = MultiplicationTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
    batch_size = 256
    curve = []
    for step in range(1, args.steps + 1):
        start = ((step - 1) * batch_size) % len(train)
        sample = train[start : start + batch_size]
        if len(sample) < batch_size:
            sample += train[: batch_size - len(sample)]
        source, decoder_input, labels, source_padding, target_padding = collate(
            sample, args.order, args.padded, device
        )
        model.train()
        logits = model(source, decoder_input, source_padding, target_padding)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 400 == 0:
            record = {"step": step, "loss": float(loss.detach())}
            curve.append(record)
            print(json.dumps({"type": "progress", **record}), flush=True)

    model.eval()
    report = {
        "classification": "RESEARCH ONLY - fixed-width multiplication interpolation",
        "operand_range": [0, 99],
        "split": "80/20 deterministic unordered-pair groups; commuted pairs remain together",
        "order": args.order,
        "explicit_leading_zero_padding": args.padded,
        "steps": args.steps,
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_examples": len(train),
        "test_examples": len(test),
        "curve": curve,
        "train": evaluate(model, train, args.order, args.padded, device),
        "test": evaluate(model, test, args.order, args.padded, device),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), args.out / "model.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
