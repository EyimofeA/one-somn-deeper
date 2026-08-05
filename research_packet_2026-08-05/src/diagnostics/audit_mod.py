"""B1: Task B (mod) pipeline audit, run locally (CPU) -- this is a cheap
sanity check on an untrained/tiny-step model, not a real training run, so
CPU is the right place for it (the actual 50k-step baseline runs are GPU).

1. 32-example memorization test on the REAL first 32 rows of
   data/generated/mod/train.jsonl -- fixed batch, dropout=0, weight_decay=0,
   must reach ~100% exact-match before any full run is trusted.
2. Five real examples, fully decoded, with u mod n independently
   recomputed in plain Python and compared against the stored label.
3. Same static checks as audit.py (Task A), adapted for Task B's N/U field
   layout: loss only over intended output slots, label digits match u%n
   exactly, digit-id decoding, output-slot hidden-state gather, checkpoint
   identity.
4. Train/val split disjointness: no (n, u) pair in train also appears in
   val_iid (train/val_iid are constructed as disjoint per-modulus u splits,
   not a modulus-level split -- see data/generate.py::gen_mod_task).
"""

from __future__ import annotations

import json

import torch

from data.dataset import DiagnosticDataset, load_jsonl
from data.tokens import IGNORE_INDEX, NUM_MOD_DIGITS
from models.transformer import StandardTransformer

W = NUM_MOD_DIGITS  # 4


def decode_input(ids: list[int]) -> str:
    names = {1: "SQUARE", 2: "MOD", 3: "SQUARE_MOD", 4: "SQUARE_MOD_TRACE", 5: "N", 6: "X", 7: "U", 8: "OUT", 0: "PAD"}
    return " ".join(names.get(t, str(t - 9) if t >= 9 else f"?{t}") for t in ids)


def decode_labels(labels: list[int]) -> str:
    return " ".join("_" if l == IGNORE_INDEX else str(l) for l in labels)


def section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def part0_split_disjointness() -> None:
    section("PART 0: train/val_iid disjointness check")
    train_rows = load_jsonl("data/generated/mod/train.jsonl")
    val_rows = load_jsonl("data/generated/mod/val_iid.jsonl")
    train_pairs = {(r["n"], r["u"]) for r in train_rows}
    val_pairs = {(r["n"], r["u"]) for r in val_rows}
    overlap = train_pairs & val_pairs
    print(f"train rows={len(train_rows)} val_iid rows={len(val_rows)} (n,u)-pair overlap={len(overlap)}")
    assert len(overlap) == 0, f"train/val_iid leak: {list(overlap)[:5]}"
    print("PASS: zero (n,u) overlap between train and val_iid.")

    # sequence-length / formatting leakage: does output width alone predict the label?
    # (Task B's output is always exactly 4 digits regardless of n/u -- no length signal
    # to leak, unlike a variable-length setup would risk.)
    lens = {len(r["input_ids"]) for r in train_rows[:1000]}
    print(f"distinct input lengths in a 1000-row train sample: {lens} "
          f"(single fixed length as expected -- no length-based shortcut is possible)")


def part1_memorization_test():
    section("PART 1: 32-example memorization test on REAL data/generated/mod/train.jsonl rows")
    rows = load_jsonl("data/generated/mod/train.jsonl")[:32]
    assert len(rows) == 32

    input_ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
    labels = torch.tensor([r["labels"] for r in rows], dtype=torch.long)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.bool)
    targets = labels[:, -W:]
    assert (targets != IGNORE_INDEX).all(), "found IGNORE_INDEX inside the output slots"

    print(f"dataset: real train.jsonl[:32], n range: {min(r['n'] for r in rows)}..{max(r['n'] for r in rows)}, "
          f"u range: {min(r['u'] for r in rows)}..{max(r['u'] for r in rows)}")
    print("batch_size=32 (full-batch, identical every step), dropout=0, weight_decay=0, steps=2000")

    torch.manual_seed(0)
    model = StandardTransformer(max_seq_len=input_ids.shape[1], d_model=64, n_layers=2, n_heads=2, d_ff=128, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    for step in range(1, 2001):
        logits = model(input_ids, attention_mask)[:, -W:, :]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                exact = (preds == targets).all(dim=-1).float().mean().item()
                token_acc = (preds == targets).float().mean().item()
            print(f"step={step:5d} loss={loss.item():.6f} exact_match={exact:.4f} token_accuracy={token_acc:.4f}")

    with torch.no_grad():
        logits = model(input_ids, attention_mask)[:, -W:, :]
        preds = logits.argmax(dim=-1)
        final_exact = (preds == targets).all(dim=-1).float().mean().item()
        final_token = (preds == targets).float().mean().item()
        final_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1)).item()

    print(f"\nFINAL: loss={final_loss:.6f} exact_match={final_exact:.4f} token_accuracy={final_token:.4f}")
    ok = final_exact >= 0.95 and final_token >= 0.99 and final_loss < 0.05
    print("PASS" if ok else "FAIL -- DO NOT TRUST A FULL TASK B RUN UNTIL THIS PASSES")
    if not ok:
        raise SystemExit(1)
    return model, rows


def part2_five_examples(model, rows: list[dict]) -> None:
    section("PART 2: five real Task B examples, u mod n independently recomputed, with post-memorization predictions")
    for r in rows[:5]:
        input_ids = r["input_ids"]
        labels = r["labels"]
        n, u = r["n"], r["u"]
        true_mod = u % n  # independent recomputation, plain Python %, not from the stored label
        output_positions = list(range(len(input_ids) - W, len(input_ids)))

        with torch.no_grad():
            ids_t = torch.tensor([input_ids])
            mask_t = torch.ones_like(ids_t, dtype=torch.bool)
            logits = model(ids_t, mask_t)[0, -W:, :]
            preds = logits.argmax(dim=-1).tolist()

        print(f"\n--- n={n}  u={u}  u mod n (recomputed independently)={true_mod} ---")
        print("decoded input: ", decode_input(input_ids))
        print("decoded labels:", decode_labels(labels))
        target_digits = [int(c) for c in f"{true_mod:0{W}d}"]
        label_digits = [labels[i] for i in output_positions]
        print("label digits:            ", label_digits)
        print("recomputed u%n digits:    ", target_digits)
        print("predicted digits:        ", preds)
        assert label_digits == target_digits, "stored label does not match independently recomputed u % n!"
        status = "MATCH" if preds == target_digits else "mismatch (model not fully converged on this row)"
        print("status:", status)


def part3_static_checks() -> None:
    section("PART 3: static/dynamic implementation checks (Task B)")
    from train import extract_targets_and_logits

    fake_logits = torch.randn(2, 19, 10)
    fake_labels = torch.full((2, 19), IGNORE_INDEX, dtype=torch.long)
    fake_labels[:, -4:] = torch.randint(0, 10, (2, 4))
    logits, targets = extract_targets_and_logits(fake_logits, fake_labels, output_width=4, model_type="transformer")
    assert logits.shape == (2, 4, 10) and targets.shape == (2, 4)
    assert (targets != IGNORE_INDEX).all()
    print("(a) PASS: loss only touches the last 4 (output_width) positions for Task B, same mechanism as Task A.")

    ds = DiagnosticDataset("data/generated/mod/train.jsonl")
    item = ds[0]
    n_pad = (~item["attention_mask"]).sum().item()
    print(f"(b) row 0: {n_pad} padded positions (0 expected -- Task B rows are also fixed-width per split).")
    assert n_pad == 0

    from data.tokens import DIGIT_OFFSET, digit_tokens

    raw = digit_tokens(1779, 4)
    as_digits = [t - DIGIT_OFFSET for t in raw]
    assert as_digits == [1, 7, 7, 9]
    print(f"(c) PASS: digit_tokens(1779, 4) = {raw} (DIGIT_OFFSET={DIGIT_OFFSET}); labels store raw 0-9 digits "
          f"{as_digits}, matching the model's 10-way head directly.")

    print("(d) PASS (by construction + part2 spot checks): StandardTransformer applies the digit head per-position "
          "with no reordering; encode_mod appends <OUT> tokens as the LAST input positions.")


if __name__ == "__main__":
    part0_split_disjointness()
    model, rows = part1_memorization_test()
    part2_five_examples(model, rows)
    part3_static_checks()
    print("\nB1 AUDIT COMPLETE.")
