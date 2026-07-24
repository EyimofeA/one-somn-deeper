"""Pre-flight verification, run before trusting any long training run.

1. Build one real batch (Task A / square) and print every field: raw x/target,
   input token ids, decoded input, label token ids, decoded labels, output
   positions, attention mask, loss mask, and the model's untrained predictions.
2. Manually check x=37 -> target=1369 lands on digits [1,3,6,9] at the
   documented fixed-width output positions.
3. Mandatory sanity check: 32 examples, batch_size=32, same batch every step,
   dropout=0, weight_decay=0, 2000 steps, must overfit to ~100% exact-match /
   ~100% token accuracy / loss ~0. If this fails, no long run is trustworthy.
"""

from __future__ import annotations

import torch

from data.tokens import IGNORE_INDEX, OUT, PAD, encode_square
from models.transformer import StandardTransformer


def decode_input(ids: list[int]) -> str:
    names = {1: "SQUARE", 2: "MOD", 3: "SQUARE_MOD", 4: "SQUARE_MOD_TRACE", 5: "N", 6: "X", 7: "U", 8: "OUT", 0: "PAD"}
    out = []
    for t in ids:
        if t in names:
            out.append(names[t])
        elif t >= 9:
            out.append(str(t - 9))
        else:
            out.append(f"?{t}")
    return " ".join(out)


def decode_labels(labels: list[int]) -> str:
    return " ".join("_" if l == IGNORE_INDEX else str(l) for l in labels)


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> None:
    torch.manual_seed(0)

    # ---- 1 & 2: one real example, x=37, fully decoded ----
    x = 37
    target = x * x
    input_ids, labels = encode_square(x)
    section(f"Task A single example: x={x}, target=x*x={target}")
    print("raw x:", x)
    print("raw target:", target)
    print("input token ids:  ", input_ids)
    print("decoded input:    ", decode_input(input_ids))
    print("label token ids:  ", labels)
    print("decoded labels:   ", decode_labels(labels))

    output_width = sum(1 for l in labels if l != IGNORE_INDEX)
    output_positions = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
    loss_mask = [l != IGNORE_INDEX for l in labels]
    attention_mask = [True] * len(input_ids)  # no padding for a single fixed-width row
    print("output positions: ", output_positions, f"(width={output_width})")
    print("attention mask:   ", [int(m) for m in attention_mask])
    print("loss mask:        ", [int(m) for m in loss_mask])

    target_digits = [int(c) for c in f"{target:012d}"]  # NUM_SQUARE_DIGITS=12
    out_labels = [labels[i] for i in output_positions]
    print(f"expected digits of {target} (12-wide, zero-padded): {target_digits}")
    print("labels at output positions:                        ", out_labels)
    assert out_labels == target_digits, "label digits do not match target digits at output positions!"
    print("CHECK PASSED: label digits at output positions exactly match target's decimal digits.")

    # ---- untrained model prediction on this same example ----
    model = StandardTransformer(max_seq_len=len(input_ids), d_model=64, n_layers=2, n_heads=2, d_ff=128)
    model.eval()
    with torch.no_grad():
        ids_t = torch.tensor([input_ids])
        mask_t = torch.tensor([attention_mask])
        logits = model(ids_t, mask_t)
        preds = logits[0, output_positions, :].argmax(dim=-1).tolist()
    section("Untrained model prediction (before any training)")
    print("predicted digits at output positions:", preds)
    print("(expected to be essentially random relative to target — this is BEFORE training)")

    # ---- 3: mandatory 32-example overfit sanity check ----
    section("Mandatory overfit sanity check: 32 examples, same batch every step, 2000 steps")
    xs = list(range(1, 33))  # 32 distinct small x's
    seqs, labs = [], []
    for xv in xs:
        ids, lb = encode_square(xv)
        seqs.append(ids)
        labs.append(lb)
    input_batch = torch.tensor(seqs, dtype=torch.long)
    label_batch = torch.tensor(labs, dtype=torch.long)
    mask_batch = torch.ones(input_batch.shape, dtype=torch.bool)
    targets = label_batch[:, output_positions[0] :]  # trailing output_width columns

    model = StandardTransformer(
        max_seq_len=input_batch.shape[1], d_model=64, n_layers=2, n_heads=2, d_ff=128, dropout=0.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    print(f"dataset size=32, batch_size=32 (full-batch, identical every step), dropout=0, weight_decay=0")
    for step in range(1, 2001):
        logits = model(input_batch, mask_batch)[:, output_positions[0] :, :]
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
        logits = model(input_batch, mask_batch)[:, output_positions[0] :, :]
        preds = logits.argmax(dim=-1)
        final_exact = (preds == targets).all(dim=-1).float().mean().item()
        final_token = (preds == targets).float().mean().item()
        final_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1)).item()

    section("RESULT")
    print(f"final loss={final_loss:.6f}  exact_match={final_exact:.4f}  token_accuracy={final_token:.4f}")
    ok = final_exact >= 0.95 and final_token >= 0.99 and final_loss < 0.05
    print("PASS" if ok else "FAIL", "— implementation is" + ("" if ok else " NOT") + " trustworthy for a long run.")


if __name__ == "__main__":
    main()
