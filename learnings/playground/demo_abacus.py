"""Tiny demo: Abacus place ids on a toy digit-token sequence."""

from __future__ import annotations

import torch

from abacus import Abacus

# toy vocab: 0..9 are digit tokens 7..16 (matches competition DIGIT_OFFSET=7 style)
DIGIT_TOKENS = list(range(7, 17))


def encode_digits(s: str) -> list[int]:
    return [7 + int(ch) for ch in s]


def main() -> None:
    # LSD-first like the paper: number 123 → tokens for "321"
    # Two numbers in one sequence: 123 + 45 → "321+54" with '+' as non-digit 0
    ids = encode_digits("321") + [0] + encode_digits("54")
    batch = torch.tensor([ids])

    abacus = Abacus(DIGIT_TOKENS, embedding_dim=8, max_seq_length=64, max_k=10)
    abacus.eval()
    with torch.no_grad():
        mask = torch.isin(batch, abacus.digits)
        places = abacus.helper(mask, batch.device)
        emb = abacus(batch)

    print("tokens ", batch.tolist()[0])
    print("places ", places.tolist()[0], "  # 1=LSD within each span")
    print("emb shape", tuple(emb.shape))


if __name__ == "__main__":
    main()
