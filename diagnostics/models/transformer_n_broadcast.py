"""Standard Transformer with a learned input-N route into output slots."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from data.tokens import NUM_MOD_DIGITS, VOCAB_SIZE


class NBroadcastTransformer(nn.Module):
    """Inject the mean input state of the four N digits into mod output slots.

    The projection is learned separately at every layer. Using the input N
    state keeps the shuffled control semantic: it swaps only N information,
    rather than another example's U-dependent contextual state.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = 32,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.0,
        digit_vocab: int = 10,
        shuffle_n_broadcast: bool = False,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        ) for _ in range(n_layers))
        self.n_to_output = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(n_layers))
        self.shuffle_n_broadcast = shuffle_n_broadcast
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, digit_vocab)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        _, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        h = self.token_embed(input_ids) + self.pos_embed(positions)[None, :, :]
        key_padding_mask = ~attention_mask if attention_mask is not None else None
        n_input_state = h[:, 2:6].mean(dim=1, keepdim=True)
        if self.shuffle_n_broadcast:
            n_input_state = n_input_state[self._wrong_n_permutation(input_ids)]
        for layer, n_to_output in zip(self.layers, self.n_to_output):
            h = layer(h, src_key_padding_mask=key_padding_mask)
            h = h.clone()
            h[:, -NUM_MOD_DIGITS:] += n_to_output(n_input_state)
        return self.head(self.norm(h))

    @staticmethod
    def _wrong_n_permutation(input_ids: Tensor) -> Tensor:
        """A deterministic in-batch permutation that mismatches N whenever possible."""
        n_tokens = input_ids[:, 2:6]
        groups = torch.unique(n_tokens, dim=0, return_inverse=True)[1]
        indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        perm = indices.clone()
        # Pair examples from the two groups in opposite directions. Any excess
        # majority-group examples must remain within that group to preserve a permutation.
        group_ids = torch.unique(groups)
        if len(group_ids) != 2:
            return torch.roll(indices, 1)
        left, right = (indices[groups == g] for g in group_ids)
        paired = min(len(left), len(right))
        perm[left[:paired]] = right[:paired]
        perm[right[:paired]] = left[:paired]
        if len(left) > paired:
            perm[left[paired:]] = torch.roll(left[paired:], 1)
        if len(right) > paired:
            perm[right[paired:]] = torch.roll(right[paired:], 1)
        return perm
