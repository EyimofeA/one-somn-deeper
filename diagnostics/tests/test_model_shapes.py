import torch

from data.tokens import NUM_MOD_DIGITS, NUM_SQUARE_DIGITS, VOCAB_SIZE, encode_mod, encode_square
from models.recurrent_workspace import RecurrentWorkspaceModel
from models.transformer import StandardTransformer
from models.transformer_n_broadcast import NBroadcastTransformer

W = NUM_SQUARE_DIGITS  # output width for Task A (square)


def _batch(n=4, x_values=(3, 12, 99, 4321)):
    seqs, labs = [], []
    for x in x_values[:n]:
        ids, labels = encode_square(x)
        seqs.append(ids)
        labs.append(labels)
    return (
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(labs, dtype=torch.long),
        torch.ones(len(seqs), len(seqs[0]), dtype=torch.bool),
    )


def test_standard_transformer_forward_backward_smoke():
    input_ids, labels, mask = _batch()
    model = StandardTransformer(max_seq_len=input_ids.shape[1], d_model=32, n_layers=2, n_heads=2, d_ff=64)
    logits = model(input_ids, mask)
    assert logits.shape == (input_ids.shape[0], input_ids.shape[1], 10)
    out_logits = logits[:, -W:, :]
    targets = labels[:, -W:]
    loss = torch.nn.functional.cross_entropy(out_logits.reshape(-1, 10), targets.reshape(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_n_broadcast_transformer_forward_backward_smoke():
    ids, labels = zip(*(encode_mod(n, u) for n, u in ((1349, 2715), (1357, 2731))))
    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    for shuffled in (False, True):
        model = NBroadcastTransformer(
            max_seq_len=input_ids.shape[1], d_model=32, n_layers=2, n_heads=2, d_ff=64,
            shuffle_n_broadcast=shuffled,
        )
        logits = model(input_ids, mask)[:, -NUM_MOD_DIGITS:, :]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), labels[:, -NUM_MOD_DIGITS:].reshape(-1))
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_recurrent_workspace_forward_backward_smoke():
    input_ids, labels, mask = _batch()
    model = RecurrentWorkspaceModel(
        max_seq_len=input_ids.shape[1], d_model=32, n_heads=2, d_ff=64,
        workspace_size=W, num_output_slots=W, num_loops=3,
    )
    logits = model(input_ids, mask)
    assert logits.shape == (input_ids.shape[0], W, 10)
    targets = labels[:, -W:]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_input_conditioned_workspace_modes_forward_backward_smoke():
    ids, labels = zip(*(encode_mod(1349, u) for u in (2715, 4087, 6791)))
    input_ids = torch.tensor(ids, dtype=torch.long)
    targets = torch.tensor(labels, dtype=torch.long)[:, -NUM_MOD_DIGITS:]
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    for mode in ("input_context", "shuffled_context"):
        model = RecurrentWorkspaceModel(
            max_seq_len=input_ids.shape[1], d_model=32, n_heads=2, d_ff=64,
            workspace_size=NUM_MOD_DIGITS, num_output_slots=NUM_MOD_DIGITS,
            num_loops=3, workspace_init_mode=mode,
        )
        kwargs = {}
        if mode == "shuffled_context":
            kwargs = {
                "init_input_ids": input_ids.roll(1, 0),
                "init_attention_mask": mask.roll(1, 0),
            }
        logits = model(input_ids, mask, **kwargs)
        assert logits.shape == (3, NUM_MOD_DIGITS, 10)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_shuffled_context_requires_explicit_nonself_context():
    model = RecurrentWorkspaceModel(d_model=16, n_heads=2, d_ff=32, workspace_init_mode="shuffled_context")
    ids = torch.randint(0, VOCAB_SIZE, (2, 5))
    try:
        model(ids)
    except ValueError as error:
        assert "init_input_ids" in str(error)
    else:
        raise AssertionError("shuffled context must not silently use its own input")


def test_recurrent_workspace_rejects_output_slots_larger_than_workspace():
    try:
        RecurrentWorkspaceModel(workspace_size=4, num_output_slots=8)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when num_output_slots > workspace_size")


def test_recurrent_workspace_override_loops_changes_effective_depth():
    input_ids, _, mask = _batch(n=2)
    model = RecurrentWorkspaceModel(
        max_seq_len=input_ids.shape[1], d_model=16, n_heads=2, d_ff=32,
        workspace_size=W, num_output_slots=W, num_loops=5,
    )
    out_full = model(input_ids, mask, override_loops=5)
    out_partial = model(input_ids, mask, override_loops=2)
    assert not torch.allclose(out_full, out_partial)


def test_tiny_dataset_overfit_near_100_percent_transformer():
    xs = list(range(1, 17))
    input_ids, labels, mask = _batch(n=len(xs), x_values=tuple(xs))
    model = StandardTransformer(max_seq_len=input_ids.shape[1], d_model=64, n_layers=2, n_heads=2, d_ff=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    targets = labels[:, -W:]
    for _ in range(400):
        logits = model(input_ids, mask)[:, -W:, :]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = model(input_ids, mask)[:, -W:, :].argmax(dim=-1)
        exact = (preds == targets).all(dim=-1).float().mean().item()
    assert exact >= 0.9, f"expected a 16-example dataset to overfit near 100%, got {exact:.2f}"
