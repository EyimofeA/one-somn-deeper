from data import tokens as tok


def test_digit_tokens_roundtrip():
    ids = tok.digit_tokens(42, 4)
    assert len(ids) == 4
    digits = [i - tok.DIGIT_OFFSET for i in ids]
    assert digits == [0, 0, 4, 2]


def test_digit_tokens_reverse():
    ids = tok.digit_tokens(42, 4, reverse=True)
    digits = [i - tok.DIGIT_OFFSET for i in ids]
    assert digits == [2, 4, 0, 0]


def test_digit_tokens_overflow_raises():
    try:
        tok.digit_tokens(12345, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a value wider than its digit budget")


def test_encode_square_shapes_and_label():
    input_ids, labels = tok.encode_square(37)
    assert len(input_ids) == len(labels)
    assert input_ids[0] == tok.SQUARE and input_ids[1] == tok.X_MARKER
    out_labels = labels[-tok.NUM_SQUARE_DIGITS :]
    assert all(l != tok.IGNORE_INDEX for l in out_labels)
    assert all(l == tok.IGNORE_INDEX for l in labels[: -tok.NUM_SQUARE_DIGITS])
    value = int("".join(str(d) for d in out_labels))
    assert value == 37 * 37


def test_encode_mod_matches_python_mod():
    n, u = 323, 1764
    input_ids, labels = tok.encode_mod(n, u)
    out_labels = labels[-tok.NUM_MOD_DIGITS :]
    value = int("".join(str(d) for d in out_labels))
    assert value == u % n


def test_encode_square_mod_matches_python():
    n, x = 323, 42
    input_ids, labels = tok.encode_square_mod(n, x)
    out_labels = labels[-tok.NUM_MOD_DIGITS :]
    value = int("".join(str(d) for d in out_labels))
    assert value == (x * x) % n


def test_encode_square_mod_trace_two_heads():
    n, x = 323, 42
    input_ids, labels = tok.encode_square_mod_trace(n, x)
    aux = labels[-(tok.NUM_SQUARE_DIGITS + tok.NUM_MOD_DIGITS) : -tok.NUM_MOD_DIGITS]
    main = labels[-tok.NUM_MOD_DIGITS :]
    assert int("".join(str(d) for d in aux)) == x * x
    assert int("".join(str(d) for d in main)) == (x * x) % n


def test_no_causal_leakage_out_tokens_are_uniform():
    # every appended <OUT> slot must be the same token id regardless of value,
    # i.e. the model cannot read the answer off the input sequence itself.
    input_ids, _ = tok.encode_square(1)
    out_slots = input_ids[-tok.NUM_SQUARE_DIGITS :]
    assert all(t == tok.OUT for t in out_slots)
