"""
Generate minimal Keras model fixtures for every rule.

Run once (or when fixtures are stale):
    python tests/fixtures/create_fixtures.py

Each rule gets:
    tests/fixtures/<RULE_ID>/before.h5
    tests/fixtures/<RULE_ID>/after.h5

The models contain only the minimum layers needed to exercise the rule in
isolation.  Default input shape: (1, 32, 32, C) for spatial rules;
(1, N) for Dense rules.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path when run directly.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tensorflow as tf

FIXTURES_DIR = Path(__file__).resolve().parent
SEED = 42


def _save(model: tf.keras.Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def _conv_block(filters: int, kernel: int = 3, activation=None, **kw) -> tf.keras.layers.Conv2D:
    return tf.keras.layers.Conv2D(
        filters, kernel, padding="same", activation=activation, use_bias=True, **kw
    )


# ---------------------------------------------------------------------------
# RULE-L01 — Channel alignment
# ---------------------------------------------------------------------------

def make_L01():
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(10, name="conv1")(inp)          # 10 not multiple of 16
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    before = tf.keras.Model(inp, out, name="L01_before")

    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)          # aligned to 16
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    after = tf.keras.Model(inp, out, name="L01_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-L02 — Per-layer INT8 quantization (structural: float vs quant-annotated)
# ---------------------------------------------------------------------------

def make_L02():
    # Both fixtures are identical architecturally; the "after" represents the
    # target dtype intent — tests assert the condition (float weights) on before.
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    out = tf.keras.layers.Dense(10, activation="softmax", name="output")(x)
    before = tf.keras.Model(inp, out, name="L02_before")

    # After: same architecture, weights zeroed to simulate post-PTQ state.
    after = tf.keras.models.clone_model(before)
    after._name = "L02_after"
    for layer in after.layers:
        w = layer.get_weights()
        if w:
            layer.set_weights([np.zeros_like(wt) for wt in w])
    return before, after


# ---------------------------------------------------------------------------
# RULE-L03 — Magnitude pruning
# ---------------------------------------------------------------------------

def make_L03():
    tf.random.set_seed(SEED)
    inp = tf.keras.Input(shape=(64,), batch_size=1, name="input")
    out = tf.keras.layers.Dense(64, activation="relu", name="dense1")(inp)
    before = tf.keras.Model(inp, out, name="L03_before")
    before.build((1, 64))
    before(tf.zeros((1, 64)))   # build weights

    after = tf.keras.models.clone_model(before)
    after._name = "L03_after"
    after(tf.zeros((1, 64)))
    # Set weights: 70% zeros to represent post-pruning state.
    for old_l, new_l in zip(before.layers, after.layers):
        ws = old_l.get_weights()
        if not ws:
            continue
        pruned = []
        for w in ws:
            mask = np.abs(w) < np.percentile(np.abs(w), 70)
            pruned.append(np.where(mask, 0.0, w))
        new_l.set_weights(pruned)
    return before, after


# ---------------------------------------------------------------------------
# RULE-L04 — K-means clustering
# ---------------------------------------------------------------------------

def make_L04():
    tf.random.set_seed(SEED)
    inp = tf.keras.Input(shape=(64,), batch_size=1, name="input")
    out = tf.keras.layers.Dense(64, activation="relu", name="dense1")(inp)
    before = tf.keras.Model(inp, out, name="L04_before")
    before(tf.zeros((1, 64)))

    after = tf.keras.models.clone_model(before)
    after._name = "L04_after"
    after(tf.zeros((1, 64)))
    # Reduce unique weight values to k=32 to represent post-clustering state.
    for old_l, new_l in zip(before.layers, after.layers):
        ws = old_l.get_weights()
        if not ws:
            continue
        clustered = []
        for w in ws:
            flat = w.flatten()
            quantized = np.round(flat * 16) / 16   # simple 4-bit grid approximation
            clustered.append(quantized.reshape(w.shape).astype(w.dtype))
        new_l.set_weights(clustered)
    return before, after


# ---------------------------------------------------------------------------
# RULE-L05 — Unsupported op replacement
# ---------------------------------------------------------------------------

def make_L05():
    # Before: model with an unsupported Lambda layer (not in Ethos-U65 op set).
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)
    x = tf.keras.layers.Lambda(lambda t: t * 1.0, name="unsupported_lambda")(x)
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    before = tf.keras.Model(inp, out, name="L05_before")

    # After: Lambda replaced by supported identity (Activation linear).
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)
    x = tf.keras.layers.Activation("linear", name="supported_activation")(x)
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    after = tf.keras.Model(inp, out, name="L05_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-L06 — Activation substitution
# ---------------------------------------------------------------------------

def make_L06():
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, activation="gelu", name="conv1")(inp)
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    before = tf.keras.Model(inp, out, name="L06_before")

    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, activation="relu6", name="conv1")(inp)
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    after = tf.keras.Model(inp, out, name="L06_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-L07 — Depthwise factorization
# ---------------------------------------------------------------------------

def make_L07():
    # Two leading Conv2D (to be skipped by factorize_depthwise) + one eligible.
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)
    x = _conv_block(16, name="conv2")(x)
    x = _conv_block(32, name="conv3")(x)     # eligible: 3rd conv, 3x3
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    before = tf.keras.Model(inp, out, name="L07_before")

    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)
    x = _conv_block(16, name="conv2")(x)
    x = tf.keras.layers.SeparableConv2D(32, 3, padding="same", name="conv3")(x)
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    after = tf.keras.Model(inp, out, name="L07_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-C01 — BN foldability
# ---------------------------------------------------------------------------

def make_C01():
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)
    x = tf.keras.layers.BatchNormalization(trainable=True, name="bn1")(x)   # trainable BN
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    before = tf.keras.Model(inp, out, name="C01_before")

    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)
    bn = tf.keras.layers.BatchNormalization(trainable=False, name="bn1")    # frozen BN
    x = bn(x)
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    after = tf.keras.Model(inp, out, name="C01_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-C02 — Conv + BN + ReLU fusion
# ---------------------------------------------------------------------------

def make_C02():
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    before = tf.keras.Model(inp, out, name="C02_before")

    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, activation="relu", name="conv1")(inp)               # BN folded + relu fused
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    after = tf.keras.Model(inp, out, name="C02_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-C03 — Residual Add channel alignment
# ---------------------------------------------------------------------------

def make_C03():
    # Before: Add with mismatched shapes (requires projection).
    inp = tf.keras.Input(shape=(32, 32, 16), batch_size=1, name="input")
    branch_a = _conv_block(16, 1, name="proj_a")(inp)
    branch_b = _conv_block(16, 1, name="proj_b")(inp)
    out = tf.keras.layers.Add(name="add1")([branch_a, branch_b])
    before = tf.keras.Model(inp, out, name="C03_before")

    # After: both branches produce channel-aligned (multiple of 16) tensors.
    inp = tf.keras.Input(shape=(32, 32, 16), batch_size=1, name="input")
    branch_a = _conv_block(16, 1, name="proj_a")(inp)
    branch_b = _conv_block(16, 1, name="proj_b")(inp)
    out = tf.keras.layers.Add(name="add1")([branch_a, branch_b])
    after = tf.keras.Model(inp, out, name="C03_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-C04 — Dense stack merge
# ---------------------------------------------------------------------------

def make_C04():
    inp = tf.keras.Input(shape=(64,), batch_size=1, name="input")
    x = tf.keras.layers.Dense(32, activation=None, name="dense1")(inp)      # linear activation
    out = tf.keras.layers.Dense(16, activation="relu", name="dense2")(x)
    before = tf.keras.Model(inp, out, name="C04_before")

    inp = tf.keras.Input(shape=(64,), batch_size=1, name="input")
    out = tf.keras.layers.Dense(16, activation="relu", name="dense_merged")(inp)  # merged
    after = tf.keras.Model(inp, out, name="C04_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-C05 — Fused LSTM operator
# ---------------------------------------------------------------------------

def make_C05():
    # Before: LSTM expressed as decomposed ops (approximated as Dense stack here
    # since Keras LSTM is already fused; a real decomposed cell would use Dense
    # + element-wise layers).
    inp = tf.keras.Input(shape=(8, 16), batch_size=1, name="input")
    x = tf.keras.layers.Dense(16, name="input_gate")(inp)
    x = tf.keras.layers.Activation("sigmoid", name="sigmoid_gate")(x)
    out = tf.keras.layers.Dense(8, name="output_gate")(x)
    before = tf.keras.Model(inp, out, name="C05_before")

    # After: single fused LSTM.
    inp = tf.keras.Input(shape=(8, 16), batch_size=1, name="input")
    out = tf.keras.layers.LSTM(8, name="lstm1")(inp)
    out = tf.keras.layers.Reshape((1, 8), name="reshape")(out)
    # Trim to (1, 8) to align output rank for comparison purposes.
    after = tf.keras.Model(inp, out, name="C05_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-M01 — Global INT8 (same as L02 at model scope)
# ---------------------------------------------------------------------------

def make_M01():
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    out = tf.keras.layers.Dense(10, activation="softmax", name="output")(x)
    before = tf.keras.Model(inp, out, name="M01_before")

    after = tf.keras.models.clone_model(before)
    after._name = "M01_after"
    return before, after


# ---------------------------------------------------------------------------
# RULE-M02 — Attention block removal
# ---------------------------------------------------------------------------

def make_M02():
    inp = tf.keras.Input(shape=(8, 16), batch_size=1, name="input")
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=8, name="mha")(inp, inp)
    out = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    before = tf.keras.Model(inp, out, name="M02_before")

    inp = tf.keras.Input(shape=(8, 16), batch_size=1, name="input")
    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", name="dw_approx")(
        tf.keras.layers.Reshape((8, 16, 1), name="reshape")(inp)
    )
    x = tf.keras.layers.Reshape((8, 16), name="reshape_back")(x)
    out = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    after = tf.keras.Model(inp, out, name="M02_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-M03 — Global sparsity (same pattern as L03 at model scope)
# ---------------------------------------------------------------------------

def make_M03():
    tf.random.set_seed(SEED)
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    out = tf.keras.layers.Dense(10, name="output")(x)
    before = tf.keras.Model(inp, out, name="M03_before")
    before(tf.zeros((1, 32, 32, 3)))

    after = tf.keras.models.clone_model(before)
    after._name = "M03_after"
    after(tf.zeros((1, 32, 32, 3)))
    for old_l, new_l in zip(before.layers, after.layers):
        ws = old_l.get_weights()
        if not ws:
            continue
        pruned = []
        for w in ws:
            mask = np.abs(w) < np.percentile(np.abs(w), 75)
            pruned.append(np.where(mask, 0.0, w))
        new_l.set_weights(pruned)
    return before, after


# ---------------------------------------------------------------------------
# RULE-M04 — First conv output channel seeding
# ---------------------------------------------------------------------------

def make_M04():
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(15, name="conv1")(inp)     # 15 not multiple of 16
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    before = tf.keras.Model(inp, out, name="M04_before")

    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)     # aligned to 16
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    after = tf.keras.Model(inp, out, name="M04_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-M05 — Batch size fix
# ---------------------------------------------------------------------------

def make_M05():
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=None, name="input")
    x = _conv_block(16, name="conv1")(inp)
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    before = tf.keras.Model(inp, out, name="M05_before")

    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, name="conv1")(inp)
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    after = tf.keras.Model(inp, out, name="M05_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-M06 — Activation fusion
# ---------------------------------------------------------------------------

def make_M06():
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, activation=None, name="conv1")(inp)   # no activation
    x = tf.keras.layers.Activation("relu", name="relu1")(x)   # standalone
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    before = tf.keras.Model(inp, out, name="M06_before")

    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = _conv_block(16, activation="relu", name="conv1")(inp)  # fused
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    after = tf.keras.Model(inp, out, name="M06_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-M06.1 — Fold MaxPool into strided conv
# ---------------------------------------------------------------------------

def make_M06_1():
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = tf.keras.layers.Conv2D(16, 3, strides=(1, 1), padding="same", name="conv1")(inp)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1")(x)
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    before = tf.keras.Model(inp, out, name="M06_1_before")

    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=1, name="input")
    x = tf.keras.layers.Conv2D(16, 3, strides=(2, 2), padding="same", name="conv1")(inp)
    out = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    after = tf.keras.Model(inp, out, name="M06_1_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-M06.2 — Bilinear Resizing → Conv2DTranspose
# ---------------------------------------------------------------------------

def make_M06_2():
    inp = tf.keras.Input(shape=(8, 8, 16), batch_size=1, name="input")
    x = tf.keras.layers.Resizing(16, 16, interpolation="bilinear", name="resize1")(inp)
    out = _conv_block(16, 1, name="conv1")(x)
    before = tf.keras.Model(inp, out, name="M06_2_before")

    inp = tf.keras.Input(shape=(8, 8, 16), batch_size=1, name="input")
    x = tf.keras.layers.Conv2DTranspose(
        16, kernel_size=4, strides=2, padding="same", use_bias=False, name="resize1"
    )(inp)
    out = _conv_block(16, 1, name="conv1")(x)
    after = tf.keras.Model(inp, out, name="M06_2_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-M06.3 — Split + Concat collapse
# ---------------------------------------------------------------------------

def make_M06_3():
    inp = tf.keras.Input(shape=(32, 32, 16), batch_size=1, name="input")
    split_out = tf.keras.layers.Lambda(
        lambda t: tf.split(t, 2, axis=-1), name="split1"
    )(inp)
    branch = _conv_block(16, 1, name="work_branch")(inp)
    out = tf.keras.layers.Concatenate(axis=-1, name="concat1")(
        [split_out[0], split_out[1], branch]
    )
    before = tf.keras.Model(inp, out, name="M06_3_before")

    inp = tf.keras.Input(shape=(32, 32, 16), batch_size=1, name="input")
    branch = _conv_block(16, 1, name="work_branch")(inp)
    out = tf.keras.layers.Concatenate(axis=-1, name="concat1")([inp, branch])
    after = tf.keras.Model(inp, out, name="M06_3_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-M06.4 — Residual Add operand reorder
# ---------------------------------------------------------------------------

def make_M06_4():
    inp = tf.keras.Input(shape=(32, 32, 16), batch_size=1, name="input")
    shortcut = inp                                              # identity shortcut
    x = _conv_block(16, 1, name="conv1")(inp)
    out = tf.keras.layers.Add(name="add1")([shortcut, x])     # shortcut at [0]
    before = tf.keras.Model(inp, out, name="M06_4_before")

    inp = tf.keras.Input(shape=(32, 32, 16), batch_size=1, name="input")
    shortcut = inp
    x = _conv_block(16, 1, name="conv1")(inp)
    out = tf.keras.layers.Add(name="add1")([x, shortcut])     # conv at [0]
    after = tf.keras.Model(inp, out, name="M06_4_after")
    return before, after


# ---------------------------------------------------------------------------
# RULE-M06.5 — In-place concat (not yet implemented)
# ---------------------------------------------------------------------------

def make_M06_5():
    inp = tf.keras.Input(shape=(32, 32, 16), batch_size=1, name="input")
    a = _conv_block(16, 1, name="branch_a")(inp)
    b = _conv_block(16, 1, name="branch_b")(inp)
    out = tf.keras.layers.Concatenate(axis=-1, name="concat1")([a, b])
    before = tf.keras.Model(inp, out, name="M06_5_before")

    after = tf.keras.models.clone_model(before)
    after._name = "M06_5_after"
    return before, after


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAKERS = {
    "RULE-L01":   make_L01,
    "RULE-L02":   make_L02,
    "RULE-L03":   make_L03,
    "RULE-L04":   make_L04,
    "RULE-L05":   make_L05,
    "RULE-L06":   make_L06,
    "RULE-L07":   make_L07,
    "RULE-C01":   make_C01,
    "RULE-C02":   make_C02,
    "RULE-C03":   make_C03,
    "RULE-C04":   make_C04,
    "RULE-C05":   make_C05,
    "RULE-M01":   make_M01,
    "RULE-M02":   make_M02,
    "RULE-M03":   make_M03,
    "RULE-M04":   make_M04,
    "RULE-M05":   make_M05,
    "RULE-M06":   make_M06,
    "RULE-M06.1": make_M06_1,
    "RULE-M06.2": make_M06_2,
    "RULE-M06.3": make_M06_3,
    "RULE-M06.4": make_M06_4,
    "RULE-M06.5": make_M06_5,
}


def create_all(force: bool = False) -> None:
    tf.random.set_seed(SEED)
    for rule_id, maker in MAKERS.items():
        rule_dir = FIXTURES_DIR / rule_id
        before_path = rule_dir / "before.h5"
        after_path  = rule_dir / "after.h5"
        if before_path.exists() and after_path.exists() and not force:
            print(f"  skip  {rule_id}  (already exists)")
            continue
        print(f"  build {rule_id} ...", end=" ", flush=True)
        try:
            before, after = maker()
            _save(before, before_path)
            _save(after,  after_path)
            print("ok")
        except Exception as exc:
            print(f"FAILED: {exc}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing fixtures")
    args = parser.parse_args()
    print(f"Writing fixtures to {FIXTURES_DIR}")
    create_all(force=args.force)
    print("Done.")
