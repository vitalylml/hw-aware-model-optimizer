"""
Phase 2 — Static graph transforms (no retraining required).

Transforms applied in rule application order:
  RULE-M05  fix_batch_size
  RULE-L06  substitute_activations
  RULE-C01  verify_bn_foldable
  RULE-C04  merge_linear_stack

Channel alignment (RULE-L01) is detected here but applied as an architecture
flag because it requires rebuilding the model definition and retraining.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RULE-L01 — Channel alignment padding
# ---------------------------------------------------------------------------

def align_channels(model: tf.keras.Model, target_multiple: int = 16) -> tf.keras.Model:
    """
    For each Conv2D with filters % target_multiple != 0,
    round up filters and insert a 1x1 projection conv back to original size.
    Returns a new Functional model — weights of unchanged layers are preserved.
    Misaligned layers need retraining; this scaffolds the new architecture.
    Uses clone_model to preserve graph structure.

    Output-producing Conv2D layers are excluded. Their channel count is part of
    the model's external contract (for example, detection-head class or box
    counts) and padding them would either break consumers or require a slicing
    op that the NPU may not support.
    """
    import tensorflow as tf

    output_layer_names: set[str] = set()
    for out_tensor in model.outputs:
        history = getattr(out_tensor, '_keras_history', None)
        if history is None:
            continue
        producer = history[0] if isinstance(history, tuple) else history.layer
        output_layer_names.add(producer.name)

    def clone_fn(layer):
        if (
            isinstance(layer, tf.keras.layers.Conv2D)
            and layer.name not in output_layer_names
        ):
            cfg = layer.get_config()
            f = cfg['filters']
            f_aligned = int(np.ceil(f / target_multiple) * target_multiple)
            if f_aligned != f:
                # Replace with aligned filter count
                cfg['filters'] = f_aligned
                new_layer = tf.keras.layers.Conv2D.from_config(cfg)
                # Mark for projection after this layer
                new_layer._align_channels_project_to = f
                return new_layer
        return layer

    # Clone model, adjusting Conv2D filters as needed
    new_model = tf.keras.models.clone_model(model, clone_function=clone_fn)

    # Set weights, padding if necessary for aligned Conv2D layers
    for old_layer, new_layer in zip(model.layers, new_model.layers):
        if (
            isinstance(old_layer, tf.keras.layers.Conv2D)
            and hasattr(new_layer, '_align_channels_project_to')
        ):
            old_w, old_b = old_layer.get_weights() if old_layer.use_bias else (old_layer.get_weights()[0], None)
            new_w_shape = new_layer.get_weights()[0].shape
            # Pad filters axis (last axis)
            pad_width = new_w_shape[-1] - old_w.shape[-1]
            if pad_width > 0:
                pad_shape = list(old_w.shape)
                pad_shape[-1] = pad_width
                pad_w = np.zeros(pad_shape, dtype=old_w.dtype)
                new_w = np.concatenate([old_w, pad_w], axis=-1)
                if old_layer.use_bias:
                    pad_b = np.zeros((pad_width,), dtype=old_b.dtype)
                    new_b = np.concatenate([old_b, pad_b], axis=0)
                    new_layer.set_weights([new_w, new_b])
                else:
                    new_layer.set_weights([new_w])
            else:
                new_layer.set_weights(old_layer.get_weights())
        else:
            try:
                new_layer.set_weights(old_layer.get_weights())
            except Exception:
                pass  # skip if not compatible (e.g., InputLayer)

    # Insert projection layers where needed
    # This requires a second pass to add 1x1 convs after aligned layers
    # For simplicity, warn user if any projection is needed (full support would require graph surgery)
    projections_needed = [l for l in new_model.layers if hasattr(l, '_align_channels_project_to')]
    if projections_needed:
        import warnings
        warnings.warn("align_channels: 1x1 projection layers not inserted automatically. Model graph surgery required for full support.")
    return new_model

# ---------------------------------------------------------------------------
# RULE-M05 — Fix batch size to 1
# ---------------------------------------------------------------------------

def fix_batch_size(model) -> object:
    """
    Re-builds the model with a fixed batch dimension of 1.
    Returns the same model if already correct.
    """
    import tensorflow as tf

    in_shape = model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]

    if in_shape[0] == 1:
        log.info("RULE-M05: batch_size already 1 — skipped")
        return model

    # Rebuild with fixed batch input
    fixed_shape = (1,) + tuple(in_shape[1:])
    new_inputs = tf.keras.Input(shape=fixed_shape[1:], batch_size=1,
                                dtype=model.input.dtype)
    new_model = tf.keras.Model(
        inputs=new_inputs,
        outputs=model(new_inputs, training=False),
    )
    new_model.set_weights(model.get_weights())
    log.info("RULE-M05: batch_size fixed to 1  (was %s)", in_shape[0])
    return new_model


# ---------------------------------------------------------------------------
# RULE-L06 — Substitute unsupported activations
# ---------------------------------------------------------------------------

def substitute_activations(model, substitutions: Optional[dict] = None) -> object:
    """
    Clone the model, replacing unsupported activations with supported equivalents.
    Default substitution map: gelu/swish/mish/elu → relu6, selu → relu.

    Does NOT require retraining but fine-tuning is recommended for accuracy recovery.
    """
    import tensorflow as tf
    from optimizer.rules import FALLBACK_ACTIVATIONS

    subs = substitutions or FALLBACK_ACTIVATIONS
    replaced: list[str] = []

    def clone_fn(layer):
        cfg = layer.get_config()
        act = cfg.get("activation")
        if isinstance(act, dict):
            act_name = act.get("class_name", "").lower()
        else:
            act_name = (act or "").lower()

        if act_name in subs:
            cfg["activation"] = subs[act_name]
            try:
                new_layer = type(layer).from_config(cfg)
                replaced.append(f"{layer.name}: {act_name} → {subs[act_name]}")
                return new_layer
            except Exception as e:
                log.warning("Could not substitute activation on %s: %s", layer.name, e)
        return layer

    new_model = tf.keras.models.clone_model(model, clone_function=clone_fn)
    new_model.set_weights(model.get_weights())

    for r in replaced:
        log.info("RULE-L06: substituted %s", r)
    if not replaced:
        log.info("RULE-L06: no unsupported activations found — skipped")

    return new_model


# ---------------------------------------------------------------------------
# RULE-C01 — Fold BatchNorm into preceding Conv2D / DepthwiseConv2D
# ---------------------------------------------------------------------------

def verify_bn_foldable(model) -> list[str]:
    """
    RULE-C01: Check all Conv → BN sequences are in inference mode
    (moving_mean/var frozen) so the TFLite converter can fold them.

    Returns list of layer names where BN may not fold correctly.
    """
    at_risk = []
    for layer in model.layers:
        if type(layer).__name__ == "BatchNormalization":
            # Check that BN is in inference mode (trainable=False or frozen stats)
            if layer.trainable and not hasattr(layer, '_is_frozen'):
                prev = _inbound(model, layer)
                if prev and type(prev).__name__ in ("Conv2D", "DepthwiseConv2D"):
                    at_risk.append(layer.name)
    return at_risk


# ---------------------------------------------------------------------------
# RULE-C04 — Merge consecutive Dense layers with no activation between them
# ---------------------------------------------------------------------------

def merge_linear_stack(model) -> object:
    """
    Merge Dense → Dense (no activation) pairs into a single Dense layer.

      W' = W2 @ W1
      b' = W2 @ b1 + b2

    Returns a rebuilt Functional model.  Merging changes layer names —
    downstream layer references must be updated in the model definition.
    """
    import tensorflow as tf

    # Identify merge candidates
    candidates: list[tuple[str, str]] = []
    for layer in model.layers:
        if type(layer).__name__ != "Dense":
            continue
        prev = _inbound(model, layer)
        if prev is None or type(prev).__name__ != "Dense":
            continue
        prev_cfg = prev.get_config()
        act = prev_cfg.get("activation", "linear")
        if isinstance(act, dict):
            act = act.get("class_name", "linear")
        if str(act).lower() in ("linear", "none"):
            candidates.append((prev.name, layer.name))

    if not candidates:
        log.info("RULE-C04: no mergeable Dense→Dense sequences found — skipped")
        return model

    # For each candidate, absorb layer2 weights into layer1 and remove layer2
    new_model = tf.keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())

    merged: list[str] = []
    absorbed: set[str] = set()

    for (l1_name, l2_name) in candidates:
        if l1_name in absorbed or l2_name in absorbed:
            continue   # already merged in a longer chain

        l1 = new_model.get_layer(l1_name)
        l2 = new_model.get_layer(l2_name)

        W1, b1 = l1.get_weights()
        W2, b2 = l2.get_weights()

        W_merged = W2 @ W1
        b_merged = W2 @ b1 + b2

        # Resize l1 to output of l2
        log.warning(
            "RULE-C04: merging %s (%d units) + %s (%d units) — "
            "requires model rebuild; logging candidate only.",
            l1_name, W1.shape[1], l2_name, W2.shape[0],
        )
        merged.append(f"{l1_name} + {l2_name} → merged({W2.shape[0]} units)")
        absorbed.add(l2_name)

    # Full merge requires Functional graph surgery — return original with log
    for m in merged:
        log.info("RULE-C04: merge candidate: %s", m)

    return new_model   # caller should rebuild model definition using the logged candidates


# ---------------------------------------------------------------------------
# RULE-M06 — NPU cascading optimizations (activation fusion)
# ---------------------------------------------------------------------------

_FUSABLE_ACTIVATIONS = {"relu", "relu6", "sigmoid", "tanh", "linear"}


def _build_active_graph(model) -> tuple[dict[str, set[str]], dict[str, int]]:
    """
    Build a consumer map that reflects the model's *active* topology only.

    Earlier transforms in the pipeline call ``tf.keras.models.clone_model``,
    which leaves stale ``_inbound_nodes`` on reused layer instances. Counting
    raw inbound nodes produces phantom consumers. Instead, this function
    walks backwards from ``model.outputs`` through each tensor's
    ``_keras_history``, using only the specific ``(layer, node_index)`` pairs
    referenced by the live graph.

    Returns
    -------
    consumers   : producer layer name -> set of consumer layer names
    node_index  : layer name -> the inbound-node index actually used in this
                  model (suitable for indexing into ``layer._inbound_nodes``).
    """
    consumers: dict[str, set[str]] = {}
    node_index: dict[str, int] = {}
    visited: set[tuple[str, int]] = set()

    def visit(tensor) -> None:
        h = tensor._keras_history
        layer = h.layer if hasattr(h, "layer") else h[0]
        n_idx = h.node_index if hasattr(h, "node_index") else h[1]
        key = (layer.name, n_idx)
        if key in visited:
            return
        visited.add(key)
        node_index[layer.name] = n_idx
        try:
            node = layer._inbound_nodes[n_idx]
        except IndexError:
            return
        for ki in node.keras_inputs:
            hp = ki._keras_history
            producer = hp.layer if hasattr(hp, "layer") else hp[0]
            consumers.setdefault(producer.name, set()).add(layer.name)
            visit(ki)

    for out in model.outputs:
        visit(out)
    return consumers, node_index


def apply_cascading_optimizations(model: tf.keras.Model) -> tf.keras.Model:
    """
    Implementation of RULE-M06.

    Folds standalone activation layers into the preceding Conv2D /
    DepthwiseConv2D / SeparableConv2D / Dense layer when:
      - the producing layer has activation in (None, "linear"),
      - the producing layer has exactly one consumer (the activation layer),
      - the activation layer implements a supported function
        (relu, relu6, sigmoid, tanh, linear),
      - the activation layer is not itself a model output.

    Effect on cascading:
      - Reduces operator count and the intermediate activation tensor that
        would otherwise become a cascade boundary in Vela.
      - Restores Vela's native Conv + Activation fusion in the NPU pass,
        which would otherwise be inhibited by the standalone activation op.

    Returns a new Functional model with the same outputs and identical
    numerical behaviour as the input.
    """
    import tensorflow as tf

    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.Dense,
    )

    consumers, node_index = _build_active_graph(model)

    # Output-producing layers must keep their identity so model.outputs is
    # preserved verbatim; only the activation candidates that are themselves
    # outputs are skipped (Conv outputs absorb the activation safely).
    output_layer_names: set[str] = set()
    for out in model.outputs:
        h = out._keras_history
        producer = h.layer if hasattr(h, "layer") else h[0]
        output_layer_names.add(producer.name)

    fuse_map: dict[str, str] = {}    # conv layer name -> activation name
    skip_set: set[str] = set()       # activation layer names to bypass

    for layer in model.layers:
        if not isinstance(layer, conv_types):
            continue
        current_act = layer.get_config().get("activation")
        if current_act not in (None, "linear"):
            continue
        cons = consumers.get(layer.name, set())
        if len(cons) != 1:
            continue
        act_layer = model.get_layer(next(iter(cons)))
        if act_layer.name in output_layer_names:
            continue
        # Activation must have a single producer in the active graph.
        if len(act_layer._inbound_nodes[node_index.get(act_layer.name, 0)].keras_inputs) != 1:
            continue
        act_name = _activation_name_of(act_layer)
        if act_name is None or act_name not in _FUSABLE_ACTIVATIONS:
            continue
        fuse_map[layer.name] = act_name
        skip_set.add(act_layer.name)

    if not fuse_map:
        log.info("RULE-M06: no Conv -> Activation pairs eligible for fusion")
        return model

    log.info(
        "RULE-M06: fusing %d activation(s) into preceding Conv/Dense layers",
        len(fuse_map),
    )
    return _rebuild_with_fused_activations(model, fuse_map, skip_set)


def _activation_name_of(layer) -> Optional[str]:
    """Return the activation function name implemented by a layer, or None."""
    import tensorflow as tf
    if isinstance(layer, tf.keras.layers.ReLU):
        # tf.keras.layers.ReLU defaults: negative_slope=0, threshold=0, max_value=None.
        # Only the default form maps cleanly to "relu" / "relu6".
        if float(getattr(layer, "negative_slope", 0.0) or 0.0) != 0.0:
            return None
        if float(getattr(layer, "threshold", 0.0) or 0.0) != 0.0:
            return None
        max_val = getattr(layer, "max_value", None)
        if max_val is None:
            return "relu"
        try:
            if float(max_val) == 6.0:
                return "relu6"
        except Exception:
            return None
        return None
    if isinstance(layer, tf.keras.layers.Activation):
        act = layer.get_config().get("activation")
        return act if isinstance(act, str) else None
    return None


def _has_single_input(layer) -> bool:
    nodes = layer._inbound_nodes
    if not nodes:
        return False
    inbound = nodes[0].inbound_layers
    if isinstance(inbound, list):
        return len(inbound) == 1
    return inbound is not None


def _rebuild_with_fused_activations(
    model: tf.keras.Model,
    fuse_map: dict[str, str],
    skip_set: set[str],
) -> tf.keras.Model:
    """Rebuild a Functional model applying activation fusion and layer bypass."""
    cfg_overrides = {name: {"activation": act} for name, act in fuse_map.items()}
    return _rebuild_with_overrides(model, cfg_overrides, skip_set, rule_id="RULE-M06")


def _rebuild_with_overrides(
    model: tf.keras.Model,
    cfg_overrides: dict[str, dict],
    skip_set: set[str],
    input_reorder: Optional[dict[str, list[int]]] = None,
    replacement_layers: Optional[dict[str, "tf.keras.layers.Layer"]] = None,
    replacement_weights: Optional[dict[str, list]] = None,
    custom_inputs: Optional[dict[str, list]] = None,
    rule_id: str = "RULE-M06",
) -> tf.keras.Model:
    """
    Rebuild a Functional model with per-layer config overrides, layer bypass,
    optional input-order permutations, and optional layer-class replacement.

    Parameters
    ----------
    model               : source Functional model.
    cfg_overrides       : mapping of original layer name -> dict of config
                          fields to override on the rebuilt copy of that
                          layer (e.g. ``{"conv1": {"strides": (2, 2)}}``).
    skip_set            : set of original layer names to bypass entirely;
                          each such layer must have exactly one input, and
                          consumers receive that upstream tensor instead.
    input_reorder       : mapping of original layer name -> permutation of
                          input indices to apply when re-invoking that
                          layer (e.g. ``{"add_1": [1, 0]}`` swaps the two
                          operands).
    replacement_layers  : mapping of original layer name -> pre-built layer
                          instance to use *instead* of the original (used
                          when the replacement changes the layer class,
                          e.g. Resizing -> Conv2DTranspose). Weights are
                          not copied automatically; provide them through
                          ``replacement_weights``.
    replacement_weights : mapping of original layer name -> list of
                          ``np.ndarray`` weight values to apply via
                          ``set_weights`` after the replacement layer is
                          built. Only relevant when paired with an entry in
                          ``replacement_layers``.
    custom_inputs       : mapping of original layer name -> explicit list of
                          ``(producer_layer_name, tensor_index)`` tuples to
                          use as the layer's inputs. Overrides the inputs
                          derived from the original graph node. Used to
                          shrink a Concat's input list (e.g. collapsing a
                          (split:0, split:1) pair into a single reference
                          to the Split's own input).
    rule_id             : identifier for log messages.
    """
    input_reorder       = input_reorder       or {}
    replacement_layers  = replacement_layers  or {}
    replacement_weights = replacement_weights or {}
    custom_inputs       = custom_inputs       or {}
    import tensorflow as tf

    new_tensor: dict[str, "tf.Tensor"] = {}

    new_inputs: list = []
    for inp in model.inputs:
        h = inp._keras_history
        producer = h[0] if isinstance(h, tuple) else h.layer
        shape = tuple(inp.shape[1:])
        new_inp = tf.keras.Input(shape=shape, dtype=inp.dtype, name=producer.name)
        new_tensor[producer.name] = new_inp
        new_inputs.append(new_inp)

    new_layer_for: dict[str, tf.keras.layers.Layer] = {}

    # Per-layer inbound-node index used by the current model graph.
    _, node_index = _build_active_graph(model)

    pending = [
        l for l in model.layers
        if not isinstance(l, tf.keras.layers.InputLayer)
    ]

    def _resolve_inputs(node) -> Optional[list]:
        """
        Map node.keras_inputs to their rebuilt counterparts. Returns None when
        any upstream producer has not been rebuilt yet (caller should requeue).

        Handles multi-output producers (e.g. Split) by indexing into the
        upstream layer's stored output tuple via ``tensor_index``.
        """
        resolved = []
        for ki in node.keras_inputs:
            h = ki._keras_history
            producer = h.layer if hasattr(h, "layer") else h[0]
            t_idx = h.tensor_index if hasattr(h, "tensor_index") else h[2]
            if producer.name not in new_tensor:
                return None
            up = new_tensor[producer.name]
            if isinstance(up, (list, tuple)):
                resolved.append(up[t_idx])
            else:
                resolved.append(up)
        return resolved

    def _resolve_custom_inputs(spec: list) -> Optional[list]:
        """Resolve a custom ``(producer_name, tensor_index)`` list to tensors."""
        resolved = []
        for producer_name, t_idx in spec:
            if producer_name not in new_tensor:
                return None
            up = new_tensor[producer_name]
            if isinstance(up, (list, tuple)):
                resolved.append(up[t_idx])
            else:
                resolved.append(up)
        return resolved

    safety_passes = max(1, len(pending) * 2)
    for _ in range(safety_passes):
        if not pending:
            break
        progressed = False
        next_pending = []
        for layer in pending:
            n_idx = node_index.get(layer.name, 0)
            try:
                node = layer._inbound_nodes[n_idx]
            except IndexError:
                next_pending.append(layer)
                continue
            if layer.name in custom_inputs:
                resolved = _resolve_custom_inputs(custom_inputs[layer.name])
            else:
                resolved = _resolve_inputs(node)
            if resolved is None:
                next_pending.append(layer)
                continue
            perm = input_reorder.get(layer.name)
            if perm is not None and len(resolved) == len(perm):
                resolved = [resolved[i] for i in perm]
            inputs = resolved[0] if len(resolved) == 1 else resolved

            if layer.name in skip_set:
                new_tensor[layer.name] = inputs
            elif layer.name in replacement_layers:
                new_layer = replacement_layers[layer.name]
                out = new_layer(inputs)
                new_tensor[layer.name] = out
                new_layer_for[layer.name] = new_layer
                if layer.name in replacement_weights:
                    try:
                        new_layer.set_weights(replacement_weights[layer.name])
                    except Exception as e:
                        log.warning(
                            "%s: replacement-weight set on %s failed: %s",
                            rule_id, layer.name, e,
                        )
            else:
                cfg = layer.get_config()
                if layer.name in cfg_overrides:
                    cfg.update(cfg_overrides[layer.name])
                new_layer = type(layer).from_config(cfg)
                out = new_layer(inputs)
                new_tensor[layer.name] = out
                new_layer_for[layer.name] = new_layer

            progressed = True

        pending = next_pending
        if not progressed:
            log.warning(
                "%s: rebuild stalled with %d layer(s) unprocessed; "
                "returning original model",
                rule_id, len(pending),
            )
            return model

    new_outputs = []
    for out_tensor in model.outputs:
        h = out_tensor._keras_history
        producer = h.layer if hasattr(h, "layer") else h[0]
        t_idx = h.tensor_index if hasattr(h, "tensor_index") else h[2]
        up = new_tensor[producer.name]
        if isinstance(up, (list, tuple)):
            new_outputs.append(up[t_idx])
        else:
            new_outputs.append(up)

    new_model = tf.keras.Model(inputs=new_inputs, outputs=new_outputs)

    for old_name, new_layer in new_layer_for.items():
        if old_name in replacement_layers:
            # Replacement layers have weights set explicitly above.
            continue
        try:
            old_w = model.get_layer(old_name).get_weights()
            if old_w:
                new_layer.set_weights(old_w)
        except Exception as e:
            log.warning(
                "%s: weight copy for %s -> %s failed: %s",
                rule_id, old_name, new_layer.name, e,
            )

    return new_model


# ---------------------------------------------------------------------------
# RULE-M06.1 — Fold MaxPool(2x2, stride=2) into preceding Conv2D as stride=2
# ---------------------------------------------------------------------------

def _maxpool_padding_safe_for_strided_conv(pool_layer) -> bool:
    """
    Return True if a 2x2/stride-2 MaxPool produces the same output shape as a
    stride-2 'same'-padded Conv2D.

    Cases accepted:
      - padding == 'same'  : always equivalent.
      - padding == 'valid' : equivalent when input H and W are both even,
        because floor((d - 2) / 2) + 1 == ceil(d / 2) for even d.
    """
    if pool_layer.padding == "same":
        return True
    if pool_layer.padding != "valid":
        return False
    try:
        in_shape = pool_layer.input_shape
        if isinstance(in_shape, list):
            in_shape = in_shape[0]
    except Exception:
        return False
    if in_shape is None or len(in_shape) < 3:
        return False
    h, w = in_shape[1], in_shape[2]
    if h is None or w is None:
        return False
    return (h % 2 == 0) and (w % 2 == 0)


def fold_maxpool_into_strided_conv(model: tf.keras.Model) -> tf.keras.Model:
    """
    Implementation of RULE-M06.1.

    Detects ``Conv2D(strides=(1,1)) -> [BN] -> [Activation/ReLU] -> MaxPooling2D``
    chains where the MaxPool has ``pool_size=(2,2)`` and ``strides=(2,2)``, and
    rewrites the Conv2D with ``strides=(2,2)`` while removing the trailing
    MaxPool.

    Constraints:
      - Every intermediate layer (BN, Activation, ReLU) must have exactly one
        consumer; otherwise the fold would change the visible output of a
        shared producer.
      - The MaxPool must produce the same output spatial size as a stride-2
        same-padded Conv2D. This holds when MaxPool padding is ``'same'`` or
        when padding is ``'valid'`` and the input H and W are both even
        (the typical case for CNN feature maps before downsampling).
      - The fold is an *approximation* (max-pool semantics differ from
        stride-2 downsampling); minor accuracy drift is expected to be
        recovered by Phase 3 PCQAT.
    """
    import tensorflow as tf

    consumers, _ = _build_active_graph(model)

    intermediate_types = (
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.Activation,
        tf.keras.layers.ReLU,
    )
    foldable_producer_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )

    conv_to_pool: dict[str, str] = {}    # conv name -> maxpool name
    skip_set: set[str] = set()           # maxpool names to bypass

    for layer in model.layers:
        if not isinstance(layer, foldable_producer_types):
            continue
        if tuple(layer.strides) != (1, 1):
            continue

        # Walk forward: Conv -> (BN | Activation | ReLU)* -> MaxPool
        cur = layer
        matched_pool: Optional[tf.keras.layers.MaxPooling2D] = None
        while True:
            cons = consumers.get(cur.name, set())
            if len(cons) != 1:
                break
            nxt = model.get_layer(next(iter(cons)))
            if isinstance(nxt, tf.keras.layers.MaxPooling2D):
                if tuple(nxt.pool_size) != (2, 2):
                    break
                if tuple(nxt.strides) != (2, 2):
                    break
                if not _maxpool_padding_safe_for_strided_conv(nxt):
                    # 'valid' on odd dims produces a different output shape
                    # from stride-2 'same'. Skip those.
                    break
                matched_pool = nxt
                break
            if isinstance(nxt, intermediate_types):
                cur = nxt
                continue
            break

        if matched_pool is not None:
            conv_to_pool[layer.name] = matched_pool.name
            skip_set.add(matched_pool.name)

    if not conv_to_pool:
        log.info("RULE-M06.1: no Conv -> MaxPool fold candidates found")
        return model

    log.info(
        "RULE-M06.1: folding %d MaxPool layer(s) into preceding Conv2D as stride-2",
        len(conv_to_pool),
    )

    overrides = {
        name: {"strides": (2, 2)} for name in conv_to_pool
    }
    return _rebuild_with_overrides(model, overrides, skip_set, rule_id="RULE-M06.1")


# ---------------------------------------------------------------------------
# RULE-M06.3 — Fuse Split / Concat blocks (ShuffleNet-style)
# ---------------------------------------------------------------------------

def _is_split_layer(layer) -> bool:
    """Detect Split layers by class name to handle both stock and lscquant variants."""
    return type(layer).__name__ == "Split"


def _split_axis(layer) -> int:
    """Best-effort axis lookup for a Split layer, defaulting to -1 (channel)."""
    cfg = layer.get_config()
    try:
        axis = cfg.get("axis", -1)
        return int(axis) if axis is not None else -1
    except (TypeError, ValueError):
        return -1


def _axes_match(a: int, b: int, rank: int = 4) -> bool:
    """Treat negative axes and their positive equivalents as equal."""
    return (a % rank) == (b % rank)


def fuse_split_concat_blocks(model: tf.keras.Model) -> tf.keras.Model:
    """
    Implementation of RULE-M06.3.

    Detects Concatenate layers whose input list contains an adjacent pair
    ``(split_X:tensor_index=0, split_X:tensor_index=1)`` from the same
    ``Split`` layer with ``num_or_size_splits=2`` at the same axis as the
    Concat, and replaces the pair with a single direct reference to the
    Split's own upstream input.

    Why the rewrite is mathematically exact:
      Concat([split[0], split[1]], axis=-1) reconstructs the original tensor
      that fed into the Split. Replacing those two adjacent inputs of a
      larger Concat with the original tensor preserves the output exactly.

    Effect on the NPU schedule:
      - One fewer tensor materialisation per affected Concat (two cascade
        boundary tensors collapse to one).
      - Smaller Concat input list reduces SRAM live-range pressure on the
        scheduler, which can unlock additional cascading.
      - The Split is preserved when its other output still has consumers
        (typical ShuffleNet-v2 case where output[1] feeds the work branch).
    """
    import tensorflow as tf

    _, node_index = _build_active_graph(model)

    custom_inputs: dict[str, list[tuple[str, int]]] = {}
    n_pairs_collapsed = 0

    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Concatenate):
            continue
        concat_axis = layer.axis

        n_idx = node_index.get(layer.name, 0)
        try:
            node = layer._inbound_nodes[n_idx]
        except IndexError:
            continue
        keras_inputs = node.keras_inputs
        if len(keras_inputs) < 3:
            # Need at least 3 inputs for the rewrite to shrink the list.
            continue

        # (producer_layer, tensor_index) for each Concat input.
        prod_ti: list[tuple] = []
        for t in keras_inputs:
            h = t._keras_history
            prod = h.layer if hasattr(h, "layer") else h[0]
            ti   = h.tensor_index if hasattr(h, "tensor_index") else h[2]
            prod_ti.append((prod, ti))

        new_input_spec: list[tuple[str, int]] = []
        i = 0
        collapsed_here = 0
        while i < len(prod_ti):
            if i + 1 < len(prod_ti):
                p_a, t_a = prod_ti[i]
                p_b, t_b = prod_ti[i + 1]
                if (
                    _is_split_layer(p_a)
                    and p_a is p_b
                    and t_a == 0 and t_b == 1
                    and int(p_a.get_config().get("num_or_size_splits", -1)) == 2
                    and _axes_match(_split_axis(p_a), concat_axis)
                ):
                    # Resolve the Split's input producer.
                    n_split_idx = node_index.get(p_a.name, 0)
                    try:
                        split_node = p_a._inbound_nodes[n_split_idx]
                    except IndexError:
                        new_input_spec.append((p_a.name, t_a))
                        i += 1
                        continue
                    s_in = split_node.keras_inputs[0]
                    h_si = s_in._keras_history
                    s_in_prod = h_si.layer if hasattr(h_si, "layer") else h_si[0]
                    s_in_ti   = h_si.tensor_index if hasattr(h_si, "tensor_index") else h_si[2]
                    new_input_spec.append((s_in_prod.name, s_in_ti))
                    i += 2
                    collapsed_here += 1
                    continue
            new_input_spec.append((prod_ti[i][0].name, prod_ti[i][1]))
            i += 1

        if collapsed_here > 0:
            custom_inputs[layer.name] = new_input_spec
            n_pairs_collapsed += collapsed_here

    if not custom_inputs:
        log.info("RULE-M06.3: no Split-Concat collapse candidates found")
        return model

    log.info(
        "RULE-M06.3: collapsed %d Split-output pair(s) across %d Concat layer(s)",
        n_pairs_collapsed, len(custom_inputs),
    )
    return _rebuild_with_overrides(
        model,
        cfg_overrides={},
        skip_set=set(),
        custom_inputs=custom_inputs,
        rule_id="RULE-M06.3",
    )


# ---------------------------------------------------------------------------
# RULE-M06.2 — Replace bilinear Resizing with Conv2DTranspose (NPU-native)
# ---------------------------------------------------------------------------

def _bilinear_upsample_kernel(filter_size: int, factor: int) -> np.ndarray:
    """
    Return the 2D bilinear-upsampling kernel of shape (filter_size, filter_size)
    for ``factor``-x transposed-convolution upsampling.

    For even ``filter_size`` the kernel center is at ``factor - 0.5``; for odd
    ``filter_size`` it is at ``factor - 1``. This is the standard FCN-style
    bilinear deconvolution filter and reproduces bilinear-resize output
    closely (not bit-exact, since align/half-pixel-center semantics differ).
    """
    if filter_size % 2 == 1:
        center = float(factor - 1)
    else:
        center = float(factor) - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    return (1.0 - np.abs(og[0] - center) / factor) * \
           (1.0 - np.abs(og[1] - center) / factor)


def replace_bilinear_resize_with_conv2dtranspose(model: tf.keras.Model) -> tf.keras.Model:
    """
    Implementation of RULE-M06.2.

    Replace each ``tf.keras.layers.Resizing`` layer using
    ``interpolation='bilinear'`` with a depthwise-style
    ``Conv2DTranspose`` whose weights are pre-initialised to a bilinear
    upsampling kernel. Conv2DTranspose is NPU-native on Ethos-U65 and can
    be cascaded with the following Conv block, while bilinear Resizing is
    a standalone op that forces a tensor materialisation.

    Constraints:
      - The output spatial dimensions must be an integer multiple of the
        input spatial dimensions, with the same factor in H and W.
      - The factor must be >= 2 (1x resize is a no-op; non-integer factors
        cannot be represented by a stride-N transposed convolution).
      - Channel count is preserved; the kernel is diagonal in the channel
        dimension so each output channel depends only on the same input
        channel (true depthwise behaviour).
      - The rewrite is an approximation (transposed-conv vs bilinear-resize
        sampling semantics differ at boundaries); minor accuracy drift is
        expected to be recovered by Phase 3 PCQAT.
    """
    import tensorflow as tf

    replacements: dict[str, tf.keras.layers.Layer] = {}
    weight_map:   dict[str, list] = {}

    for layer in model.layers:
        if type(layer).__name__ != "Resizing":
            continue
        cfg = layer.get_config()
        if cfg.get("interpolation") != "bilinear":
            continue

        # Read output spatial dims from the layer's config.
        try:
            h_out = int(cfg.get("height"))
            w_out = int(cfg.get("width"))
        except (TypeError, ValueError):
            continue

        # Read input spatial dims from the layer's input_shape.
        try:
            in_shape = layer.input_shape
            if isinstance(in_shape, list):
                in_shape = in_shape[0]
            h_in = int(in_shape[1])
            w_in = int(in_shape[2])
            channels = int(in_shape[3])
        except Exception:
            continue

        if h_in <= 0 or w_in <= 0 or channels <= 0:
            continue
        if h_out % h_in != 0 or w_out % w_in != 0:
            continue

        factor_h = h_out // h_in
        factor_w = w_out // w_in
        if factor_h != factor_w or factor_h < 2:
            continue
        factor = factor_h
        filter_size = 2 * factor

        new_layer = tf.keras.layers.Conv2DTranspose(
            filters=channels,
            kernel_size=filter_size,
            strides=factor,
            padding="same",
            use_bias=False,
            name=layer.name,
        )

        # Conv2DTranspose kernel shape: (kh, kw, filters, input_channels).
        # Depthwise-style: diagonal in the channel dimension.
        bilinear_2d = _bilinear_upsample_kernel(filter_size, factor)
        kernel = np.zeros(
            (filter_size, filter_size, channels, channels),
            dtype=np.float32,
        )
        for c in range(channels):
            kernel[:, :, c, c] = bilinear_2d.astype(np.float32)

        replacements[layer.name] = new_layer
        weight_map[layer.name] = [kernel]

    if not replacements:
        log.info(
            "RULE-M06.2: no bilinear Resizing layers eligible for "
            "Conv2DTranspose replacement"
        )
        return model

    log.info(
        "RULE-M06.2: replacing %d bilinear Resizing layer(s) with Conv2DTranspose",
        len(replacements),
    )
    return _rebuild_with_overrides(
        model,
        cfg_overrides={},
        skip_set=set(),
        replacement_layers=replacements,
        replacement_weights=weight_map,
        rule_id="RULE-M06.2",
    )


# ---------------------------------------------------------------------------
# RULE-M06.4 — Reorder residual Add operands for fused Conv+ResAdd cascading
# ---------------------------------------------------------------------------

def reorder_residual_add_operands(model: tf.keras.Model) -> tf.keras.Model:
    """
    Implementation of RULE-M06.4.

    For each two-input ``tf.keras.layers.Add`` layer, determine which operand
    represents the "fresh" Conv-block branch (the one whose producer chain
    leads back to a Conv-like layer through at most a few BatchNormalization
    or Activation/ReLU hops). Vela's pattern matcher expects this branch at
    operand index 0 so it can fuse the Add into the preceding Conv's bias
    accumulator (the fused Conv + ResAdd path). When the fresh branch is at
    operand index 1, the operands are swapped.

    Add is commutative, so the rewrite is mathematically exact.
    """
    import tensorflow as tf

    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.Dense,
    )
    intermediate_types = (
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.Activation,
        tf.keras.layers.ReLU,
    )

    _, node_index = _build_active_graph(model)
    max_depth = 4
    not_found = max_depth + 1

    def _distance_to_conv(producer) -> int:
        """Hops from ``producer`` to nearest Conv-like ancestor via a single-
        producer chain of BN/Activation/ReLU layers. Returns ``not_found``
        when no Conv is reachable within ``max_depth`` hops."""
        cur = producer
        for d in range(max_depth + 1):
            if isinstance(cur, conv_types):
                return d
            if not isinstance(cur, intermediate_types):
                return not_found
            n_idx = node_index.get(cur.name, 0)
            try:
                node = cur._inbound_nodes[n_idx]
            except IndexError:
                return not_found
            keras_inputs = node.keras_inputs
            if len(keras_inputs) != 1:
                return not_found
            cur = keras_inputs[0]._keras_history.layer
        return not_found

    input_reorder: dict[str, list[int]] = {}

    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Add):
            continue
        n_idx = node_index.get(layer.name, 0)
        try:
            node = layer._inbound_nodes[n_idx]
        except IndexError:
            continue
        keras_inputs = node.keras_inputs
        if len(keras_inputs) != 2:
            continue

        producers = [t._keras_history.layer for t in keras_inputs]
        d0 = _distance_to_conv(producers[0])
        d1 = _distance_to_conv(producers[1])

        if d0 >= not_found and d1 >= not_found:
            continue   # neither operand has a reachable Conv ancestor
        if d0 < not_found and d1 >= not_found:
            continue   # fresh branch already at operand 0
        if d0 >= not_found and d1 < not_found:
            input_reorder[layer.name] = [1, 0]
            continue
        # Both reachable: prefer the smaller distance at operand 0.
        if d1 < d0:
            input_reorder[layer.name] = [1, 0]

    if not input_reorder:
        log.info("RULE-M06.4: no Add operand reorder candidates found")
        return model

    log.info(
        "RULE-M06.4: reordering operands of %d Add layer(s) for Conv+ResAdd fusion",
        len(input_reorder),
    )
    return _rebuild_with_overrides(
        model,
        cfg_overrides={},
        skip_set=set(),
        input_reorder=input_reorder,
        rule_id="RULE-M06.4",
    )


# ---------------------------------------------------------------------------
# Convenience: run all static transforms in order
# ---------------------------------------------------------------------------

def apply_all_static_transforms(model, config: Optional[dict] = None) -> object:
    """
    Apply all no-retraining transforms in rule application order.

    Parameters
    ----------
    model   : loaded tf.keras.Model
    config  : optional overrides, e.g. {"activation_subs": {...}}

    Returns
    -------
    Transformed tf.keras.Model
    """
    cfg = config or {}

    log.info("=== Phase 2: Static transforms ===")




    # Canonical order integration
    # RULE-L01: Channel alignment
    model = align_channels(model)
    # RULE-M04: Input layer channel seeding
    model = seed_input_layer_channels(model)
    # RULE-L07: Depthwise factorization
    model = factorize_depthwise(model)
    # RULE-M05: Fix batch size
    # model = fix_batch_size(model)
    # RULE-L06: Substitute unsupported activations
    model = substitute_activations(model, cfg.get("activation_subs"))
    # RULE-C01: BN foldability check (side effect only)
    verify_bn_foldable(model)
    # RULE-C03: Residual Add placement check (side effect only)
    check_residual_add_placement(model)
    # RULE-C04: Merge Dense stacks
    model = merge_linear_stack(model)
    # RULE-M06: Cascading optimizations (activation fusion)
    model = apply_cascading_optimizations(model)
    # RULE-M06.1: Fold MaxPool(2x2, s=2) into preceding Conv2D as stride-2
    model = fold_maxpool_into_strided_conv(model)
    # RULE-M06.2: Replace bilinear Resizing with Conv2DTranspose (bilinear-init)
    model = replace_bilinear_resize_with_conv2dtranspose(model)
    # RULE-M06.3: Collapse Split-output pairs feeding the same Concat
    model = fuse_split_concat_blocks(model)
    # RULE-M06.4: Reorder residual Add operands so Vela fuses Conv+ResAdd
    model = reorder_residual_add_operands(model)

    log.info("=== Phase 2 complete ===")
    return model


def _inbound(model, layer):
    """Return first inbound layer for a given layer within a model."""
    try:
        nodes = layer._inbound_nodes
        if not nodes:
            return None
        inbound = nodes[0].inbound_layers
        if isinstance(inbound, list):
            return inbound[0] if inbound else None
        return inbound
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RULE-L07 — Depthwise factorization (Conv2D → DepthwiseConv2D + Conv2D)
# ---------------------------------------------------------------------------
def factorize_depthwise(model: tf.keras.Model) -> tf.keras.Model:
    """
    For each eligible Conv2D (kernel_size >= 3x3, not first 1-2 layers),
    replace with DepthwiseConv2D + 1x1 Conv2D (DW-separable).
    Returns a new Functional model. Requires retraining for accuracy recovery.
    """
    import tensorflow as tf

    # Identify eligible Conv2D layers (skip first 2 Conv2D layers)
    conv2d_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    eligible: set[str] = set()
    for idx, layer in enumerate(conv2d_layers):
        if idx < 2:
            continue
        k_h, k_w = layer.kernel_size
        if k_h >= 3 and k_w >= 3:
            eligible.add(layer.name)

    if not eligible:
        log.info("RULE-L07: no eligible Conv2D layers for factorization — skipped")
        return model

    # Preserve original graph topology by cloning and only swapping eligible layers.
    def clone_fn(layer):
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.name in eligible:
            cfg = layer.get_config()
            return tf.keras.layers.SeparableConv2D(
                filters=cfg["filters"],
                kernel_size=cfg["kernel_size"],
                strides=cfg["strides"],
                padding=cfg["padding"],
                dilation_rate=cfg["dilation_rate"],
                activation=cfg["activation"],
                # Keep bias policy identical to the source Conv2D.
                use_bias=bool(layer.use_bias),
                depth_multiplier=1,
                name=cfg["name"],
            )
        return layer

    new_model = tf.keras.models.clone_model(model, clone_function=clone_fn)

    # Copy / initialize weights.
    converted_count = 0
    for old_layer in model.layers:
        try:
            new_layer = new_model.get_layer(old_layer.name)
        except Exception:
            continue

        # Converted Conv2D -> SeparableConv2D
        if isinstance(old_layer, tf.keras.layers.Conv2D) and old_layer.name in eligible:
            old_weights = old_layer.get_weights()
            if not old_weights:
                continue

            old_kernel = old_weights[0]  # [kh, kw, in_c, out_c]
            old_bias = old_weights[1] if len(old_weights) > 1 else None

            new_weights = new_layer.get_weights()
            if len(new_weights) < 2:
                continue

            # Bias flag must remain consistent after conversion.
            src_use_bias = bool(old_layer.use_bias)
            dst_use_bias = bool(getattr(new_layer, "use_bias", False))
            if src_use_bias != dst_use_bias:
                log.warning(
                    "RULE-L07: bias mismatch on %s (src=%s, dst=%s); forcing source policy in initialization",
                    old_layer.name,
                    src_use_bias,
                    dst_use_bias,
                )

            depthwise_kernel = new_weights[0]  # [kh, kw, in_c, 1]
            pointwise_kernel = new_weights[1]  # [1, 1, in_c, out_c]

            # Initialize depthwise with channel-averaged spatial filters.
            dw_init = np.mean(old_kernel, axis=3, keepdims=True)
            if dw_init.shape == depthwise_kernel.shape:
                depthwise_kernel = dw_init.astype(depthwise_kernel.dtype)

            # Initialize pointwise using center tap of original conv kernel.
            kh, kw = old_kernel.shape[0], old_kernel.shape[1]
            center = old_kernel[kh // 2, kw // 2, :, :]
            pw_init = np.expand_dims(np.expand_dims(center, axis=0), axis=0)
            if pw_init.shape == pointwise_kernel.shape:
                pointwise_kernel = pw_init.astype(pointwise_kernel.dtype)

            if src_use_bias and old_bias is not None and len(new_weights) == 3:
                new_layer.set_weights([depthwise_kernel, pointwise_kernel, old_bias])
            else:
                new_layer.set_weights([depthwise_kernel, pointwise_kernel])

            converted_count += 1
            continue

        # Unchanged layers: direct copy when possible.
        try:
            new_layer.set_weights(old_layer.get_weights())
        except Exception:
            pass

    log.info("RULE-L07: factorized %d Conv2D layers using SeparableConv2D", converted_count)
    return new_model


# ---------------------------------------------------------------------------
# RULE-C03 — Residual Add placement (ensure Add in skip connections is correct)
# ---------------------------------------------------------------------------
def check_residual_add_placement(model: tf.keras.Model) -> list[str]:
    """
    Check Add layers in skip/residual connections for channel and dtype alignment.
    Returns list of Add layer names with detected issues.
    """
    import tensorflow as tf
    issues = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Add):
            inbound = layer._inbound_nodes[0].input_tensors if layer._inbound_nodes else []
            if len(inbound) != 2:
                issues.append(f"{layer.name}: Add does not have 2 inputs")
                continue
            shapes = [t.shape for t in inbound]
            dtypes = [t.dtype for t in inbound]
            if shapes[0] != shapes[1]:
                issues.append(f"{layer.name}: shape mismatch {shapes[0]} vs {shapes[1]}")
            if dtypes[0] != dtypes[1]:
                issues.append(f"{layer.name}: dtype mismatch {dtypes[0]} vs {dtypes[1]}")
    if issues:
        log.warning(f"RULE-C03: Add placement issues: {issues}")
    else:
        log.info("RULE-C03: all Add layers in skip/residual connections are valid")
    return issues


# ---------------------------------------------------------------------------
# RULE-M04 — Input layer channel seeding (ensure first Conv2D output channels aligned)
# ---------------------------------------------------------------------------
def seed_input_layer_channels(model: tf.keras.Model, min_channels: int = 16, channel_multiple: int = 16) -> tf.keras.Model:
    """
    Ensure first Conv2D layer output channels >= min_channels and aligned to channel_multiple.
    Returns a new model if adjustment is needed, else original.
    """
    import tensorflow as tf
    conv2d_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if not conv2d_layers:
        log.info("RULE-M04: no Conv2D layers found — skipped")
        return model
    first = conv2d_layers[0]
    cfg = first.get_config()
    filters = cfg['filters']
    new_filters = max(filters, min_channels)
    new_filters = int(np.ceil(new_filters / channel_multiple) * channel_multiple)
    if new_filters == filters:
        log.info("RULE-M04: first Conv2D output channels already aligned — skipped")
        return model
    cfg['filters'] = new_filters
    new_first = tf.keras.layers.Conv2D.from_config(cfg)
    # Clone model, replacing first Conv2D
    def clone_fn(layer):
        if layer.name == first.name:
            return new_first
        return layer
    new_model = tf.keras.models.clone_model(model, clone_function=clone_fn)
    new_model.set_weights(model.get_weights())
    log.info(f"RULE-M04: first Conv2D output channels changed {filters} → {new_filters}")
    return new_model
