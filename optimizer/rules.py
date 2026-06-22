"""
Rule definitions for Ethos-U65 model optimization.
Each rule maps directly to the documented rule set (RULE-L01 … RULE-M05).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Scope(str, Enum):
    LAYER = "layer"
    COMBINATION = "combination"
    MODEL = "model"


class Priority(str, Enum):
    HIGH = "HIGH"
    MED = "MED"
    LOW = "LOW"


class Impact(str, Enum):
    MAC_UTIL = "MAC utilization"
    AXI1_BW = "AXI1 bandwidth / weight fetch"
    CYCLES = "cycles per inference"
    FALLBACK = "operator fallback elimination"
    MEMORY = "SRAM/DRAM footprint"
    FUSION = "Vela layer fusion"


@dataclass
class Rule:
    id: str
    scope: Scope
    target: str
    condition: str
    action: str
    impact: List[Impact]
    verify: str
    priority: Priority
    params: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.id} [{self.priority.value}] ({self.scope.value})\n"
            f"  target   : {self.target}\n"
            f"  condition: {self.condition}\n"
            f"  action   : {self.action}\n"
            f"  impact   : {', '.join(i.value for i in self.impact)}\n"
            f"  verify   : {self.verify}\n"
            f"  params   : {self.params}"
        )


# ---------------------------------------------------------------------------
# Supported op / activation sets for Ethos-U65
# ---------------------------------------------------------------------------

SUPPORTED_OPS: set[str] = {
    "Conv2D", "DepthwiseConv2D", "Dense",
    "BatchNormalization", "ReLU", "Activation",
    "MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D",
    "Add", "Multiply", "Subtract",
    "LSTM", "RNN", "GRU",
    "Reshape", "Softmax", "Flatten",
    "ZeroPadding2D", "Concatenate",
    "InputLayer",
    "Split"
}

SUPPORTED_ACTIVATIONS: set[Optional[str]] = {
    "relu", "relu6", "sigmoid", "tanh", "linear", "softmax", None,
}

FALLBACK_ACTIVATIONS: dict[str, str] = {
    "gelu": "relu6",
    "swish": "relu6",
    "mish": "relu6",
    "elu": "relu",
    "selu": "relu",
    "leaky_relu": "relu6",
}

ATTENTION_LAYER_TYPES: set[str] = {
    "MultiHeadAttention", "Attention", "AdditiveAttention",
}

FOLDABLE_CONV_TYPES: set[str] = {"Conv2D", "DepthwiseConv2D"}
WEIGHT_LAYER_TYPES: set[str] = {"Conv2D", "DepthwiseConv2D", "Dense"}

# ---------------------------------------------------------------------------
# Rule catalogue
# ---------------------------------------------------------------------------

CHANNEL_MULTIPLE = 16          # NHCWB16 internal format
DEFAULT_SPARSITY_TARGET = 0.70
FIRST_CONV_SPARSITY_TARGET = 0.30
DEFAULT_CLUSTER_COUNT = 32


def build_rule_catalogue() -> dict[str, Rule]:
    return {
        # ── Layer-level rules ────────────────────────────────────────────────
        "RULE-L01": Rule(
            id="RULE-L01",
            scope=Scope.LAYER,
            target="Conv2D, DepthwiseConv2D, Dense",
            condition=f"output_channels % {CHANNEL_MULTIPLE} != 0",
            action=(
                f"Round output_channels up to nearest multiple of {CHANNEL_MULTIPLE}; "
                "adjust downstream layer input_channels to match."
            ),
            impact=[Impact.MAC_UTIL],
            verify="All connected layers remain channel-consistent; accuracy delta < threshold.",
            priority=Priority.HIGH,
            params={"channel_multiple": CHANNEL_MULTIPLE},
        ),
        "RULE-L02": Rule(
            id="RULE-L02",
            scope=Scope.LAYER,
            target="any layer with dtype=INT16 or FLOAT32",
            condition="layer task is classification | detection | segmentation",
            action=(
                "Quantize activations to INT8 via QAT (preferred) or PTQ. "
                "Keep INT16 only for audio or HDR image layers."
            ),
            impact=[Impact.CYCLES],
            verify="Per-layer dtype audit; accuracy delta within threshold.",
            priority=Priority.HIGH,
        ),
        "RULE-L03": Rule(
            id="RULE-L03",
            scope=Scope.LAYER,
            target="Conv2D, DepthwiseConv2D, Dense",
            condition="sparsity(layer.weights) < 0.5",
            action=(
                "Apply magnitude pruning targeting 50–80% zero weights; "
                "retrain with gradual sparsity schedule."
            ),
            impact=[Impact.AXI1_BW],
            verify="sparsity(layer.weights) >= target; accuracy delta < threshold.",
            priority=Priority.HIGH,
            params={
                "target_sparsity": DEFAULT_SPARSITY_TARGET,
                "first_layer_sparsity": FIRST_CONV_SPARSITY_TARGET,
            },
        ),
        "RULE-L04": Rule(
            id="RULE-L04",
            scope=Scope.LAYER,
            target="Conv2D, DepthwiseConv2D, Dense",
            condition=f"unique_weight_values(layer) > {DEFAULT_CLUSTER_COUNT}",
            action=(
                f"Apply k-means clustering with k={DEFAULT_CLUSTER_COUNT} (or 16 for "
                "maximum compression). Prefer k=16 when accuracy permits."
            ),
            impact=[Impact.AXI1_BW],
            verify="Vela-reported encoded weight size reduced; accuracy delta < threshold.",
            priority=Priority.MED,
            params={"n_clusters": DEFAULT_CLUSTER_COUNT, "min_clusters": 16},
        ),
        "RULE-L05": Rule(
            id="RULE-L05",
            scope=Scope.LAYER,
            target="any operator not in supported op set",
            condition="layer.operator NOT IN ethos_u65_supported_ops",
            action=(
                "Replace with nearest supported equivalent; if no equivalent exists, "
                "fuse into adjacent supported layers; last resort: accept CMSIS-NN fallback."
            ),
            impact=[Impact.FALLBACK],
            verify="Vela compilation produces single NPU job (no fallback).",
            priority=Priority.HIGH,
        ),
        "RULE-L06": Rule(
            id="RULE-L06",
            scope=Scope.LAYER,
            target="activation functions not natively supported",
            condition="activation NOT IN {relu, relu6, sigmoid, tanh}",
            action=(
                "Substitute with relu6 for classification/detection; "
                "approximate GELU with relu or piecewise linear for non-accuracy-critical layers."
            ),
            impact=[Impact.FALLBACK],
            verify="Functional equivalence or acceptable accuracy delta; no fallback in Vela.",
            priority=Priority.HIGH,
            params={"substitutions": FALLBACK_ACTIVATIONS},
        ),
        "RULE-L07": Rule(
            id="RULE-L07",
            scope=Scope.LAYER,
            target="Conv2D with kernel_size >= 3x3",
            condition=(
                "layer is compute-bound (active cycles scale with MAC config); "
                "layer not in first 1-2 input layers"
            ),
            action=(
                "Factorize into DepthwiseConv2D(3x3) + Conv2D(1x1). "
                "Reduces FLOPs by ~8-9x for 3x3 case."
            ),
            impact=[Impact.CYCLES, Impact.AXI1_BW],
            verify="Channel alignment rule L01 still satisfied; accuracy delta < threshold.",
            priority=Priority.MED,
        ),

        # ── Combination rules ────────────────────────────────────────────────
        "RULE-C01": Rule(
            id="RULE-C01",
            scope=Scope.COMBINATION,
            target="Conv2D → BatchNorm (sequential)",
            condition=(
                "BN parameters are not constant at export time "
                "(e.g. model exported in training mode, or trainable=True with unfrozen stats)"
            ),
            action=(
                "The TFLite converter automatically folds BN into Conv2D bias during conversion. "
                "No manual folding required. "
                "Ensure model is exported in inference mode: "
                "  model(x, training=False) or model.trainable = False before save. "
                "Manual fold only needed if: (1) debugging numerics pre-conversion, "
                "or (2) doing model surgery that requires BN-free Keras weights."
            ),
            impact=[Impact.FUSION, Impact.MEMORY],
            verify=(
                "After TFLite conversion, inspect the flatbuffer — "
                "no BATCH_NORM op should appear; only CONV_2D with bias. "
                "Vela NNG output shows Conv2DBias, not separate BN operator."
            ),
            priority=Priority.HIGH,
        ),
        "RULE-C02": Rule(
            id="RULE-C02",
            scope=Scope.COMBINATION,
            target="Conv2D → BatchNorm → ReLU (linear sequence)",
            condition="sequence is linear (no branches or skip connections between three layers)",
            action=(
                "Ensure BN is folded (RULE-C01) and activation is ReLU/ReLU6 (RULE-L06); "
                "flag sequence for Vela fusion."
            ),
            impact=[Impact.FUSION, Impact.MEMORY],
            verify="Vela compilation log shows fused operator; no intermediate tensor written.",
            priority=Priority.HIGH,
        ),
        "RULE-C03": Rule(
            id="RULE-C03",
            scope=Scope.COMBINATION,
            target="Add layer in residual/skip connections",
            condition="Add inputs from two branches with matching channel counts and spatial dims",
            action=(
                "Ensure both branches satisfy RULE-L01; ensure Add output dtype is INT8; "
                "do not insert unsupported ops between branch and Add."
            ),
            impact=[Impact.FALLBACK],
            verify="Add resolved by element-wise engine in Vela; no channel mismatch.",
            priority=Priority.MED,
        ),
        "RULE-C04": Rule(
            id="RULE-C04",
            scope=Scope.COMBINATION,
            target="Dense → Dense (two consecutive FC layers with no nonlinearity)",
            condition="no activation between the two Dense layers",
            action=(
                "Merge into single Dense layer: "
                "W' = W2 @ W1, b' = W2 @ b1 + b2"
            ),
            impact=[Impact.CYCLES, Impact.MEMORY],
            verify="Numerical equivalence; new weight matrix satisfies RULE-L01.",
            priority=Priority.MED,
        ),
        "RULE-C05": Rule(
            id="RULE-C05",
            scope=Scope.COMBINATION,
            target="LSTM/RNN cell decomposed into MatMul + Add + Sigmoid + Tanh + Mul",
            condition="cell is implemented as decomposed ops rather than a single LSTM operator",
            action=(
                "Replace with a single fused LSTM/RNN operator natively supported by Ethos-U65; "
                "use element-wise engine for scaling ops."
            ),
            impact=[Impact.FALLBACK, Impact.CYCLES],
            verify="Vela reports LSTM as single NPU operator; hidden state accuracy preserved.",
            priority=Priority.HIGH,
        ),

        # ── Model-level rules ────────────────────────────────────────────────
        "RULE-M01": Rule(
            id="RULE-M01",
            scope=Scope.MODEL,
            target="entire model",
            condition="any layer has dtype != INT8 (excluding justified INT16 layers)",
            action=(
                "Apply INT8 quantization globally via QAT (preferred) or PTQ; "
                "annotate INT16 exceptions explicitly with justification tag."
            ),
            impact=[Impact.CYCLES, Impact.AXI1_BW, Impact.MEMORY],
            verify="Per-layer dtype audit; top-level accuracy delta within accepted threshold.",
            priority=Priority.HIGH,
        ),
        "RULE-M02": Rule(
            id="RULE-M02",
            scope=Scope.MODEL,
            target="entire model",
            condition=(
                "model contains attention/self-attention blocks, "
                "transformer encoder/decoder stacks, or layer norms"
            ),
            action=(
                "Flag as partially unsupported on U65; evaluate replacing attention with "
                "depthwise conv approximations; or migrate to Ethos-U85."
            ),
            impact=[Impact.FALLBACK],
            verify="Vela job count = 1; if > 1, count and cost each fallback boundary.",
            priority=Priority.HIGH,
        ),
        "RULE-M03": Rule(
            id="RULE-M03",
            scope=Scope.MODEL,
            target="entire model weight distribution",
            condition=(
                "global_sparsity(model) < 0.5 OR any critical layer has sparsity < 0.3"
            ),
            action=(
                "Apply gradual magnitude pruning across all Conv2D and Dense layers; "
                "target 50-70% global sparsity; protect first conv (20-30% sparsity)."
            ),
            impact=[Impact.AXI1_BW],
            verify=(
                "Per-layer sparsity report; global sparsity >= target; "
                "Vela encoded weight size reduced vs baseline."
            ),
            priority=Priority.HIGH,
            params={
                "global_sparsity_target": DEFAULT_SPARSITY_TARGET,
                "first_layer_sparsity_target": FIRST_CONV_SPARSITY_TARGET,
            },
        ),
        "RULE-M04": Rule(
            id="RULE-M04",
            scope=Scope.MODEL,
            target="first Conv2D layer",
            condition="input has 1 or 3 channels (standard RGB/grayscale)",
            action=(
                "Keep input channels as-is; ensure output channels of first conv "
                f"satisfy RULE-L01 (multiple of {CHANNEL_MULTIPLE}), minimum 16."
            ),
            impact=[Impact.MAC_UTIL],
            verify=f"first conv output_channels % {CHANNEL_MULTIPLE} == 0.",
            priority=Priority.HIGH,
        ),
        "RULE-M05": Rule(
            id="RULE-M05",
            scope=Scope.MODEL,
            target="model batch dimension",
            condition="batch_size != 1",
            action=(
                "Fix batch_size = 1 at export. Ethos-U65 supports only N=1. "
                "Use system-level pipelining for throughput, not batching."
            ),
            impact=[Impact.CYCLES],
            verify="Exported model input shape[0] == 1.",
            priority=Priority.HIGH,
        ),
        "RULE-M06": Rule(
            id="RULE-M06",
            scope=Scope.MODEL,
            target="NPU execution schedule (Vela cascades)",
            condition=(
                "Conv2D/DepthwiseConv2D/SeparableConv2D/Dense has activation in (None, 'linear') "
                "and a single standalone activation layer (ReLU, ReLU6, supported Activation) "
                "as its only consumer."
            ),
            action=(
                "Fuse the standalone activation into the producing layer's `activation` slot "
                "and remove the now-redundant activation layer. Reduces operator count, "
                "eliminates an intermediate cascade boundary, and restores Vela's native "
                "Conv+Activation fusion. Implemented by optimizer.transforms.apply_cascading_optimizations."
            ),
            impact=[Impact.CYCLES, Impact.MEMORY, Impact.AXI1_BW, Impact.FUSION],
            verify=(
                "Vela shows reduced operator count, lower intermediate SRAM/DRAM traffic, "
                "and improved cycle estimate for affected regions vs baseline."
            ),
            priority=Priority.MED,
            params={
                "implemented": True,
                "optimise": "Performance",
                "memory_mode": "Shared_Sram",
                "fusable_activations": ["relu", "relu6", "sigmoid", "tanh", "linear"],
                "fusable_producer_types": [
                    "Conv2D", "DepthwiseConv2D", "SeparableConv2D", "Dense",
                ],
            },
        ),
        # ── Sub-rules of RULE-M06 (cascading) ─────────────────────────────────
        "RULE-M06.1": Rule(
            id="RULE-M06.1",
            scope=Scope.MODEL,
            target="Conv2D/DepthwiseConv2D/SeparableConv2D(stride=1) followed by MaxPooling2D(pool=2, stride=2)",
            condition=(
                "A Conv2D / DepthwiseConv2D / SeparableConv2D with strides=(1,1) is "
                "followed (with only BatchNormalization and/or Activation/ReLU in "
                "between, each a single-consumer chain) by a MaxPooling2D layer of "
                "pool_size=(2,2) and strides=(2,2)."
            ),
            action=(
                "Set the producing layer's strides to (2,2) and remove the trailing "
                "MaxPooling2D. Halves activation traffic between the two layers and "
                "removes a hard cascade boundary on the NPU. Implemented by "
                "optimizer.transforms.fold_maxpool_into_strided_conv."
            ),
            impact=[Impact.CYCLES, Impact.MEMORY, Impact.AXI1_BW, Impact.FUSION],
            verify=(
                "Operator count drops by one per fold; mean MAC utilization stable or "
                "higher; minor accuracy drift recovered by Phase 3 PCQAT."
            ),
            priority=Priority.MED,
            params={
                "implemented": True,
                "pool_size":   [2, 2],
                "pool_strides": [2, 2],
                "allowed_intermediate_layers": [
                    "BatchNormalization", "Activation", "ReLU",
                ],
            },
        ),
        "RULE-M06.2": Rule(
            id="RULE-M06.2",
            scope=Scope.MODEL,
            target="Resizing(interpolation='bilinear') in feature-pyramid path",
            condition=(
                "Resizing layer uses bilinear interpolation, output H and W are integer "
                "multiples of input H and W with equal scale factor >= 2, and the "
                "channel count is preserved."
            ),
            action=(
                "Replace the Resizing layer with a depthwise-style "
                "Conv2DTranspose(strides=factor, kernel_size=2*factor, padding='same') "
                "whose weights are pre-initialised to a bilinear upsampling kernel "
                "(diagonal in the channel dimension so each output channel depends only "
                "on the same input channel). Conv2DTranspose is NPU-native and "
                "cascadable with the following Conv block, whereas bilinear Resizing is "
                "a standalone op that forces a tensor materialisation. Implemented by "
                "optimizer.transforms.replace_bilinear_resize_with_conv2dtranspose. "
                "Approximate (boundary sampling differs from tf.image.resize); minor "
                "accuracy drift recovered by Phase 3 PCQAT."
            ),
            impact=[Impact.CYCLES, Impact.MEMORY, Impact.FUSION],
            verify=(
                "Vela report shows the Resizing op replaced by a TRANSPOSE_CONV op that "
                "fuses with the following Conv block; mean MAC utilization improves in "
                "the affected region."
            ),
            priority=Priority.LOW,
            params={
                "implemented": True,
                "min_factor": 2,
                "kernel_size_formula": "2 * factor",
                "use_bias": False,
                "weight_init": "bilinear (FCN-style)",
                "channel_mixing": False,
            },
        ),
        "RULE-M06.3": Rule(
            id="RULE-M06.3",
            scope=Scope.MODEL,
            target="ShuffleNet-style Split + Concat boundary tensors",
            condition=(
                "A Concatenate(axis=A) layer has at least 3 inputs and contains an "
                "adjacent input pair (Split:tensor_index=0, Split:tensor_index=1) "
                "from the same Split layer with num_or_size_splits=2 at axis A."
            ),
            action=(
                "Replace the adjacent (split:0, split:1) input pair in the Concat with "
                "a single direct reference to the Split's own upstream input tensor. "
                "Concat([split[0], split[1]], axis=A) reconstructs the Split's input "
                "exactly, so the rewrite preserves math. Reduces Concat's input list "
                "size, eliminates two cascade-boundary tensors per pair, and shrinks "
                "SRAM live ranges. The Split itself is preserved when its second "
                "output still has other consumers (typical ShuffleNet-v2 work-branch). "
                "Implemented by optimizer.transforms.fuse_split_concat_blocks."
            ),
            impact=[Impact.MEMORY, Impact.AXI1_BW, Impact.FUSION],
            verify=(
                "Concatenation tensor count per affected Concat drops by one; Vela "
                "schedule shows smaller SRAM live ranges for the affected region."
            ),
            priority=Priority.MED,
            params={
                "implemented": True,
                "min_concat_inputs": 3,
                "required_split_num": 2,
                "matched_axis_convention": "channel (axis = -1 / 3)",
                "preserves_split_layer": True,
            },
        ),
        "RULE-M06.4": Rule(
            id="RULE-M06.4",
            scope=Scope.MODEL,
            target="Residual Add nodes with mixed-distance operands",
            condition=(
                "Two-input Add layer where one operand's producer chain reaches a "
                "Conv2D / DepthwiseConv2D / SeparableConv2D / Dense layer through at "
                "most a few BatchNormalization or Activation/ReLU hops, and the "
                "other operand does not (or does so at greater distance), and the "
                "shorter-distance operand is currently at index 1."
            ),
            action=(
                "Swap Add operands so the operand whose producer is the immediately "
                "preceding Conv is at index 0. Vela's pattern matcher then fuses the "
                "Add into the preceding Conv's bias accumulator (fused Conv + "
                "ResAdd), removing the standalone Add op from the schedule. "
                "Implemented by optimizer.transforms.reorder_residual_add_operands. "
                "Add is commutative, so the rewrite is mathematically exact."
            ),
            impact=[Impact.CYCLES, Impact.FUSION],
            verify=(
                "Vela report shows fused Conv+ResAdd ops in place of standalone Add ops "
                "for the affected residual blocks."
            ),
            priority=Priority.LOW,
            params={
                "implemented": True,
                "max_chain_depth": 4,
                "fresh_branch_intermediate_types": [
                    "BatchNormalization", "Activation", "ReLU",
                ],
                "fresh_branch_producer_types": [
                    "Conv2D", "DepthwiseConv2D", "SeparableConv2D", "Dense",
                ],
            },
        ),
        "RULE-M06.5": Rule(
            id="RULE-M06.5",
            scope=Scope.MODEL,
            target="Concatenate layers with disparate-cost producers",
            condition=(
                "Concatenate(axis=-1) with two or more producers whose dtypes match and "
                "whose write regions could be made adjacent in SRAM."
            ),
            action=(
                "Annotate the Concat producers with a layout hint (or pre-allocate the "
                "concat output region) so Vela can perform in-place concatenation, "
                "skipping the explicit copy operation."
            ),
            impact=[Impact.MEMORY, Impact.AXI1_BW, Impact.FUSION],
            verify=(
                "Vela schedule reports adjacent-region writes for the affected concat "
                "tensors; SRAM bandwidth for the concat block drops."
            ),
            priority=Priority.LOW,
            params={
                "implemented": False,
                "todo": (
                    "Investigate Vela hooks for in-place concatenation. May require a "
                    "post-export TFLite metadata annotation rather than a Keras-level "
                    "rewrite."
                ),
            },
        ),
    }


# Canonical application order (rule IDs in sequence)
APPLICATION_ORDER: list[str] = [
    "RULE-M02", "RULE-M05", "RULE-M01",
    "RULE-L05", "RULE-L06", "RULE-C05",
    "RULE-C01", "RULE-C02", "RULE-C03",
    "RULE-L01", "RULE-M04", "RULE-L07",
    "RULE-C04", "RULE-M06", "RULE-M06.1", "RULE-M06.2", "RULE-M06.3", "RULE-M06.4",
    "RULE-M03", "RULE-L03", "RULE-L04",
]
