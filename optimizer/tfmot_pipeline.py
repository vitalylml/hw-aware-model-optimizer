"""
Phase 3 — TFMOT collaborative optimization pipeline.

Step order (must not be reversed):
  1. Pruning            (RULE-L03, RULE-M03)
  2. Sparsity-preserving clustering  (RULE-L04)
  3. PCQAT              (RULE-M01, RULE-L02)

Each step wraps the model with TFMOT instrumentation, fine-tunes for a few
epochs, then strips the wrapper before passing to the next step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TFMOTPipelineConfig:
    # ── Pruning ─────────────────────────────────────────────────────────
    pruning_target_sparsity: float = 0.70
    pruning_first_layer_sparsity: float = 0.30
    pruning_epochs: int = 5
    pruning_lr: float = 1e-4
    pruning_first_layer_name: Optional[str] = None  # autodetect if None

    # ── Clustering ──────────────────────────────────────────────────────
    n_clusters: int = 32
    clustering_epochs: int = 3
    clustering_lr: float = 1e-4

    # ── PCQAT ───────────────────────────────────────────────────────────
    pcqat_epochs: int = 3
    pcqat_lr: float = 1e-5

    # ── General ─────────────────────────────────────────────────────────
    loss: str = "categorical_crossentropy"
    metrics: list = field(default_factory=lambda: ["accuracy"])
    checkpoint_dir: str = "./checkpoints"

    # ── Skip flags (for partial pipeline runs) ──────────────────────────
    skip_pruning: bool = False
    skip_clustering: bool = False
    skip_pcqat: bool = False


# ---------------------------------------------------------------------------
# Step 1 — Pruning
# ---------------------------------------------------------------------------

def apply_pruning(
    model,
    train_dataset,
    config: TFMOTPipelineConfig,
) -> object:
    """
    Wraps each Conv2D and Dense layer with tfmot pruning, fine-tunes,
    then strips pruning wrappers.

    First Conv2D layer gets a lower sparsity target (RULE-M03).
    """
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot

    steps_per_epoch = _dataset_steps(train_dataset)
    end_step = steps_per_epoch * config.pruning_epochs

    # Detect first conv layer name if not provided
    first_layer_name = config.pruning_first_layer_name
    if first_layer_name is None:
        for layer in model.layers:
            if type(layer).__name__ == "Conv2D":
                first_layer_name = layer.name
                break

    def make_schedule(final_sparsity: float):
        return tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=final_sparsity,
            begin_step=0,
            end_step=end_step,
        )

    def pruning_fn(layer):
        ltype = type(layer).__name__
        if ltype not in ("Conv2D", "DepthwiseConv2D", "Dense"):
            return layer
        sparsity = (
            config.pruning_first_layer_sparsity
            if layer.name == first_layer_name
            else config.pruning_target_sparsity
        )
        return tfmot.sparsity.keras.prune_low_magnitude(
            layer, pruning_schedule=make_schedule(sparsity)
        )

    pruning_model = tf.keras.models.clone_model(model, clone_function=pruning_fn)
    pruning_model.set_weights(model.get_weights())
    pruning_model.compile(
        optimizer=tf.keras.optimizers.Adam(config.pruning_lr),
        loss=config.loss,
        metrics=config.metrics,
    )

    log.info(
        "RULE-L03/M03: pruning — target_sparsity=%.0f%%, first_layer=%.0f%%, epochs=%d",
        config.pruning_target_sparsity * 100,
        config.pruning_first_layer_sparsity * 100,
        config.pruning_epochs,
    )

    pruning_model.fit(
        train_dataset,
        epochs=config.pruning_epochs,
        callbacks=[
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(
                log_dir=str(Path(config.checkpoint_dir) / "pruning_logs")
            ),
        ],
    )

    stripped = tfmot.sparsity.keras.strip_pruning(pruning_model)

    # Report achieved sparsity
    _log_sparsity(stripped)
    return stripped


# ---------------------------------------------------------------------------
# Step 2 — Sparsity-preserving clustering
# ---------------------------------------------------------------------------

def apply_clustering(
    stripped_pruned_model,
    train_dataset,
    config: TFMOTPipelineConfig,
) -> object:
    """
    Applies k-means weight clustering while preserving pruning sparsity.
    Uses preserve_sparsity=True to avoid filling zero weights with cluster centroids.
    """
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot

    CentroidInit = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
        "number_of_clusters": config.n_clusters,
        "cluster_centroids_init": CentroidInit.KMEANS_PLUS_PLUS,
        "preserve_sparsity": True,
    }

    log.info(
        "RULE-L04: clustering — n_clusters=%d, preserve_sparsity=True, epochs=%d",
        config.n_clusters,
        config.clustering_epochs,
    )

    clustered_model = tfmot.clustering.keras.cluster_weights(
        stripped_pruned_model, **clustering_params
    )
    clustered_model.compile(
        optimizer=tf.keras.optimizers.Adam(config.clustering_lr),
        loss=config.loss,
        metrics=config.metrics,
    )
    clustered_model.fit(train_dataset, epochs=config.clustering_epochs)

    stripped = tfmot.clustering.keras.strip_clustering(clustered_model)
    log.info("RULE-L04: clustering complete and stripped")
    return stripped


# ---------------------------------------------------------------------------
# Step 3 — PCQAT (Pruning + Clustering Preserving QAT)
# ---------------------------------------------------------------------------

def apply_pcqat(
    stripped_clustered_model,
    train_dataset,
    config: TFMOTPipelineConfig,
) -> object:
    """
    Applies PCQAT — quantization-aware training that preserves both
    sparsity (from pruning) and clustering.

    If the PCQAT API is unavailable (older tfmot), falls back to standard QAT.
    """
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot

    log.info(
        "RULE-M01/L02: PCQAT — epochs=%d, lr=%g",
        config.pcqat_epochs,
        config.pcqat_lr,
    )

    try:
        pcqat_model = tfmot.experimental.combine.create_model_for_pcqat(
            stripped_clustered_model
        )
        log.info("Using PCQAT (sparsity + cluster preserving QAT)")
    except AttributeError:
        log.warning(
            "tfmot.experimental.combine.create_model_for_pcqat not available "
            "in this version — falling back to standard QAT"
        )
        pcqat_model = tfmot.quantization.keras.quantize_model(stripped_clustered_model)

    pcqat_model.compile(
        optimizer=tf.keras.optimizers.Adam(config.pcqat_lr),
        loss=config.loss,
        metrics=config.metrics,
    )
    pcqat_model.fit(train_dataset, epochs=config.pcqat_epochs)

    log.info("RULE-M01/L02: PCQAT complete")
    return pcqat_model


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_tfmot_pipeline(
    model,
    train_dataset,
    config: Optional[TFMOTPipelineConfig] = None,
    save_intermediates: bool = True,
) -> object:
    """
    Run the full collaborative optimization pipeline: Prune → Cluster → PCQAT.

    Parameters
    ----------
    model          : tf.keras.Model from Phase 2 (static transforms)
    train_dataset  : tf.data.Dataset yielding (inputs, labels)
    config         : TFMOTPipelineConfig; defaults used if None
    save_intermediates : if True, saves .keras checkpoint after each step

    Returns
    -------
    PCQAT-wrapped tf.keras.Model ready for TFLite INT8 export (Phase 4)
    """
    import tensorflow as tf

    cfg = config or TFMOTPipelineConfig()
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    log.info("=== Phase 3: TFMOT collaborative optimization pipeline ===")

    current = model

    # Step 1: Pruning
    if not cfg.skip_pruning:
        current = apply_pruning(current, train_dataset, cfg)
        if save_intermediates:
            _save(current, cfg.checkpoint_dir, "after_pruning")
    else:
        log.info("Pruning skipped (skip_pruning=True)")

    # Step 2: Clustering
    if not cfg.skip_clustering:
        current = apply_clustering(current, train_dataset, cfg)
        if save_intermediates:
            _save(current, cfg.checkpoint_dir, "after_clustering")
    else:
        log.info("Clustering skipped (skip_clustering=True)")

    # Step 3: PCQAT
    if not cfg.skip_pcqat:
        current = apply_pcqat(current, train_dataset, cfg)
        if save_intermediates:
            _save(current, cfg.checkpoint_dir, "after_pcqat")
    else:
        log.info("PCQAT skipped (skip_pcqat=True)")

    log.info("=== Phase 3 complete ===")
    return current


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(model, checkpoint_dir: str, tag: str) -> None:
    import tensorflow as tf
    path = str(Path(checkpoint_dir) / f"model_{tag}.keras")
    lscquant.save_model(model, path)
    log.info("Checkpoint saved: %s", path)


def _dataset_steps(dataset) -> int:
    """Estimate number of steps/batches in a tf.data.Dataset."""
    try:
        return len(dataset)
    except TypeError:
        # Unbounded or unknown cardinality
        log.warning("Dataset length unknown — defaulting to 100 steps for pruning schedule")
        return 100


def _log_sparsity(model) -> None:
    """Log per-layer and global sparsity after pruning."""
    all_zeros, all_total = 0, 0
    for layer in model.layers:
        for w in layer.weights:
            arr = w.numpy()
            zeros = int(np.sum(arr == 0))
            total = arr.size
            all_zeros += zeros
            all_total += total
            if total > 0 and type(layer).__name__ in ("Conv2D", "DepthwiseConv2D", "Dense"):
                log.debug("  %s: sparsity=%.1f%%", layer.name, zeros / total * 100)
    if all_total > 0:
        log.info("Global sparsity after pruning: %.1f%%", all_zeros / all_total * 100)
