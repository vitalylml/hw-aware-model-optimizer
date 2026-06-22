"""
Unit tests for the hw-aware-model-optimizer.

Run with:  pytest tests/ -v
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cnn():
    """A small CNN with intentional rule violations for testing."""
    import tensorflow as tf
    tf.random.set_seed(0)

    inputs = tf.keras.Input(shape=(32, 32, 3), batch_size=None, name="input")
    # L01 violation: 30 filters (not multiple of 16)
    x = tf.keras.layers.Conv2D(30, 3, padding="same", activation="relu", name="conv1")(inputs)
    # C01 candidate: Conv → BN
    x = tf.keras.layers.Conv2D(64, 3, padding="same", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    # L06 violation: gelu activation
    x = tf.keras.layers.Dense(100, activation="gelu", name="dense1")(x)
    # L01 violation: 10 filters
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs, outputs)


@pytest.fixture
def simple_cnn_path(simple_cnn, tmp_path):
    """Save simple_cnn to a temp .h5 file and return the path."""
    path = str(tmp_path / "test_model.h5")
    simple_cnn.save(path)
    return path


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------

class TestModelAnalyzer:

    def test_loads_model(self, simple_cnn_path):
        from optimizer.analyzer import ModelAnalyzer
        report = ModelAnalyzer(simple_cnn_path).analyze()
        assert report.total_params > 0

    def test_detects_channel_misalignment(self, simple_cnn_path):
        from optimizer.analyzer import ModelAnalyzer
        report = ModelAnalyzer(simple_cnn_path).analyze()
        rule_ids = [f.rule_id for f in report.violations]
        assert "RULE-L01" in rule_ids

    def test_detects_unsupported_activation(self, simple_cnn_path):
        from optimizer.analyzer import ModelAnalyzer
        report = ModelAnalyzer(simple_cnn_path).analyze()
        rule_ids = [f.rule_id for f in report.violations]
        assert "RULE-L06" in rule_ids

    def test_detects_bn_after_conv(self, simple_cnn_path):
        from optimizer.analyzer import ModelAnalyzer
        report = ModelAnalyzer(simple_cnn_path).analyze()
        rule_ids = [f.rule_id for f in report.warnings]
        assert "RULE-C01" in rule_ids

    def test_report_has_layers(self, simple_cnn_path):
        from optimizer.analyzer import ModelAnalyzer
        report = ModelAnalyzer(simple_cnn_path).analyze()
        assert len(report.layers) > 0

    def test_report_to_dict(self, simple_cnn_path):
        from optimizer.analyzer import ModelAnalyzer
        report = ModelAnalyzer(simple_cnn_path).analyze()
        d = report.to_dict()
        assert "violations" in d
        assert "layers" in d

    def test_report_save(self, simple_cnn_path, tmp_path):
        from optimizer.analyzer import ModelAnalyzer
        report = ModelAnalyzer(simple_cnn_path).analyze()
        out = tmp_path / "report.json"
        report.save(out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Transforms tests
# ---------------------------------------------------------------------------

class TestStaticTransforms:

    def test_substitute_activations(self, simple_cnn):
        from optimizer.transforms import substitute_activations
        transformed = substitute_activations(simple_cnn)
        # dense1 should now have relu6 instead of gelu
        layer = transformed.get_layer("dense1")
        cfg = layer.get_config()
        act = cfg.get("activation", "")
        if isinstance(act, dict):
            act = act.get("class_name", "").lower()
        assert str(act).lower() in ("relu6", "relu"), f"Expected relu6, got {act}"

    # def test_fold_batch_norm_numerics(self, simple_cnn):
    #     from optimizer.transforms import fold_batch_norm
    #     # After folding, BN should be identity (gamma=1, beta=0)
    #     folded = fold_batch_norm(simple_cnn)
    #     bn = folded.get_layer("bn1")
    #     gamma, beta, mean, var = bn.get_weights()
    #     np.testing.assert_allclose(gamma, np.ones_like(gamma), atol=1e-5)
    #     np.testing.assert_allclose(beta, np.zeros_like(beta), atol=1e-5)

    # def test_fold_preserves_output_shape(self, simple_cnn):
    #     from optimizer.transforms import fold_batch_norm
    #     import tensorflow as tf
    #     x = tf.random.normal((1, 32, 32, 3))
    #     orig_out = simple_cnn(x, training=False).numpy()
    #     folded = fold_batch_norm(simple_cnn)
    #     fold_out = folded(x, training=False).numpy()
    #     # Outputs should be numerically close
    #     np.testing.assert_allclose(orig_out, fold_out, atol=1e-4)

    def test_apply_all_static_transforms(self, simple_cnn_path):
        import tensorflow as tf
        import lscquant
        from optimizer.transforms import apply_all_static_transforms
        model = lscquant.load_model(simple_cnn_path)
        transformed = apply_all_static_transforms(model)
        assert transformed is not None


# ---------------------------------------------------------------------------
# Rules tests
# ---------------------------------------------------------------------------

class TestRules:

    def test_build_rule_catalogue(self):
        from optimizer.rules import build_rule_catalogue, APPLICATION_ORDER
        rules = build_rule_catalogue()
        assert len(rules) == 16
        for rule_id in APPLICATION_ORDER:
            assert rule_id in rules

    def test_rule_str(self):
        from optimizer.rules import build_rule_catalogue
        rules = build_rule_catalogue()
        s = str(rules["RULE-L01"])
        assert "RULE-L01" in s

    def test_application_order_complete(self):
        from optimizer.rules import build_rule_catalogue, APPLICATION_ORDER
        rules = build_rule_catalogue()
        for rid in APPLICATION_ORDER:
            assert rid in rules, f"{rid} in APPLICATION_ORDER but not in catalogue"


# ---------------------------------------------------------------------------
# Exporter tests (no Vela required)
# ---------------------------------------------------------------------------

class TestExporter:

    def test_make_representative_dataset(self):
        from optimizer.exporter import make_representative_dataset
        samples = np.random.rand(10, 32, 32, 3).astype(np.float32)
        fn = make_representative_dataset(samples)
        batches = list(fn())
        assert len(batches) == 10
        assert batches[0][0].shape == (1, 32, 32, 3)

    def test_export_tflite(self, simple_cnn, tmp_path):
        from optimizer.exporter import export_tflite_int8, make_representative_dataset
        samples = np.random.rand(50, 32, 32, 3).astype(np.float32)
        rep_dataset = make_representative_dataset(samples)
        out = tmp_path / "model.tflite"
        result = export_tflite_int8(simple_cnn, rep_dataset, output_path=out)
        assert result.exists()
        assert result.stat().st_size > 0


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestReport:

    def test_save_html_report(self, simple_cnn_path, tmp_path):
        from optimizer.analyzer import ModelAnalyzer
        from optimizer.report import save_html_report
        report = ModelAnalyzer(simple_cnn_path).analyze()
        out = tmp_path / "report.html"
        save_html_report(report, output_path=out)
        assert out.exists()
        content = out.read_text()
        assert "Ethos-U65" in content
