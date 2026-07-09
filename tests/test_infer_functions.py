#!/usr/bin/env python3

import os
import tempfile

import numpy as np
import pytest

from ifcb_infer.cli import (
    argparse_init,
    argparse_runtime_args,
    ensure_softmax,
    get_embedding_output_path,
    get_output_path,
    resolve_emit_embeddings,
    validate_score_output_args,
    validate_score_output_model,
    write_embeddings,
    write_output,
)

# The torch and notorch variants share a single argparse/runtime implementation.
argparse_init_torch = argparse_init
argparse_runtime_args_torch = argparse_runtime_args
get_output_path_torch = get_output_path


class TestGetOutputPath:
    """Test the get_output_path function from both inference scripts."""

    def setup_method(self):
        """Set up test fixtures."""
        self.args = type("Args", (), {})()
        self.args.outdir = "./outputs"
        self.args.outfile = "{MODEL_NAME}/{SUBPATH}/{BIN}.csv"
        self.args.run_date_str = "2025-01-15"
        self.args.model_name = "test_model"

    def test_default_pattern_no_subpath(self):
        """Test default pattern when bin has no relative directory (SUBPATH empty)."""
        result = get_output_path(self.args, "test_bin")
        assert result == os.path.normpath("./outputs/test_model/test_bin.csv")

    def test_default_pattern_with_subpath(self):
        """Test default pattern with a deep relative path."""
        result = get_output_path(
            self.args, "test_bin", "MVCO/2006/IFCB1_2006_157/test_bin"
        )
        assert result == os.path.normpath(
            "./outputs/test_model/MVCO/2006/IFCB1_2006_157/test_bin.csv"
        )

    def test_bin_only_pattern_flat_output(self):
        """Test {BIN}.csv pattern produces a flat list regardless of input structure."""
        self.args.outfile = "{BIN}.csv"
        result = get_output_path(
            self.args, "test_bin", "MVCO/2006/IFCB1_2006_157/test_bin"
        )
        assert result == os.path.normpath("./outputs/test_bin.csv")

    def test_run_date_pattern(self):
        """Test {RUN_DATE}/{SUBPATH}/{BIN}.csv pattern."""
        self.args.outfile = "{RUN_DATE}/{SUBPATH}/{BIN}.csv"
        result = get_output_path(self.args, "test_bin", "MVCO/2023/D20230108/test_bin")
        assert result == os.path.normpath(
            "./outputs/2025-01-15/MVCO/2023/D20230108/test_bin.csv"
        )

    def test_output_path_torch_version(self):
        """Test torch version of get_output_path function."""
        result = get_output_path_torch(self.args, "test_bin", "site/year/day/test_bin")
        assert result == os.path.normpath(
            "./outputs/test_model/site/year/day/test_bin.csv"
        )

    def test_output_path_with_custom_outdir(self):
        """Test output path with custom output directory."""
        self.args.outdir = "/custom/output/dir"
        result = get_output_path(
            self.args, "test_bin", "MVCO/2006/IFCB1_2006_157/test_bin"
        )
        assert result == os.path.normpath(
            "/custom/output/dir/test_model/MVCO/2006/IFCB1_2006_157/test_bin.csv"
        )

    def test_bin_relative_path_none_falls_back_to_bin_id(self):
        """Test that omitting bin_relative_path uses bin_id as the full subpath."""
        result = get_output_path(self.args, "test_bin", None)
        assert result == os.path.normpath("./outputs/test_model/test_bin.csv")

    def test_bin_at_root_of_input_dir(self):
        """Test when bin_relative_path has no directory component (bin sits at input root)."""
        result = get_output_path(self.args, "test_bin", "test_bin")
        assert result == os.path.normpath("./outputs/test_model/test_bin.csv")

    def test_subpath_without_bin_token_loses_bin_name(self):
        """Test that {SUBPATH} without {BIN} only captures the directory — bin name is absent.

        Users who rely on {SUBPATH} alone should migrate to {SUBPATH}/{BIN}.
        """
        self.args.outfile = "{MODEL_NAME}/{SUBPATH}.csv"
        result = get_output_path(
            self.args, "test_bin", "MVCO/2006/IFCB1_2006_157/test_bin"
        )
        assert result == os.path.normpath(
            "./outputs/test_model/MVCO/2006/IFCB1_2006_157.csv"
        )


class TestArgparseRuntimeArgs:
    """Test the argparse_runtime_args function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.args = type("Args", (), {})()
        self.args.MODEL = "/path/to/my_awesome_model.onnx"
        self.args.classes = None
        self.args.BINS = ["bin1", "bin2"]

    def test_runtime_args_basic(self, mocker):
        """Test basic runtime args processing."""
        # Mock environment and datetime
        mocker.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"})
        mock_datetime = mocker.patch("datetime.datetime")
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T14:30:45"

        argparse_runtime_args(self.args)

        assert self.args.run_date_str == "2025-01-15"
        assert self.args.run_time_str == "14:30:45"
        assert self.args.model_name == "my_awesome_model"
        assert self.args.gpus == [0, 1, 2]

    def test_runtime_args_no_gpu(self, mocker):
        """Test runtime args with no GPU specified."""
        mocker.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""})
        mock_datetime = mocker.patch("datetime.datetime")
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T14:30:45"

        argparse_runtime_args(self.args)

        assert self.args.gpus == []  # Empty list when no GPUs

    def test_runtime_args_with_classes_file(self, mocker):
        """Test runtime args with classes file."""
        mock_datetime = mocker.patch("datetime.datetime")
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T14:30:45"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("class1\nclass2\nclass3\n")
            classes_file = f.name

        try:
            self.args.classes = classes_file
            argparse_runtime_args(self.args)
            assert self.args.classes == ["class1", "class2", "class3"]
        finally:
            os.unlink(classes_file)

    def test_runtime_args_with_classes_json(self, mocker):
        """Test runtime args with labels.json format (index-keyed dict)."""
        mock_datetime = mocker.patch("datetime.datetime")
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T14:30:45"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            import json

            json.dump({"0": "class1", "1": "class2", "2": "class3"}, f)
            classes_file = f.name

        try:
            self.args.classes = classes_file
            argparse_runtime_args(self.args)
            assert self.args.classes == ["class1", "class2", "class3"]
        finally:
            os.unlink(classes_file)

    def test_runtime_args_with_classes_json_no_extension(self, mocker):
        """Test runtime args with JSON content but no .json extension."""
        mock_datetime = mocker.patch("datetime.datetime")
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T14:30:45"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            import json

            json.dump({"0": "classA", "1": "classB"}, f)
            classes_file = f.name

        try:
            self.args.classes = classes_file
            argparse_runtime_args(self.args)
            assert self.args.classes == ["classA", "classB"]
        finally:
            os.unlink(classes_file)

    def test_runtime_args_with_classes_json_order(self, mocker):
        """Test that JSON classes are sorted by integer index, not insertion order."""
        mock_datetime = mocker.patch("datetime.datetime")
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T14:30:45"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            # Write out-of-order keys to verify integer sorting
            f.write('{"2": "third", "0": "first", "1": "second"}')
            classes_file = f.name

        try:
            self.args.classes = classes_file
            argparse_runtime_args(self.args)
            assert self.args.classes == ["first", "second", "third"]
        finally:
            os.unlink(classes_file)

    def test_runtime_args_torch_version(self, mocker):
        """Test torch version of runtime args."""
        mock_datetime = mocker.patch("datetime.datetime")
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T14:30:45"
        argparse_runtime_args_torch(self.args)

        assert self.args.run_date_str == "2025-01-15"
        assert self.args.model_name == "my_awesome_model"


class TestOutfilePattern:
    """Test --outfile pattern parsing and defaults."""

    def test_default_outfile_pattern(self):
        """Test that default outfile pattern uses MODEL_NAME, SUBPATH, and BIN."""
        parser = argparse_init()
        args = parser.parse_args(["model.onnx", "bins/"])
        assert args.outfile == "{MODEL_NAME}/{SUBPATH}/{BIN}.csv"

    def test_custom_outfile_pattern(self):
        """Test that a custom outfile pattern is accepted."""
        parser = argparse_init()
        args = parser.parse_args(
            ["--outfile", "{RUN_DATE}/{SUBPATH}.csv", "model.onnx", "bins/"]
        )
        assert args.outfile == "{RUN_DATE}/{SUBPATH}.csv"

    def test_combined_pattern(self):
        """Test a pattern combining multiple tokens."""
        parser = argparse_init()
        args = parser.parse_args(
            [
                "--outfile",
                "{MODEL_NAME}/{RUN_DATE}/{SUBPATH}.csv",
                "model.onnx",
                "bins/",
            ]
        )
        assert args.outfile == "{MODEL_NAME}/{RUN_DATE}/{SUBPATH}.csv"


class TestBinDirectoryMapping:
    """Test bin-to-input-directory mapping logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.args = type("Args", (), {})()
        self.args.MODEL = "test_model.onnx"
        self.args.classes = None

    def test_bin_to_input_dir_structure(self, mocker):
        """Test that bin_to_input_dir mapping is created correctly."""
        self.args.BINS = ["dir1", "individual_bin"]

        # Mock datetime
        mock_datetime = mocker.patch("datetime.datetime")
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T14:30:45"

        # Mock directory and file checks
        mock_isdir = mocker.patch("os.path.isdir")
        mocker.patch("ifcbkit.sync_list_data_dirs", return_value=["dir1"])
        mock_dd_class = mocker.patch("ifcbkit.SyncIfcbDataDirectory")
        mock_dd_instance = mock_dd_class.return_value
        mock_dd_instance.list.return_value = [
            {
                "pid": "bin1",
                "hdr": "dir1/bin1.hdr",
                "adc": "dir1/bin1.adc",
                "roi": "dir1/bin1.roi",
            }
        ]

        # Set up mocks
        mock_isdir.side_effect = lambda x: x == "dir1"

        argparse_runtime_args(self.args)

        # Check that bin_to_input_dir mapping exists
        assert hasattr(self.args, "bin_to_input_dir")
        assert "dir1/bin1" in self.args.bin_to_input_dir
        assert self.args.bin_to_input_dir["dir1/bin1"] == "dir1"


@pytest.mark.parametrize(
    "model_path,expected_name",
    [
        ("/path/to/classifier.onnx", "classifier"),
        ("simple_model.onnx", "simple_model"),
        ("/complex/path/with_underscores_v2.onnx", "with_underscores_v2"),
        ("model.onnx", "model"),
    ],
)
def test_model_name_extraction(model_path, expected_name, mocker):
    """Parametrized test for model name extraction."""
    args = type("Args", (), {})()
    args.MODEL = model_path
    args.classes = None
    args.BINS = []

    mock_datetime = mocker.patch("datetime.datetime")
    mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T14:30:45"
    argparse_runtime_args(args)

    assert args.model_name == expected_name


class TestEnsureSoftmax:
    def test_logits_are_softmaxed(self):
        logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])
        result = ensure_softmax(logits)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(2), atol=1e-6)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_already_softmaxed_returned_as_is(self):
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        result = ensure_softmax(probs)
        np.testing.assert_array_equal(result, probs)

    def test_negative_values_trigger_softmax(self):
        logits = np.array([[-1.0, 0.0, 1.0]])
        result = ensure_softmax(logits)
        np.testing.assert_allclose(result.sum(axis=1), [1.0], atol=1e-6)


def _build_tiny_classifier(path, in_dim=4, n_classes=3):
    """Build a minimal ONNX classifier: data -> Relu (the embedding) -> Gemm.

    The Relu output is the penultimate tensor that add_embedding_output should
    auto-detect as the embedding.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    relu = helper.make_node("Relu", ["data"], ["feat"], name="relu")
    # transB=1: weight is [n_classes, in_dim] (like torch fc.weight), so the
    # Gemm computes feat[batch,in_dim] @ W^T -> [batch, n_classes].
    gemm = helper.make_node("Gemm", ["feat", "W", "b"], ["output"], name="fc", transB=1)

    w = numpy_helper.from_array(
        np.ones((n_classes, in_dim), dtype=np.float32), name="W"
    )
    b = numpy_helper.from_array(np.zeros((n_classes,), dtype=np.float32), name="b")

    inp = helper.make_tensor_value_info(
        "data", TensorProto.FLOAT, ["batch_size", in_dim]
    )
    out = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch_size", n_classes]
    )
    graph = helper.make_graph([relu, gemm], "tiny", [inp], [out], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    # Pin a conservative IR version: newer onnx defaults to IR 13, but the
    # onnxruntime on CI may only support up to IR 11. Opset 13 needs IR >= 7.
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


class TestAddEmbeddingOutput:
    """Test the ONNX graph surgery in add_embedding_output."""

    def test_auto_detect_embedding_tensor(self, tmp_path):
        import onnx

        from ifcb_infer.add_embedding_output import detect_embedding_tensor

        src = str(tmp_path / "tiny.onnx")
        _build_tiny_classifier(src)
        model = onnx.load(src)
        assert detect_embedding_tensor(model.graph) == "feat"

    def test_adds_second_output_and_runs(self, tmp_path):
        import onnxruntime as ort

        from ifcb_infer.add_embedding_output import add_embedding_output

        src = str(tmp_path / "tiny.onnx")
        dst = str(tmp_path / "tiny_emb.onnx")
        _build_tiny_classifier(src, in_dim=4, n_classes=3)
        add_embedding_output(src, dst)

        sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
        out_names = [o.name for o in sess.get_outputs()]
        assert len(out_names) == 2
        assert out_names[1] == "feat"

        x = np.array([[1.0, -2.0, 3.0, 4.0]], dtype=np.float32)
        logits, emb = sess.run(None, {"data": x})
        assert logits.shape == (1, 3)
        assert emb.shape == (1, 4)
        # embedding is Relu(data): negatives clamped to 0
        np.testing.assert_array_equal(emb, np.array([[1.0, 0.0, 3.0, 4.0]]))

    def test_idempotent_when_already_output(self, tmp_path):
        import onnx

        from ifcb_infer.add_embedding_output import add_embedding_output

        src = str(tmp_path / "tiny.onnx")
        dst = str(tmp_path / "tiny_emb.onnx")
        dst2 = str(tmp_path / "tiny_emb2.onnx")
        _build_tiny_classifier(src)
        add_embedding_output(src, dst)
        # Re-running on the already-modified model must not add a duplicate.
        add_embedding_output(dst, dst2, tensor_name="feat")
        model = onnx.load(dst2)
        assert [o.name for o in model.graph.output] == ["output", "feat"]

    def test_explicit_tensor_name_override(self, tmp_path):
        import onnx

        from ifcb_infer.add_embedding_output import add_embedding_output

        src = str(tmp_path / "tiny.onnx")
        dst = str(tmp_path / "tiny_emb.onnx")
        _build_tiny_classifier(src)
        add_embedding_output(src, dst, tensor_name="feat")
        model = onnx.load(dst)
        assert "feat" in [o.name for o in model.graph.output]


class _FakeSession:
    def __init__(self, n_outputs, output_shapes=None):
        output_shapes = output_shapes or [None] * n_outputs
        self._outs = [
            type("O", (), {"name": f"out{i}", "shape": output_shapes[i]})()
            for i in range(n_outputs)
        ]

    def get_outputs(self):
        return self._outs


class TestResolveEmitEmbeddings:
    def test_off_by_default(self):
        args = type("Args", (), {"embeddings": False})()
        assert resolve_emit_embeddings(args, _FakeSession(1)) is False
        assert resolve_emit_embeddings(args, _FakeSession(2)) is False

    def test_on_with_dual_output_model(self):
        args = type("Args", (), {"embeddings": True})()
        assert resolve_emit_embeddings(args, _FakeSession(2)) is True

    def test_raises_on_single_output_model(self):
        args = type("Args", (), {"embeddings": True})()
        with pytest.raises(ValueError, match="single output"):
            resolve_emit_embeddings(args, _FakeSession(1))


class TestValidateScoreOutput:
    def setup_method(self):
        self.args = type("Args", (), {})()
        self.args.outfile = "{BIN}.csv"
        self.args.classes = ["class_a", "class_b"]
        self.args.embeddings_only = False

    def test_h5_requires_classes_before_inference(self):
        self.args.outfile = "{BIN}.h5"
        self.args.classes = None
        with pytest.raises(ValueError, match="requires --classes"):
            validate_score_output_args(self.args)

    def test_h5_requires_readable_classes_file_before_inference(self):
        self.args.outfile = "{BIN}.h5"
        self.args.classes = "missing.classes"
        with pytest.raises(ValueError, match="readable class list"):
            validate_score_output_args(self.args)

    def test_h5_requires_h5py_before_inference(self, mocker):
        self.args.outfile = "{BIN}.h5"

        def fake_find_spec(name):
            if name == "h5py":
                return None
            return object()

        mocker.patch("importlib.util.find_spec", side_effect=fake_find_spec)
        with pytest.raises(ImportError, match="requires h5py"):
            validate_score_output_args(self.args)

    def test_embeddings_only_does_not_validate_score_h5(self):
        self.args.outfile = "{BIN}.h5"
        self.args.classes = None
        self.args.embeddings_only = True
        validate_score_output_args(self.args)

    def test_known_model_class_count_mismatch_fails_before_bins(self):
        self.args.outfile = "{BIN}.h5"
        session = _FakeSession(1, output_shapes=[[None, 3]])
        with pytest.raises(ValueError, match="2 labels.*3 classes"):
            validate_score_output_model(self.args, session)

    def test_unknown_model_class_count_is_checked_by_writer_later(self):
        self.args.outfile = "{BIN}.h5"
        session = _FakeSession(1, output_shapes=[["batch", "classes"]])
        validate_score_output_model(self.args, session)


class TestWriteEmbeddings:
    def setup_method(self):
        self.args = type("Args", (), {})()
        self.args.outdir = "./outputs"
        self.args.run_date_str = "2025-01-15"
        self.args.model_name = "test_model"
        self.args.embeddings_outfile = "{MODEL_NAME}/{SUBPATH}/{BIN}.emb.parquet"

    def test_embedding_output_path(self):
        result = get_embedding_output_path(
            self.args, "test_bin", "MVCO/2023/D20230108/test_bin"
        )
        assert result == os.path.normpath(
            "./outputs/test_model/MVCO/2023/D20230108/test_bin.emb.parquet"
        )

    def test_writes_parquet_float16_with_pids(self, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        self.args.outdir = str(tmp_path)
        pids = ["pidA", "pidB"]
        emb = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        write_embeddings(self.args, "test_bin", pids, emb)

        outpath = get_embedding_output_path(self.args, "test_bin")
        assert os.path.exists(outpath)
        table = pq.read_table(outpath)
        assert table.column_names == ["pid", "embedding"]
        assert table.column("pid").to_pylist() == pids
        emb_back = np.array(table.column("embedding").to_pylist())
        assert emb_back.shape == (2, 3)
        assert (
            table.schema.field("embedding").type.value_type
            == __import__("pyarrow").float16()
        )
        np.testing.assert_array_equal(emb_back, emb)

    def test_none_matrix_writes_nothing(self, tmp_path):
        self.args.outdir = str(tmp_path)
        write_embeddings(self.args, "test_bin", [], None)
        assert not os.path.exists(get_embedding_output_path(self.args, "test_bin"))


class TestEmbeddingArgparse:
    def test_embeddings_flags_default_off(self):
        parser = argparse_init()
        args = parser.parse_args(["model.onnx", "bins/"])
        assert args.embeddings is False
        assert args.embeddings_only is False
        assert args.embeddings_outfile == "{MODEL_NAME}/{SUBPATH}/{BIN}.emb.parquet"

    def test_embeddings_only_implies_embeddings(self, mocker):
        mock_datetime = mocker.patch("datetime.datetime")
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T14:30:45"
        parser = argparse_init()
        args = parser.parse_args(["--embeddings-only", "model.onnx", "bins/"])
        args.BINS = []
        argparse_runtime_args(args)
        assert args.embeddings is True


class TestWriteOutput:
    def setup_method(self):
        self.args = type("Args", (), {})()
        self.args.outdir = "./outputs"
        self.args.cmd_timestamp = "2025-01-15T14:30:45+00:00"
        self.args.run_date_str = "2025-01-15"
        self.args.model_name = "test_model"
        self.args.outfile = "{MODEL_NAME}/{SUBPATH}/{BIN}.csv"
        self.args.classes = ["class_a", "class_b"]
        self.args.skip_ensure_softmax = False

    def test_csv_default_unchanged(self, tmp_path):
        self.args.outdir = str(tmp_path)
        pids = ["pidA", "pidB"]
        # already-softmaxed rows pass through unchanged
        scores = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32)
        write_output(self.args, "test_bin", pids, scores)

        outpath = get_output_path(self.args, "test_bin")
        assert os.path.exists(outpath)
        with open(outpath) as f:
            lines = f.read().splitlines()
        assert lines[0] == "pid,class_a,class_b"
        assert lines[1].startswith("pidA,")
        assert lines[2].startswith("pidB,")

    def test_parquet_written_when_extension_parquet(self, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        self.args.outdir = str(tmp_path)
        self.args.outfile = "{MODEL_NAME}/{SUBPATH}/{BIN}.parquet"
        pids = ["pidA", "pidB"]
        scores = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32)
        write_output(self.args, "test_bin", pids, scores)

        outpath = get_output_path(self.args, "test_bin")
        assert os.path.exists(outpath)
        table = pq.read_table(outpath)
        assert table.column_names == ["pid", "class_a", "class_b"]
        assert table.column("pid").to_pylist() == pids
        back = np.array(
            [table.column("class_a").to_pylist(), table.column("class_b").to_pylist()]
        ).T
        np.testing.assert_allclose(back, scores, rtol=1e-6)

    def test_parquet_softmax_applied(self, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        self.args.outdir = str(tmp_path)
        self.args.outfile = "{BIN}.parquet"
        pids = ["pidA"]
        # logits (not softmaxed) -> ensure_softmax should normalize rows to sum 1
        logits = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)
        self.args.classes = ["a", "b", "c"]
        write_output(self.args, "test_bin", pids, logits)

        outpath = get_output_path(self.args, "test_bin")
        table = pq.read_table(outpath)
        row = np.array([table.column(c).to_pylist()[0] for c in ["a", "b", "c"]])
        np.testing.assert_allclose(row.sum(), 1.0, atol=1e-5)

    def test_parquet_fallback_column_names_without_classes(self, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        self.args.outdir = str(tmp_path)
        self.args.outfile = "{BIN}.parquet"
        self.args.classes = None
        pids = ["pidA"]
        scores = np.array([[0.7, 0.3]], dtype=np.float32)
        write_output(self.args, "test_bin", pids, scores)

        outpath = get_output_path(self.args, "test_bin")
        table = pq.read_table(outpath)
        assert table.column_names == ["pid", "score_0", "score_1"]

    def test_h5_written_when_extension_h5(self, tmp_path):
        pytest.importorskip("h5py")
        import h5py as h5

        self.args.outdir = str(tmp_path)
        self.args.outfile = "{BIN}_class.h5"
        bin_id = "D20250503T073255_IFCB188"
        pids = [f"{bin_id}_00002", f"{bin_id}_00003"]
        scores = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32)
        write_output(self.args, bin_id, pids, scores)

        outpath = get_output_path(self.args, bin_id)
        assert os.path.exists(outpath)
        with h5.File(outpath, "r") as f:
            assert set(f.keys()) == {
                "metadata",
                "output_classes",
                "output_scores",
                "class_labels",
                "roi_numbers",
            }
            metadata = f["metadata"]
            assert metadata.attrs["version"] == "v3"
            assert metadata.attrs["model_id"] == "test_model"
            assert metadata.attrs["timestamp"] == self.args.cmd_timestamp
            assert metadata.attrs["bin_id"] == bin_id

            assert f["output_classes"].compression == "gzip"
            assert f["output_classes"].dtype == np.dtype("float16")
            np.testing.assert_array_equal(
                f["output_classes"][:], np.array([0, 1], dtype=np.float16)
            )

            assert f["output_scores"].compression == "gzip"
            assert f["output_scores"].dtype == np.dtype("float16")
            np.testing.assert_allclose(f["output_scores"][:], scores.astype(np.float16))

            assert f["class_labels"].compression == "gzip"
            class_labels = [
                label.decode() if isinstance(label, bytes) else label
                for label in f["class_labels"][:]
            ]
            assert class_labels == ["class_a", "class_b"]

            assert f["roi_numbers"].compression == "gzip"
            assert f["roi_numbers"].dtype == np.dtype("uint32")
            np.testing.assert_array_equal(
                f["roi_numbers"][:], np.array([2, 3], dtype=np.uint32)
            )

    def test_h5_roi_numbers_support_large_targets(self, tmp_path):
        pytest.importorskip("h5py")
        import h5py as h5

        self.args.outdir = str(tmp_path)
        self.args.outfile = "{BIN}_class.h5"
        bin_id = "D20250503T073255_IFCB188"
        pids = [f"{bin_id}_70000"]
        scores = np.array([[0.7, 0.3]], dtype=np.float32)
        write_output(self.args, bin_id, pids, scores)

        outpath = get_output_path(self.args, bin_id)
        with h5.File(outpath, "r") as f:
            assert f["roi_numbers"].dtype == np.dtype("uint32")
            np.testing.assert_array_equal(
                f["roi_numbers"][:], np.array([70000], dtype=np.uint32)
            )

    def test_h5_requires_classes(self, tmp_path):
        self.args.outdir = str(tmp_path)
        self.args.outfile = "{BIN}.h5"
        self.args.classes = None
        scores = np.array([[0.7, 0.3]], dtype=np.float32)
        with pytest.raises(ValueError, match="requires --classes"):
            write_output(
                self.args,
                "D20250503T073255_IFCB188",
                ["D20250503T073255_IFCB188_00002"],
                scores,
            )

    def test_h5_requires_roi_pids(self, tmp_path):
        self.args.outdir = str(tmp_path)
        self.args.outfile = "{BIN}.h5"
        scores = np.array([[0.7, 0.3]], dtype=np.float32)
        with pytest.raises(ValueError, match="requires IFCB ROI IDs"):
            write_output(self.args, "test_bin", ["pidA"], scores)

    def test_none_matrix_writes_nothing(self, tmp_path):
        self.args.outdir = str(tmp_path)
        write_output(self.args, "test_bin", [], None)
        assert not os.path.exists(get_output_path(self.args, "test_bin"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
