#!/usr/bin/env python3

import datetime as dt
import os
import tempfile

import numpy as np
import pytest

from ifcb_infer.cli import (
    argparse_init,
    argparse_runtime_args,
    ensure_softmax,
    get_output_path,
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
            {"pid": "bin1", "hdr": "dir1/bin1.hdr", "adc": "dir1/bin1.adc", "roi": "dir1/bin1.roi"}
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
