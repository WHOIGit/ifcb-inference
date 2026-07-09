import argparse
import datetime as dt
import importlib.util
import json
import os

import ifcbkit
import numpy as np
import onnxruntime as ort


def argparse_init(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Perform onnx-model inference on IFCB bins"
        )

    parser.add_argument("MODEL", help="Path to a previously-trained model file")
    parser.add_argument(
        "BINS",
        nargs="+",
        help="Bin(s) to be classified. Can be a directory, bin-path, or list-file thereof",
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=int,
        help="Specify inference batchsize (for dynamically-batched MODEL only)",
    )
    parser.add_argument(
        "--classes",
        "-c",
        help="Path to row-delimited classlist file. Required for output-csv's headers",
    )
    parser.add_argument(
        "--outdir", "-o", default="./outputs", help='Default is "./outputs"'
    )
    parser.add_argument(
        "--outfile",
        default="{MODEL_NAME}/{SUBPATH}/{BIN}.csv",
        help='Output filename pattern. Tokens: {MODEL_NAME}, {RUN_DATE}, {SUBPATH} (relative dir), {BIN} (bin name). Default is "{MODEL_NAME}/{SUBPATH}/{BIN}.csv"',
    )
    parser.add_argument(
        "--notorch",
        action="store_true",
        help="Force non-torch dataloader even if torch is available",
    )
    parser.add_argument(
        "--cpuonly",
        action="store_true",
        help="Force CPU-only inference, disabling CUDA even if available",
    )
    parser.add_argument(
        "--skip-ensure-softmax",
        action="store_true",
        help="Skip softmax normalization check on model output",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Emit penultimate-layer embedding vectors. Requires a MODEL whose "
        "graph exposes the embedding tensor (see add_embedding_output).",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Skip writing the score CSV; write only embeddings. Implies --embeddings.",
    )
    parser.add_argument(
        "--embeddings-outfile",
        default="{MODEL_NAME}/{SUBPATH}/{BIN}.emb.parquet",
        help="Embedding output filename pattern. Same tokens as --outfile. "
        'Default is "{MODEL_NAME}/{SUBPATH}/{BIN}.emb.parquet"',
    )

    return parser


def get_providers(args):
    available_providers = ort.get_available_providers()
    if args.cpuonly or "CUDAExecutionProvider" not in available_providers:
        return ["CPUExecutionProvider"]
    else:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def argparse_runtime_args(args):
    args.cmd_timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    args.run_date_str, args.run_time_str = args.cmd_timestamp.split("T")

    args.model_name = os.path.splitext(os.path.basename(args.MODEL))[0]

    if getattr(args, "embeddings_only", False):
        args.embeddings = True

    gpu_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    args.gpus = [int(gpu) for gpu in gpu_str.split(",") if gpu.strip()]

    if args.classes and os.path.isfile(args.classes):
        with open(args.classes) as f:
            content = f.read()
        if args.classes.endswith(".json") or content.lstrip().startswith("{"):
            mapping = json.loads(content)
            args.classes = [mapping[str(i)] for i in range(len(mapping))]
        else:
            args.classes = content.strip().splitlines()

    _DIR_BLACKLIST = ("bad", "skip", "beads", "temp", "data_temp")

    bins = []
    bin_to_input_dir = {}
    for bin_thing in args.BINS:
        if os.path.isdir(bin_thing):
            bin_paths = []
            for leaf_dir in ifcbkit.sync_list_data_dirs(
                bin_thing, exclude=_DIR_BLACKLIST
            ):
                dd = ifcbkit.SyncIfcbDataDirectory(leaf_dir)
                for entry in dd.list():
                    bin_paths.append(os.path.splitext(entry["hdr"])[0])
            bins.extend(bin_paths)
            for bin_path in bin_paths:
                bin_to_input_dir[bin_path] = bin_thing
        elif bin_thing.endswith(".txt") or bin_thing.endswith(".list"):
            with open(bin_thing, "r") as f:
                bin_list_from_file = f.read().splitlines()
            bins.extend(bin_list_from_file)
            for bin_path in bin_list_from_file:
                bin_to_input_dir[bin_path] = None
        else:
            bins.append(bin_thing)
            bin_to_input_dir[bin_thing] = None
    args.BINS = bins
    args.bin_to_input_dir = bin_to_input_dir


def softmax(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def is_row_softmaxed(row, tol=1e-5):
    if np.any(row < 0) or np.any(row > 1):
        return False
    return np.isclose(np.sum(row), 1.0, atol=tol)


def ensure_softmax(score_matrix):
    if not is_row_softmaxed(score_matrix[0]):
        return softmax(score_matrix, axis=1)
    return score_matrix


def pad_batch(batch: np.ndarray, target_batch_size: int):
    current_size = batch.shape[0]
    if current_size == target_batch_size:
        return batch
    elif current_size > target_batch_size:
        raise ValueError(
            f"Batch size {current_size} exceeds target size {target_batch_size}"
        )
    pad_size = target_batch_size - current_size
    pad_shape = (pad_size,) + batch.shape[1:]
    pad = np.zeros(pad_shape, dtype=batch.dtype)
    return np.concatenate([batch, pad], axis=0)


def _format_output_path(args, outfile, bin_id, bin_relative_path=None):
    full_subpath = bin_relative_path if bin_relative_path is not None else bin_id
    subpath_dir = os.path.dirname(full_subpath)
    bin_name = os.path.basename(full_subpath)
    outpath = os.path.join(args.outdir, outfile)
    result = outpath.format(
        RUN_DATE=args.run_date_str,
        MODEL_NAME=args.model_name,
        SUBPATH=subpath_dir,
        BIN=bin_name,
    )
    return os.path.normpath(result)


def get_output_path(args, bin_id, bin_relative_path=None):
    return _format_output_path(args, args.outfile, bin_id, bin_relative_path)


def get_embedding_output_path(args, bin_id, bin_relative_path=None):
    return _format_output_path(args, args.embeddings_outfile, bin_id, bin_relative_path)


def _is_parquet(path):
    return path.lower().endswith(".parquet")


def _is_h5(path):
    return path.lower().endswith(".h5")


def _writes_score_output(args):
    return not getattr(args, "embeddings_only", False)


def _score_output_is_h5(args):
    return _writes_score_output(args) and _is_h5(args.outfile)


def _score_output_is_parquet(args):
    return _writes_score_output(args) and _is_parquet(args.outfile)


def _writes_embedding_output(args):
    return getattr(args, "embeddings", False) or getattr(args, "embeddings_only", False)


def _has_class_labels(args):
    return isinstance(args.classes, (list, tuple)) and len(args.classes) > 0


def _require_dependency(module_name, extra_name, output_name):
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"{output_name} output requires {module_name}; install with the "
            f"[{extra_name}] extra"
        )


def _score_column_names(args, n_classes):
    if args.classes:
        if not isinstance(args.classes, (list, tuple)):
            raise ValueError("--classes must point to a readable class list file")
        if len(args.classes) != n_classes:
            raise ValueError(
                f"--classes has {len(args.classes)} labels, but model output has "
                f"{n_classes} classes"
            )
        return args.classes
    return [f"score_{i}" for i in range(n_classes)]


def validate_score_output_args(args):
    if _score_output_is_parquet(args):
        _require_dependency("pyarrow", "parquet", "Parquet score")

    if _writes_embedding_output(args):
        _require_dependency("pyarrow", "parquet", "Embedding")

    if not _score_output_is_h5(args):
        return

    if not _has_class_labels(args):
        raise ValueError(
            "H5 score output requires --classes with a readable class list"
        )

    _require_dependency("h5py", "h5", "H5 score")


def validate_score_output_model(args, ort_session):
    if not _writes_score_output(args) or not args.classes:
        return

    output_shape = getattr(ort_session.get_outputs()[0], "shape", None)
    if not output_shape:
        return

    n_classes = output_shape[-1]
    if isinstance(n_classes, int):
        _score_column_names(args, n_classes)


def _required_h5_class_labels(args, n_classes):
    if not _has_class_labels(args):
        raise ValueError(
            "H5 score output requires --classes with a readable class list"
        )
    return _score_column_names(args, n_classes)


def _roi_numbers_from_pids(pids):
    roi_numbers = []
    for pid in pids:
        try:
            _, target = ifcbkit.parse_roi_id(pid)
        except ValueError as exc:
            raise ValueError(
                "H5 score output requires IFCB ROI IDs with target numbers"
            ) from exc
        roi_numbers.append(target)
    return roi_numbers


def write_output(args, bin_id, pids, score_matrix, bin_relative_path=None):
    outpath = get_output_path(args, bin_id, bin_relative_path)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    if score_matrix is None:
        print(f"Warning: No data processed for bin {bin_id}")
        return

    pids = list(pids)

    if not args.skip_ensure_softmax:
        score_matrix = ensure_softmax(score_matrix)

    if _is_parquet(outpath):
        _write_scores_parquet(args, outpath, pids, score_matrix)
    elif _is_h5(outpath):
        _write_scores_h5(args, outpath, bin_id, pids, score_matrix)
    else:
        _write_scores_csv(args, outpath, pids, score_matrix)


def _write_scores_csv(args, outpath, pids, score_matrix):
    with open(outpath, "w") as f:
        if args.classes:
            f.write(",".join(["pid"] + args.classes) + "\n")
        for pid, score_row in zip(pids, score_matrix):
            str_row = ",".join(map(str, [pid] + score_row.tolist()))
            f.write(str_row + "\n")


def _write_scores_parquet(args, outpath, pids, score_matrix):
    # Imported lazily so CSV-only runs don't require pyarrow.
    import pyarrow as pa
    import pyarrow.parquet as pq

    score_matrix = np.ascontiguousarray(score_matrix.astype(np.float32))
    n_classes = score_matrix.shape[1]
    columns = {"pid": pa.array(list(pids), type=pa.string())}
    for i, name in enumerate(_score_column_names(args, n_classes)):
        columns[name] = pa.array(score_matrix[:, i], type=pa.float32())
    pq.write_table(pa.table(columns), outpath)


def _write_scores_h5(args, outpath, bin_id, pids, score_matrix):
    n_classes = score_matrix.shape[1]
    class_labels = _required_h5_class_labels(args, n_classes)
    roi_numbers = _roi_numbers_from_pids(pids)

    # Imported lazily so non-H5 runs don't require h5py.
    import h5py as h5

    output_classes = np.argmax(score_matrix, axis=1)
    with h5.File(outpath, "w") as f:
        meta = f.create_dataset("metadata", data=h5.Empty("f"))
        meta.attrs["version"] = "v3"
        meta.attrs["model_id"] = args.model_name
        meta.attrs["timestamp"] = args.cmd_timestamp
        meta.attrs["bin_id"] = bin_id
        f.create_dataset(
            "output_classes", data=output_classes, compression="gzip", dtype="float16"
        )
        f.create_dataset(
            "output_scores", data=score_matrix, compression="gzip", dtype="float16"
        )
        f.create_dataset(
            "class_labels",
            data=list(class_labels),
            compression="gzip",
            dtype=h5.string_dtype(),
        )
        f.create_dataset(
            "roi_numbers", data=roi_numbers, compression="gzip", dtype="uint32"
        )


def resolve_emit_embeddings(args, ort_session):
    """Decide whether to emit embeddings for this run, validating that the model
    actually exposes the embedding output when --embeddings was requested."""
    emit = args.embeddings and len(ort_session.get_outputs()) > 1
    if args.embeddings and not emit:
        raise ValueError(
            "--embeddings requested but MODEL exposes a single output. Run "
            "`python -m ifcb_infer.add_embedding_output` to add the embedding "
            "tensor to the model's graph outputs."
        )
    return emit


def write_embeddings(args, bin_id, pids, embedding_matrix, bin_relative_path=None):
    if embedding_matrix is None:
        print(f"Warning: No embeddings processed for bin {bin_id}")
        return

    # Imported lazily so non-embedding runs don't require pyarrow.
    import pyarrow as pa
    import pyarrow.parquet as pq

    outpath = get_embedding_output_path(args, bin_id, bin_relative_path)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    embedding_matrix = np.ascontiguousarray(embedding_matrix.astype(np.float16))
    n_rows, dim = embedding_matrix.shape
    embedding_col = pa.FixedSizeListArray.from_arrays(
        pa.array(embedding_matrix.reshape(-1), type=pa.float16()), dim
    )
    table = pa.table(
        {"pid": pa.array(list(pids), type=pa.string()), "embedding": embedding_col}
    )
    pq.write_table(table, outpath)


def main():
    # ort.preload_dlls(directory="") useful for TRT on Windows

    parser = argparse_init()
    args = parser.parse_args()
    argparse_runtime_args(args)
    validate_score_output_args(args)

    use_torch = False
    if not args.notorch:
        try:
            import torch  # noqa: F401

            use_torch = True
        except ImportError:
            pass

    if use_torch:
        from ifcb_infer.withtorch import main as torch_main

        torch_main(args)
    else:
        from ifcb_infer.sanstorch import main as notorch_main

        notorch_main(args)
