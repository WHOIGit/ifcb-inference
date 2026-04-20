import argparse
import datetime as dt
import json
import os

import ifcb
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
        help="Path to row-delimited classlist file. Required for output-csv's headers",
    )
    parser.add_argument("--outdir", default="./outputs", help='Default is "./outputs"')
    parser.add_argument(
        "--outfile",
        default="{MODEL_NAME}/{SUBPATH}.csv",
        help='Output filename pattern. Tokens: {MODEL_NAME}, {RUN_DATE}, {SUBPATH}. Default is "{MODEL_NAME}/{SUBPATH}.csv"',
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

    return parser


def get_providers(args):
    if args.cpuonly:
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    return [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]


def argparse_runtime_args(args):
    args.cmd_timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    args.run_date_str, args.run_time_str = args.cmd_timestamp.split("T")

    args.model_name = os.path.splitext(os.path.basename(args.MODEL))[0]

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
            leaf_dirs = ifcb.data.files.list_data_dirs(bin_thing, blacklist=_DIR_BLACKLIST)
            bin_paths = []
            for leaf_dir in leaf_dirs:
                dd = ifcb.DataDirectory(leaf_dir)
                bin_paths.extend(binobj.fileset.basepath for binobj in dd)
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


def get_output_path(args, bin_id, bin_relative_path=None):
    outpath = os.path.join(args.outdir, args.outfile)
    subpath = bin_relative_path if bin_relative_path is not None else bin_id
    return outpath.format(
        RUN_DATE=args.run_date_str, MODEL_NAME=args.model_name, SUBPATH=subpath
    )


def write_output(args, bin_id, pids, score_matrix, bin_relative_path=None):
    outpath = get_output_path(args, bin_id, bin_relative_path)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        if args.classes:
            f.write(",".join(["pid"] + args.classes) + "\n")
        if score_matrix is not None:
            for pid, score_row in zip(pids, score_matrix):
                str_row = ",".join(map(str, [pid] + score_row.tolist()))
                f.write(str_row + "\n")
        else:
            print(f"Warning: No data processed for bin {bin_id}")


def main():
    ort.preload_dlls(directory="")

    parser = argparse_init()
    args = parser.parse_args()
    argparse_runtime_args(args)

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
