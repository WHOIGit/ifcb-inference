import os

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from ifcb_infer.cli import get_output_path, pad_batch, write_output
from ifcb_infer.datasets_torch import IfcbBinsDataset


def main(args):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(
        args.MODEL, sess_options=sess_options, providers=providers
    )

    input0 = ort_session.get_inputs()[0]
    model_batch = input0.shape[0]
    img_size = input0.shape[-1]
    input_type = getattr(torch, input0.type[7:-1])

    dynamic_batching = True
    if isinstance(model_batch, str):
        assert (
            args.batch is not None
        ), "Must specify inference batch size for dynamically batched MODEL"
        inference_batchsize = args.batch
    else:
        assert (
            args.batch is None or model_batch == args.batch
        ), "MODEL is statically batched, inference batch size cannot be adjusted"
        dynamic_batching = False
        inference_batchsize = model_batch

    transformer = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(input_type, scale=True),
    ])

    pbar = tqdm(
        args.BINS,
        desc=f"batchsize={inference_batchsize}",
        unit="bins",
        dynamic_ncols=False,
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for bin_accessor in pbar:
        img_pids = []
        score_matrix = None

        root_dir = os.path.dirname(bin_accessor)
        bin_id = os.path.basename(bin_accessor)

        bin_relative_path = None
        input_dir = args.bin_to_input_dir.get(bin_accessor)
        if input_dir and os.path.isdir(input_dir):
            try:
                rel_path = os.path.relpath(bin_accessor, input_dir)
                bin_name = os.path.basename(bin_accessor)
                if rel_path != bin_name:
                    rel_dir = os.path.dirname(rel_path)
                    bin_relative_path = os.path.join(rel_dir, bin_name)
                else:
                    bin_relative_path = bin_name
            except ValueError:
                bin_relative_path = None

        dataset = IfcbBinsDataset(
            bin_dirs=[root_dir],
            bin_whitelist=[bin_id],
            transform=transformer,
            with_sources=True,
            shuffle=False,
            use_len=False,
        )
        dataloader = DataLoader(
            dataset, batch_size=inference_batchsize, num_workers=0, drop_last=False
        )
        bin_pid = list(dataset.iter_binfilesets())[0].pid.pid

        expected_output_path = get_output_path(args, bin_pid, bin_relative_path)
        if os.path.exists(expected_output_path):
            pbar.set_description(
                f"batchsize={inference_batchsize} (skipping {bin_pid})"
            )
            continue

        for batch_tuple in dataloader:
            batch, batch_pids = batch_tuple[0], batch_tuple[1]
            batch = batch.numpy()
            size_of_batch = batch.shape[0]

            if dynamic_batching or size_of_batch == inference_batchsize:
                outputs = ort_session.run(None, {input0.name: batch})
            else:
                batch = pad_batch(batch, inference_batchsize)
                outputs = ort_session.run(None, {input0.name: batch})
                outputs = [output[:size_of_batch] for output in outputs]

            batch_score_matrix = outputs[0]
            if score_matrix is None:
                score_matrix = batch_score_matrix
            else:
                score_matrix = np.concatenate(
                    [score_matrix, batch_score_matrix], axis=0
                )
            img_pids.extend(batch_pids)

        write_output(args, bin_pid, img_pids, score_matrix, bin_relative_path)
