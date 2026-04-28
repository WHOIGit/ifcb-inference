import itertools
import os
import random
from functools import cache
from typing import Union

import humanize
import ifcbkit
import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import v2
from tqdm import tqdm

DEFAULT_BLACKLIST = ("bad", "skip", "beads", "temp", "data_temp")


class IfcbBinsDataset(IterableDataset):
    def __init__(
        self,
        bin_dirs: list,
        transform,
        with_sources: bool = True,
        shuffle: bool = True,
        bin_whitelist: list = None,
        bin_blacklist: list = None,
        dir_blacklist: list = DEFAULT_BLACKLIST,
        dir_whitelist: list = None,
        use_len: Union[bool, int] = False,
    ):
        self.bin_whitelist = bin_whitelist or []
        self.bin_blacklist = bin_blacklist or []
        self.dir_blacklist = dir_blacklist
        leaf_dirs = [
            ifcbkit.sync_list_data_dirs(bin_dir, exclude=dir_blacklist)
            for bin_dir in bin_dirs
        ]
        self.bin_dirs = sorted(
            itertools.chain.from_iterable(leaf_dirs), key=lambda p: os.path.basename(p)
        )  # flatten
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.bin_dirs)
        self.transform = (
            v2.Compose(transform) if isinstance(transform, list) else transform
        )
        self.with_sources = with_sources
        self.use_len = use_len
        if self.use_len is True:
            self.calculate_len()

    def _get_worker_chunk(self):
        worker_info = torch.utils.data.get_worker_info()
        dirs = list(self.bin_dirs)
        if worker_info is None:
            if self.shuffle:
                random.shuffle(dirs)
            return dirs
        else:
            N = worker_info.num_workers
            n, m = divmod(len(dirs), N)
            per_worker = [
                dirs[i * n + min(i, m) : (i + 1) * n + min(i + 1, m)] for i in range(N)
            ]
            chunk = per_worker[worker_info.id]
            if self.shuffle:
                random.shuffle(chunk)
            return chunk

    def __iter__(self):
        for entry in self.iter_binfilesets():
            yield from self.iter_bin_images(
                entry, transform=self.transform, with_sources=self.with_sources
            )

    def iter_binfilesets(self):
        for bin_dir in self._get_worker_chunk():
            dd = ifcbkit.SyncIfcbDataDirectory(bin_dir)
            for entry in dd.list():
                bin_pid = entry["pid"]
                if self.bin_whitelist and bin_pid not in self.bin_whitelist:
                    continue
                if bin_pid in self.bin_blacklist:
                    continue
                yield entry

    @staticmethod
    def iter_bin_images(entry, transform=None, with_sources=True):
        bin_pid = entry["pid"]
        with open(entry["adc"], "rb") as f:
            adc_bytes = f.read()
        with open(entry["roi"], "rb") as f:
            roi_bytes = f.read()
        images = ifcbkit.bin_images(bin_pid, adc_bytes, roi_bytes)
        for target_number, roi in images.items():
            target_pid = ifcbkit.add_target(bin_pid, target_number)
            img = roi.convert("RGB")
            if transform is not None:
                img = transform(img)
            if with_sources:
                yield img, target_pid
            else:
                yield img

    @cache
    def calculate_len(self):
        count_sum = 0
        bin_sum = 0
        pbar = tqdm(self.bin_dirs, desc="caching dataset length")
        for bin_dir in pbar:
            dd = ifcbkit.SyncIfcbDataDirectory(bin_dir)
            for entry in dd.list():
                bin_pid = entry["pid"]
                if self.bin_whitelist and bin_pid not in self.bin_whitelist:
                    continue
                if bin_pid in self.bin_blacklist:
                    continue
                bin_sum += 1
                bin_count = len(dd.list_images(bin_pid))
                count_sum += bin_count
            pbar.set_postfix(
                dict(BINs=humanize.intcomma(bin_sum), ROIs=humanize.intword(count_sum))
            )
        return count_sum

    def __len__(self):
        if self.use_len is True:
            return self.calculate_len()
        elif self.use_len:
            return self.use_len


def make_dataset(source, resize, img_norm=None, dtype=torch.float32):
    # create dataset
    whitelist = None
    blacklist = None
    # Formatting Dataset
    if os.path.isdir(source):
        root_dir = source
    elif os.path.isfile(source) and source.endswith(
        ".txt"
    ):  # TODO TEST: textfile bin run
        with open(source, "r") as f:
            bins = f.read().splitlines()
        root_dir = os.path.commonpath(bins)
        whitelist = bins
    else:  # single bin # TODO TEST: single bin run
        root_dir = os.path.dirname(source)
        bin_id = os.path.basename(source)
        whitelist = [bin_id]

    transforms = [
        v2.Resize((resize, resize)),
        v2.ToImage(),
        v2.ToDtype(dtype, scale=True),
    ]
    if img_norm:
        norm = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.insert(2, norm)
        raise NotImplementedError
    transform = v2.Compose(transforms)

    dataset = IfcbBinsDataset(
        bin_dirs=[root_dir],
        bin_whitelist=whitelist,
        bin_blacklist=blacklist,
        transform=transform,
        with_sources=True,
        shuffle=False,
        use_len=True,
    )

    return dataset
