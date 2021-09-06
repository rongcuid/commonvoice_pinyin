import functools
from pathlib import Path
from typing import Optional
from diskcache import *
from torch.utils.data.dataset import Dataset

from . import common_voice as cv

cache = FanoutCache(directory="cache/",
                    shards=64,
                    timeout=1,
                    size_limit=3e11,
                    disk=Disk,
                    disk_min_file_size=4194304)


@functools.lru_cache(1, typed=True)
def get_commonvoice_ds(root: Path, table: str,
                       raw_rate: int = 48000,
                       train_rate: int = 3000,
                       n_fft: int = 400,
                       win_ms: float = 50.0,
                       win_length: Optional[float] = None,
                       hop_ms: float = 12.5,
                       hop_length: Optional[float] = None,
                       n_mels: int = 80,
                       f_min: float = 125.0,
                       f_max: Optional[float] = None):
    return cv.CommonVoiceDataset(root, table,
                                 raw_rate,
                                 train_rate,
                                 n_fft,
                                 win_ms,
                                 win_length,
                                 hop_ms,
                                 hop_length,
                                 n_mels,
                                 f_min,
                                 f_max)


@cache.memoize(typed=True)
def get_commonvoice_entry_from_idx(
        idx,
        root: Path, table: str,
        raw_rate: int = 48000,
        train_rate: int = 3000,
        n_fft: int = 400,
        win_ms: float = 50.0,
        win_length: Optional[float] = None,
        hop_ms: float = 12.5,
        hop_length: Optional[float] = None,
        n_mels: int = 80,
        f_min: float = 125.0,
        f_max: Optional[float] = None):
    return get_commonvoice_ds(
        root, table,
        raw_rate,
        train_rate,
        n_fft,
        win_ms,
        win_length,
        hop_ms,
        hop_length,
        n_mels,
        f_min,
        f_max)[idx]


class CommonVoiceDatasetCached(Dataset):
    def __init__(self,
                 root, table,
                 raw_rate: int = 48000,
                 train_rate: int = 3000,
                 n_fft: int = 400,
                 win_ms: float = 50.0,
                 win_length: Optional[float] = None,
                 hop_ms: float = 12.5,
                 hop_length: Optional[float] = None,
                 n_mels: int = 80,
                 f_min: float = 125.0,
                 f_max: Optional[float] = None
                 ):
        self.root = root
        self.table = table
        self.raw_rate = raw_rate
        self.train_rate = train_rate
        self.n_fft = n_fft
        self.win_ms = win_ms
        self.win_length = win_length
        self.hop_ms = hop_ms
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

    def __len__(self):
        return len(get_commonvoice_ds(
            self.root,
            self.table,
            self.raw_rate,
            self.train_rate,
            self.n_fft,
            self.win_ms,
            self.win_length,
            self.hop_ms,
            self.hop_length,
            self.n_mels,
            self.f_min,
            self.f_max))

    def __getitem__(self, idx):
        return get_commonvoice_entry_from_idx(
            idx,
            self.root,
            self.table,
            self.raw_rate,
            self.train_rate,
            self.n_fft,
            self.win_ms,
            self.win_length,
            self.hop_ms,
            self.hop_length,
            self.n_mels,
            self.f_min,
            self.f_max)
