import copy
from typing import NamedTuple, Tuple, Optional
from glob import glob
import functools
from pathlib import Path

from .pinyin import PinyinInput
import librosa

import pandas as pd

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


@functools.lru_cache(1, typed=True)
def load_audio(root: Path, relative_path: str) -> Tuple[torch.Tensor, int]:
    p = root / "clips" / relative_path
    return torchaudio.load(p)


class CommonVoiceRaw(NamedTuple):
    speaker_id: str
    waveform: torch.Tensor
    sample_rate: int
    sentence: str


class CommonVoiceRawDataset(Dataset):
    """Raw data straight from disk"""

    def __init__(self, root, table):
        """
        :param root: Path to dataset
        :param table: Pandas table path relative to root
        """
        root = Path(root)
        self.root = root
        self.table_path = table
        self.table = pd.read_table(root / table)
        self.speakers = {}
        for i, s in enumerate(self.table.client_id.unique()):
            self.speakers[s] = i

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx) -> CommonVoiceRaw:
        entry = self.table.iloc[idx]
        waveform, sample_rate = load_audio(self.root, entry.path)
        return CommonVoiceRaw(
            entry.client_id,
            waveform.detach(),
            sample_rate, entry.sentence)


class CommonVoiceEntry(NamedTuple):
    speaker_id: torch.Tensor
    # waveform: torch.Tensor
    mel: torch.Tensor
    specgram: torch.Tensor
    phoneme: torch.Tensor
    length: int
    frames: int
    # sentence: str


class CommonVoiceDataset(Dataset):
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
        """
        :param root: Path to dataset
        :param table: Table path relative to root
        """
        root = Path(root)
        self.root = root
        self.table_path = table
        self.pinyin = PinyinInput()
        self.raw_ds = CommonVoiceRawDataset(root, table)
        self.raw_rate = raw_rate
        self.train_rate = train_rate
        self.n_fft = n_fft
        self.win_ms = win_ms
        if win_length is None:
            win_length = int(win_ms * train_rate // 1000)
        self.win_length = win_length
        self.hop_ms = hop_ms
        if hop_length is None:
            hop_length = int(hop_ms * train_rate // 1000)
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        if f_max is None:
            # This is the Nyquist limit
            self.f_max = min(7600.0, train_rate / 2)
        self.f_max = f_max
        # Downsample transformer
        if raw_rate != train_rate:
            self.downsample = T.Resample(
                orig_freq=self.raw_rate, new_freq=self.train_rate)
        else:
            self.downsample = None
        self.vad = T.Vad(raw_rate)
        # Mel Spectrogram transformer
        self.mel = T.MelSpectrogram(
            sample_rate=self.train_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        self.spec = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )

    def __getitem__(self, idx) -> Optional[CommonVoiceEntry]:
        raw = self.raw_ds[idx]
        trimmed, idx = librosa.effects.trim(raw.waveform.squeeze(0),
                                            top_db=40, frame_length=256, hop_length=64)
        trimmed = trimmed.unsqueeze(0)
        wav_min = trimmed.min(1).values
        wav_max = trimmed.max(1).values
        normalized = trimmed / max(abs(wav_min), abs(wav_max)) * 0.99
        if self.downsample is not None:
            downsampled = self.downsample(normalized)
        else:
            downsampled = normalized
        try:
            phoneme, l = self.pinyin([raw.sentence])
        except ValueError:
            return None
        assert len(l) == 1, f"{l}"
        mel = torch.log(self.mel(downsampled) + 1e-9).detach()
        entry = CommonVoiceEntry(
            speaker_id=torch.tensor(bytearray.fromhex(
                raw.speaker_id)),
            mel=mel,
            specgram=torch.log(self.spec(downsampled) + 1e-9).detach(),
            phoneme=torch.from_numpy(phoneme),
            length=l[0],
            frames=mel.size(-1)
        )
        return entry

    def __len__(self) -> int:
        return len(self.raw_ds)
    
    @staticmethod
    def collate_fn(batch):
        return [b for b in batch if b is not None]
