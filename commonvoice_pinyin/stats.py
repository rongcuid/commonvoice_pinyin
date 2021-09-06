import sys
from pathlib import Path
from . import common_voice as cv
import argparse
from torch.utils.data import DataLoader


def stat(dl, name):
    max_chars = 0
    max_mel_frames = 0
    max_sg_frames = 0
    total = len(dl)
    perc = total // 100
    dec = total // 10
    print(f"{name} contains {total} samples")
    print(". = 1%; | = 10%; ! = bad sample")
    bad_samples = 0
    for i, d in enumerate(dl):
        if (i+1) % dec == 0:
            print("|", flush=True)
        elif (i+1) % perc == 0:
            print(".", end="", flush=True)
        if d is None:
            bad_samples += 1
            print("!", end="", flush=True)
            continue
        max_chars = max(d.length, max_chars)
        frames_mel = d.mel.size(-1)
        frames = d.specgram.size(-1)
        max_mel_frames = max(frames_mel, max_mel_frames)
        max_sg_frames = max(frames, max_sg_frames)
    print()
    print(f"{name} dataset max sentence length: {max_chars}")
    print(f"{name} dataset max frames: {max_mel_frames} == {max_sg_frames}")
    print(f"{name} had {bad_samples} invalid entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-dev", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("dataset_path", type=Path)
    args = parser.parse_args()
    if not args.skip_train:
        train = cv.CommonVoiceDataset(args.dataset_path, "train.tsv")
        stat(train, "Train")
    if not args.skip_dev:
        dev = cv.CommonVoiceDataset(args.dataset_path, "dev.tsv")
        stat(dev, "Validate")
    if not args.skip_test:
        test = cv.CommonVoiceDataset(args.dataset_path, "test.tsv")
        stat(test, "Test")
