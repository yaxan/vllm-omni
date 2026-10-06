# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the two-video mode and the cache options of benchmarks/diffusion/quantization_quality.py."""

import sys

import imageio.v3 as iio
import lpips
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from benchmarks.diffusion import quantization_quality

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


class _MeanAbsDiff(nn.Module):
    """Stands in for the LPIPS network, so the test needs no pretrained weights."""

    def __init__(self, net):
        super().__init__()

    def forward(self, a, b):
        return (a - b).abs().mean()


@pytest.fixture(autouse=True)
def _cpu_lpips_stub(monkeypatch):
    monkeypatch.setattr(lpips, "LPIPS", _MeanAbsDiff)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _frames(count, height=16, width=32):
    # Values stay below 204 so the +51 offset used below cannot overflow.
    return np.random.default_rng(0).integers(0, 204, size=(count, height, width, 3), dtype=np.uint8)


def _png_dir(path, frames):
    path.mkdir()
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(path / f"frame_{i:03d}.png")
    return str(path)


def _compare(monkeypatch, capsys, baseline, variant):
    monkeypatch.setattr(sys, "argv", ["quantization_quality.py", "--compare", baseline, variant])
    quantization_quality.compare_videos(quantization_quality.parse_args())
    return capsys.readouterr().out


def test_png_frame_dirs_report_mean_worst_and_count(tmp_path, monkeypatch, capsys):
    frames = _frames(4)
    changed = frames.copy()
    changed[2] += 51  # 51/255 of the [0, 1] range is 0.4 of the [-1, 1] range

    out = _compare(monkeypatch, capsys, _png_dir(tmp_path / "a", frames), _png_dir(tmp_path / "b", changed))

    assert "Frames: 4 (32x16)" in out
    assert "Mean LPIPS (alex): 0.1000" in out
    assert "Worst frame: 0.4000 (frame 2)" in out


def test_mp4_files_are_decoded(tmp_path, monkeypatch, capsys):
    path = str(tmp_path / "clip.mp4")
    iio.imwrite(path, _frames(6), plugin="pyav", codec="libx264", fps=4)

    out = _compare(monkeypatch, capsys, path, path)

    assert "Frames: 6 (32x16)" in out
    assert "Mean LPIPS (alex): 0.0000" in out


@pytest.mark.parametrize(
    ("variant_shape", "message"),
    [
        ((3, 16, 32), r"4 frames of 32x16; .* 3 frames of 32x16"),
        ((4, 16, 48), r"4 frames of 32x16; .* 4 frames of 48x16"),
    ],
)
def test_mismatched_videos_are_rejected(tmp_path, monkeypatch, capsys, variant_shape, message):
    baseline = _png_dir(tmp_path / "a", _frames(4))
    variant = _png_dir(tmp_path / "b", _frames(*variant_shape))

    with pytest.raises(ValueError, match=message):
        _compare(monkeypatch, capsys, baseline, variant)


def test_benchmark_mode_takes_a_cache_backend_as_the_variant(monkeypatch):
    argv = ["quantization_quality.py", "--model", "m", "--cache-backend", "leap_cache"]
    monkeypatch.setattr(sys, "argv", argv + ["--cache-config", '{"leap_threshold": 0.032}'])
    args = quantization_quality.parse_args()
    assert (args.cache_backend, args.cache_config, args.quantization) == ("leap_cache", {"leap_threshold": 0.032}, [])

    monkeypatch.setattr(sys, "argv", ["quantization_quality.py", "--model", "m"])
    with pytest.raises(SystemExit):
        quantization_quality.parse_args()
