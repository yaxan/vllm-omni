# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Configuration for LeapCache. One knob; everything else is fixed here."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vllm_omni.diffusion.data import DiffusionCacheConfig

# Pipeline attributes set by LeapCacheBackend.enable().
LEAP_CACHE_CONFIG_ATTR = "_leapcache_config"
LEAP_CACHE_RUNTIME_ATTR = "_leapcache_runtime"

# The tolerance multiplier is 1 down to SIGMA_PHASE and grows to this value at SIGMA_LATE and below.
TOLERANCE_RATIO = 2.211180530344344

# The tail: the negative pass runs only while lo <= timestep <= hi, unless the request sets its own interval.
GUIDANCE_INTERVAL: tuple[float, float] = (600.0, 1000.0)


# Noise levels where the warm-up ends, the leap phase ends, and the tolerance stops growing. They were tuned once on
# the 40-step shift-5 Wan2.1 schedule (its steps 7, 18 and 31) and rounded; the rounded values land on the same steps.
SIGMA_WARMUP, SIGMA_PHASE, SIGMA_LATE = 0.96, 0.86, 0.60


@dataclass(frozen=True)
class LeapCacheConfig:
    """Runtime config for LeapCache.

    ``threshold`` is the predicted change in the model output, relative to that output, accepted
    before the model runs again in the early phase; the late phase accepts ``TOLERANCE_RATIO`` times it.
    """

    threshold: float = 0.044

    def __post_init__(self) -> None:
        threshold = self.threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError(f"leap_threshold must be a finite float >= 0, got {threshold!r}")

    @classmethod
    def from_diffusion_cache_config(cls, config: DiffusionCacheConfig) -> LeapCacheConfig:
        if config._extra_params:
            raise ValueError(f"leap_cache takes only leap_threshold, got {sorted(config._extra_params)}")
        return cls(threshold=config.leap_threshold)
