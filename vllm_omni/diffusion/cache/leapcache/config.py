# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Configuration for LeapCache. One knob; everything else is fixed here."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler

# Pipeline attributes set by LeapCacheBackend.enable().
LEAP_CACHE_CONFIG_ATTR = "_leapcache_config"
LEAP_CACHE_RUNTIME_ATTR = "_leapcache_runtime"

# The tolerance multiplier is 1 down to SIGMA_PHASE and grows to this value at SIGMA_LATE and below.
TOLERANCE_RATIO = 2.211180530344344

# The tail: the negative pass runs only while lo <= timestep <= hi, unless the request sets its own interval.
GUIDANCE_INTERVAL: tuple[float, float] = (600.0, 1000.0)


def _reference_sigmas(*steps: int) -> tuple[float, ...]:
    """Sigma at the given steps of the 40-step shift-5 schedule the profile was tuned on."""
    reference = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, prediction_type="flow_prediction")
    reference.set_timesteps(40, device="cpu", shift=5.0)
    return tuple(float(reference.sigmas[step]) for step in steps)


# Noise levels where the warm-up ends, the leap phase ends, and the tolerance stops growing.
SIGMA_WARMUP, SIGMA_PHASE, SIGMA_LATE = _reference_sigmas(7, 18, 31)


@dataclass(frozen=True)
class LeapCacheConfig:
    """Runtime config for LeapCache.

    ``threshold`` is the predicted change in the model output, relative to that output,
    accepted before the model runs again.
    """

    threshold: float = 0.064

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
