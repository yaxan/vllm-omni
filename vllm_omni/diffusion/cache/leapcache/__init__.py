# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
LeapCache: a step-skipping cache for Wan2.1 text-to-video (14B and 1.3B).

It keeps EasyCache's skip-or-run rule (Zhou et al., 2025) and changes what happens
around it. Early in the run it never reuses an output: trial solver steps pick the next
step that needs the model and the steps in between leave the schedule. Late in the run it
visits every step, holds the last output on skipped steps, and replays the held stretch
with a blend of the two real outputs once the model runs again. Below timestep 600 only
the prompt pass runs, through the pipeline's ``guidance_interval`` option, unless the
request sets its own interval.

Usage:
    from vllm_omni import Omni

    omni = Omni(
        model="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        cache_backend="leap_cache",
        cache_config={"leap_threshold": 0.044},
    )
"""

from vllm_omni.diffusion.cache.leapcache.backend import (
    LeapCacheBackend,
    get_leapcache_config,
    get_leapcache_runtime,
)
from vllm_omni.diffusion.cache.leapcache.config import (
    GUIDANCE_INTERVAL,
    LEAP_CACHE_CONFIG_ATTR,
    LEAP_CACHE_RUNTIME_ATTR,
    LeapCacheConfig,
)
from vllm_omni.diffusion.cache.leapcache.runtime import LeapCacheRuntime, LeapCacheStats
from vllm_omni.diffusion.cache.leapcache.state import CacheState

__all__ = [
    "GUIDANCE_INTERVAL",
    "LEAP_CACHE_CONFIG_ATTR",
    "LEAP_CACHE_RUNTIME_ATTR",
    "CacheState",
    "LeapCacheBackend",
    "LeapCacheConfig",
    "LeapCacheRuntime",
    "LeapCacheStats",
    "get_leapcache_config",
    "get_leapcache_runtime",
]
