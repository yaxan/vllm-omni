# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
LeapCache: a step-skipping cache for Wan2.1 text-to-video (14B and 1.3B).

It keeps EasyCache's skip-or-run rule (Zhou et al., 2025) and changes what happens
around it. Early in the run, skipped steps are dropped from the schedule, so the solver
takes one long step instead of reusing an old output. Late in the run, skipped steps keep
the last output. In both phases, when the model next runs, the clip is rewound to the last
model run and the skipped steps are redone with a blend of the old and the new output, at
no model cost. The solver then continues from the redone steps. Below timestep 600 only
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
