# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""LeapCache backend: attaches the config and the runtime to a Wan2.1 text-to-video pipeline."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.leapcache.config import (
    LEAP_CACHE_CONFIG_ATTR,
    LEAP_CACHE_RUNTIME_ATTR,
    LeapCacheConfig,
)
from vllm_omni.diffusion.cache.leapcache.runtime import LeapCacheRuntime

logger = init_logger(__name__)

_PARALLEL_DEGREES = (
    "cfg_parallel_size",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "ulysses_degree",
    "ring_degree",
    "allgather_degree",
)


def get_leapcache_config(pipeline: Any) -> LeapCacheConfig | None:
    config = getattr(pipeline, LEAP_CACHE_CONFIG_ATTR, None)
    return config if isinstance(config, LeapCacheConfig) else None


def get_leapcache_runtime(pipeline: Any) -> LeapCacheRuntime | None:
    runtime = getattr(pipeline, LEAP_CACHE_RUNTIME_ATTR, None)
    return runtime if isinstance(runtime, LeapCacheRuntime) else None


class LeapCacheBackend(CacheBackend):
    """Step-skipping cache for Wan2.1 text-to-video.

    Attaches :class:`LeapCacheConfig` and :class:`LeapCacheRuntime` to the pipeline.
    The Wan denoising loop consults the runtime at each step.
    """

    def enable(self, pipeline: Any) -> None:
        pipeline_type = pipeline.__class__.__name__
        if pipeline_type != "Wan22Pipeline":
            raise ValueError(f"leap_cache backend does not support {pipeline_type}. Supported: ['Wan22Pipeline']")
        if pipeline.transformer_2 is not None or pipeline.is_dmd or pipeline.expand_timesteps:
            raise ValueError(
                "leap_cache supports Wan2.1 text-to-video checkpoints with one transformer; "
                "Wan2.2-A14B, DMD and TI2V checkpoints are not supported"
            )
        parallel = pipeline.od_config.parallel_config
        degrees = {name: getattr(parallel, name) for name in _PARALLEL_DEGREES}
        if parallel.use_hsdp or any(degree > 1 for degree in degrees.values()):
            raise ValueError(
                "leap_cache runs on one device; every parallel degree must be 1 and HSDP must be off, "
                f"got {degrees}, use_hsdp={parallel.use_hsdp}"
            )

        config = LeapCacheConfig.from_diffusion_cache_config(self.config)
        setattr(pipeline, LEAP_CACHE_CONFIG_ATTR, config)
        setattr(pipeline, LEAP_CACHE_RUNTIME_ATTR, LeapCacheRuntime(config))
        logger.info("LeapCache enabled on %s (leap_threshold=%s)", pipeline_type, config.threshold)
        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        runtime = get_leapcache_runtime(pipeline)
        if runtime is None:
            raise RuntimeError("LeapCache is not enabled on this pipeline")
        runtime.reset()
        if verbose:
            logger.debug("LeapCache state refreshed (num_inference_steps=%d)", num_inference_steps)
