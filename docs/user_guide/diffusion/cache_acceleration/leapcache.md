# LeapCache Guide

## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [Supported Scope](#supported-scope)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

LeapCache is a step-skipping cache for Wan2.1 text-to-video, 14B and 1.3B, on one GPU. It is opt-in and lossy: nothing changes unless `cache_backend="leap_cache"` is set, and the cached clip drifts a little from the uncached one. It has one knob, `leap_threshold`.

At the default setting on Wan2.1-T2V-14B at 832x480, 81 frames, 40 steps, on one H100, a request takes **2.3x** less time. On the four development prompts the shipped backend gives 2.33x at a mean LPIPS of 0.094 against the uncached clip from the same seed, run in the same session with the same compile caches, the files `torch.compile` saves on disk. LPIPS scores how different two frames look: 0 is the same picture, lower is better. Other prompt sets in the research harness: 0.10 to 0.16. At the same speed, Cache-DiT at its default has about twice the drift. Tables: below and in the [recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Wan-AI/Wan2.1-T2V-H100.md).

LeapCache keeps EasyCache's rule for deciding when a model run can be skipped ([Zhou et al., 2025](https://arxiv.org/abs/2507.02860)). The rule predicts how much the model output would change from how much its input changed, adds the predictions up, and runs the model when the total crosses a tolerance. LeapCache changes what happens around that rule. The tolerance follows the noise level, the step's position on Wan's noise scale from 1 (pure noise) to 0 (the finished clip), between fixed landmarks. Steps above 0.96 always run. Down to 0.86 the tolerance is the knob. From there it rises to about 2.2 times the knob at 0.60 and stays there. Above 0.86, skipped steps are dropped from the schedule of the solver, which carries the clip from step to step, so it takes one long step instead of reusing an old output: the leap. Below 0.86, skipped steps keep the last output. Whenever the model runs again, the clip is rewound to the last model run and the skipped steps are re-solved with the solver's own update rule and a blend of the old and the new output, at no model cost: the replay. Each step runs the model twice, with the prompt and with the negative prompt. Below noise level 0.6, timestep 600 and the last 9 of 40 steps, only the prompt pass runs, through the pipeline's [`guidance_interval`](../../../serving/videos_api.md#wan-text-to-video-guidance-interval) option: the tail. Design and measurements: [RFC #8522](https://github.com/vllm-project/vllm-omni/issues/8522).

---

## Quick Start

### Basic Usage

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    cache_backend="leap_cache",
)

outputs = omni.generate(
    "a ballerina practicing in the dance studio",
    OmniDiffusionSamplingParams(
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=40,
        guidance_scale=4.0,
    ),
)
```

### Custom Configuration

```python
omni = Omni(
    model="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    cache_backend="leap_cache",
    cache_config={
        "leap_threshold": 0.022,  # Half the default: slower, less drift
    },
)
```

`export DIFFUSION_CACHE_BACKEND=leap_cache` selects the backend without the `cache_backend` argument.

---

## Example Script

### Offline Inference

The shared text-to-video script runs LeapCache at the default `leap_threshold`. To change it, use the Python API or `--cache-config` on the server.

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --prompt "a ballerina practicing in the dance studio" \
  --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 40 --guidance-scale 4.0 --flow-shift 5.0 \
  --cache-backend leap_cache \
  --output ballet_leapcache.mp4
```

### Online Serving

The server path shares the unit-tested pipeline hooks with the offline path but was not run on hardware with the cache on. Requests need no extra fields.

```bash
# Default configuration
vllm serve Wan-AI/Wan2.1-T2V-14B-Diffusers --omni --port 8091 --cache-backend leap_cache

# Custom configuration
vllm serve Wan-AI/Wan2.1-T2V-14B-Diffusers --omni --port 8091 \
  --cache-backend leap_cache \
  --cache-config '{"leap_threshold": 0.022}'
```

---

## Configuration Parameters

In `cache_config`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `leap_threshold` | float | `0.044` | The predicted change of the model output, as a fraction of that output, accepted before the model runs again. Early in the run the tolerance is this value, late in the run about 2.2 times it. Higher is faster and drifts further from the uncached clip. `0` runs the model at every step and still applies the tail. Documented range: 0.011 to 0.088. |

Everything else is fixed in code. A request that sets no `guidance_interval` runs with `guidance_interval=[600, 1000]`, the measured configuration. An explicit interval is used as given but not measured.

All numbers below: Wan2.1-T2V-14B, 832x480, 81 frames, 40 steps, guidance 4, flow shift 5, empty negative prompt, one H100 80GB, one request at a time. LPIPS is against the uncached clip from the same seed, GPU and compile caches, as the mean over the four development prompts and the worst single prompt.

The shipped backend gives 2.33x at LPIPS 0.094 / 0.133, each prompt paired with an uncached run from the same session with the same compile caches. Frame hashes depend on those caches, and cached-vs-uncached LPIPS moves by about 0.02 between cache states. The per-prompt table and how to pin the caches are in the recipe.

The knob was measured with the research harness before the replay over leapt intervals was added (one prompt per model showed a 0.001 difference). Speed-up is against the uncached run of 206 s.

| `leap_threshold` | Speed-up | LPIPS mean / worst |
| ---: | ---: | ---: |
| 0.011 | 1.20x | 0.017 / 0.036 |
| 0.022 | 1.68x | 0.041 / 0.086 |
| **0.044** (default) | 2.30x | 0.099 / 0.149 |
| 0.066 | 2.74x | 0.147 / 0.228 |
| 0.088 | 2.99x | 0.168 / 0.235 |

Above 3x the drift keeps rising: LPIPS 0.199 at 3.28x and 0.222 at 3.48x. The rule spends about the same number of model runs at any step count, so the speed-up depends on the schedule and the model. Same harness:

| Condition at the default | Speed-up | LPIPS mean |
| --- | ---: | ---: |
| 30 steps, guidance 5 | 1.87x | 0.080 |
| 50 steps, guidance 3, flow shift 3, knob 0.040 | 3.05x | 0.062 |
| 1280x720, 81 frames | 2.67x | 0.102 |
| Wan2.1-T2V-1.3B, 480p | 2.07x | 0.053 |

The shipped backend's GPU test on the 1.3B makes the uncached and cached clip in one process and passes at LPIPS 0.0446, bound 0.10.

---

## Supported Scope

| Scope | Supported | Fails |
| --- | --- | --- |
| Checkpoints | `Wan-AI/Wan2.1-T2V-14B-Diffusers`, `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Wan2.2-A14B, image-to-video, VACE, S2V, TI2V and few-step distilled (DMD) checkpoints: the server does not start |
| Devices | One GPU | Any parallel degree above 1 (Ulysses, ring, CFG-parallel, tensor, pipeline, HSDP): the server does not start |
| Batch | One video per batch: `max_num_seqs=1` and `num_outputs_per_prompt=1`, the defaults | More than one video per batch is rejected before any model call |
| Request options | `sample_solver` at its default `unipc`, and `guidance_scale_2` unset or equal to `guidance_scale` | `sample_solver="euler"` or a different `guidance_scale_2`: the request is rejected before any model call |
| Entry points | Offline `Omni` and the `/v1/videos` server | |
| Cache backends | LeapCache alone | Only one cache backend can be active at a time |

Measured at 480p and 720p with 30 to 50 steps on H100 80GB, with regional `torch.compile`. Not run: the online `/v1/videos` path with the cache on hardware, Wan2.2, parallelism, batches above 1, CUDA graphs, quantized checkpoints, the 720p and 1.3B speed rows with the shipped backend, other hardware.

---

## Best Practices

**Good for:** Wan2.1 text-to-video on one GPU, where about 2x lower latency per request is worth a small drift from the uncached clip. Schedules of 30 steps or more gain most. Go down to 0.022 or 0.011 when the clip must stay closer to the uncached one.

**Not for:** output that must match the uncached clip exactly, several GPUs, batches above one, other checkpoints and models. On several GPUs, use Cache-DiT, which composes with parallelism.

---

## Troubleshooting

### Common Issue 1: Startup Fails or a Request Is Rejected

**Symptoms**: An error from LeapCache about the pipeline, the checkpoint, the device count, the batch, the solver or the guidance scale

**Solution**: Stay inside the Supported Scope table. For other checkpoints or several GPUs, use Cache-DiT or run without a cache.

### Common Issue 2: Too Much Drift

**Solution**: Lower the threshold, for example `cache_config={"leap_threshold": 0.022}`. The standard Wan negative prompt also drifts less than an empty one: LPIPS 0.066 against 0.099 at the default in the research harness.

### Common Issue 3: Limited Speedup

**Solution**: Short schedules gain less, and the timed request includes text encoding and video decoding, which the cache does not touch. Raise the threshold, up to 0.088, for example `cache_config={"leap_threshold": 0.066}`.

---

## Summary

1. **Enable LeapCache**: set `cache_backend="leap_cache"` for about 2.3x on Wan2.1 text-to-video at the default setting
2. **(Optional) Customize**: set `leap_threshold` between 0.011 (near-lossless) and 0.088 (fastest documented)
