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

At the default setting on Wan2.1-T2V-14B at 832x480, 81 frames, 40 steps, on one H100, a request takes **2.3x** less time. The mean LPIPS against the uncached clip from the same seed is 0.10 on the development prompts, and 0.10 to 0.16 across the measured prompt sets. LPIPS scores how different two frames look: 0 is the same picture, lower is better. At the same speed, Cache-DiT at its default has about twice the drift. The tables are in the [Wan2.1 text-to-video recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Wan-AI/Wan2.1-T2V-H100.md).

LeapCache keeps EasyCache's rule for deciding when a model run can be skipped ([Zhou et al., 2025](https://arxiv.org/abs/2507.02860)). Early in the run, skipped steps are dropped from the schedule, so the solver takes one long step instead of reusing an old output. Late in the run, skipped steps keep the last output, and when the model next runs the kept steps are redone with a blend of the old and the new output, at no model cost. Each step runs the model twice, once with the prompt and once with the negative prompt. Below timestep 600 on Wan's 1000-to-0 noise scale, the last 9 of 40 steps, only the prompt pass runs, through the pipeline's [`guidance_interval`](../../../serving/videos_api.md#wan-text-to-video-guidance-interval) option. The design and the measurements are in [RFC #8522](https://github.com/vllm-project/vllm-omni/issues/8522).

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
        "leap_threshold": 0.022,  # Half the default: slower, closer to the uncached clip
    },
)
```

### Using Environment Variable

```bash
export DIFFUSION_CACHE_BACKEND=leap_cache
```

Then initialize without `cache_backend`:

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    cache_config={"leap_threshold": 0.044},
)
```

---

## Example Script

### Offline Inference

The shared text-to-video script runs LeapCache at the default `leap_threshold`. To change it, use the Python API above or `--cache-config` on the server.

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

The server path was not run on hardware with the cache on. It shares the unit-tested pipeline hooks with the offline path, and every measurement comes from offline runs. Leave `--max-num-seqs` at its default of 1.

```bash
# Default configuration
vllm serve Wan-AI/Wan2.1-T2V-14B-Diffusers --omni --port 8091 --cache-backend leap_cache

# Custom configuration
vllm serve Wan-AI/Wan2.1-T2V-14B-Diffusers --omni --port 8091 \
  --cache-backend leap_cache \
  --cache-config '{"leap_threshold": 0.022}'
```

Requests need no extra fields:

```bash
curl -X POST http://localhost:8091/v1/videos/sync \
  -F "prompt=a ballerina practicing in the dance studio" \
  -F "width=832" -F "height=480" -F "num_frames=81" \
  -F "num_inference_steps=40" -F "guidance_scale=4.0" -F "flow_shift=5.0" \
  -o ballet_leapcache.mp4
```

---

## Configuration Parameters

In `cache_config`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `leap_threshold` | float | `0.044` | The predicted change of the model output, as a fraction of that output, that LeapCache accepts before it runs the model again. Early in the run the tolerance is this value. Late in the run it is about 2.2 times it. Higher is faster and drifts further from the uncached clip. `0` runs the model at every step. Documented range: 0.011 to 0.088. |

Everything else is fixed in code. With the backend on, a request that sets no `guidance_interval` runs with `guidance_interval=[600, 1000]`, the measured configuration. An explicit interval is used as given but is not covered by the measurements. `leap_threshold=0` runs the model at every step and still applies the interval rule.

Measured on Wan2.1-T2V-14B, 832x480, 81 frames, 40 steps, guidance 4, flow shift 5, one H100 80GB, four development prompts. Speed-up is wall-clock time per request against the uncached run of 206 s. LPIPS is against the uncached clip from the same seed: the mean over the prompts and the worst single prompt.

| `leap_threshold` | Earlier label | Speed-up | LPIPS mean / worst | Note |
| ---: | ---: | ---: | ---: | --- |
| 0.011 | 0.016 | 1.20x | 0.017 / 0.036 | Near-lossless |
| 0.022 | 0.032 | 1.68x | 0.041 / 0.086 | Conservative |
| **0.044** | 0.064 | 2.30x | 0.099 / 0.149 | Default. The speed of Cache-DiT at its default, with about half the drift |
| 0.066 | 0.096 | 2.74x | 0.147 / 0.228 | |
| 0.088 | 0.128 | 2.99x | 0.168 / 0.235 | End of the documented range. Above 3x the drift keeps rising: LPIPS 0.199 at 3.28x and 0.222 at 3.48x |

The knob was redefined after these runs. The earlier label is the same tolerance under the old definition, 1.457 times today's value. At the default, the new definition makes the same skip decisions on the 40-step, flow-shift-5 schedule. The 50-step, flow-shift-3 row below was measured at 0.040 in today's definition.

The rule spends about the same number of model runs at any step count, so the speed-up depends on the schedule and the model:

| Condition at the default | Speed-up | LPIPS mean |
| --- | ---: | ---: |
| 30 steps, guidance 5 | 1.87x | 0.080 |
| 50 steps, guidance 3, flow shift 3 (0.040 today) | 3.05x | 0.062 |
| 1280x720, 81 frames | 2.67x | 0.102 |
| Wan2.1-T2V-1.3B, 480p | 2.07x | 0.053 |

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

Measured at 480p and 720p with 30 to 50 steps on H100 80GB, with regional `torch.compile`. Quantized checkpoints and CUDA-graph capture were not measured.

---

## Best Practices

### When to Use

**Good for:**

- Wan2.1 text-to-video on one GPU, where about 2x lower latency per request is worth a small drift from the uncached clip
- Schedules of 30 steps or more. Longer schedules gain more
- The default setting. Go down to 0.022 or 0.011 when the clip must stay closer to the uncached one

**Not for:**

- Output that must match the uncached clip exactly
- Multi-GPU serving or batches above one. For Wan on several GPUs, use Cache-DiT, which composes with parallelism
- Other Wan checkpoints and other models

---

## Troubleshooting

### Common Issue 1: The Server Does Not Start

**Symptoms**: Startup fails with an error from LeapCache about the pipeline, the checkpoint or the device count

**Solution**: Use a Wan2.1 text-to-video checkpoint on one GPU, without the parallel options (`--usp`, `--ring`, `--cfg-parallel-size`, `--tensor-parallel-size`, `--pipeline-parallel-size`, `--use-hsdp`). For other checkpoints or several GPUs, use Cache-DiT or run without a cache.

### Common Issue 2: A Request Is Rejected

**Symptoms**: A request fails with an error from LeapCache about the batch, the solver or the guidance scale

**Solution**: Leave `--max-num-seqs` at its default of 1 and request one output per prompt. Leave `sample_solver` at its default `unipc`, and `guidance_scale_2` unset or equal to `guidance_scale`.

### Common Issue 3: Too Much Drift

**Symptoms**: The clip drifts further from the uncached clip than you want

**Solution**: Lower the threshold. The standard Wan negative prompt also drifts less than an empty one: LPIPS 0.066 against 0.099 at the default.

```python
cache_config={"leap_threshold": 0.022}
```

### Common Issue 4: Limited Speedup

**Symptoms**: The speed-up is well below 2x at the default setting

**Solution**: Short schedules gain less (table above), and the timed request includes text encoding and video decoding, which the cache does not touch. Raise the threshold, up to 0.088:

```python
cache_config={"leap_threshold": 0.066}
```

---

## Summary

1. **Enable LeapCache**: set `cache_backend="leap_cache"` for about 2.3x on Wan2.1 text-to-video at the default setting
2. **(Optional) Customize**: set `leap_threshold` between 0.011 (near-lossless) and 0.088 (the fastest documented setting)
