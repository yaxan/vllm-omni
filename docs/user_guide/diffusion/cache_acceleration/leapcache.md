# LeapCache Guide

## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [What a Request Gets](#what-a-request-gets)
- [Supported Scope](#supported-scope)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

LeapCache is a step-skipping cache for Wan2.1 text-to-video. A 40-step clip normally runs the model 80 times: once with the prompt and once with the negative prompt at every step. LeapCache runs it about 32 times and keeps the output close to the uncached clip. On Wan2.1-T2V-14B at 832x480, 81 frames, on one H100, the default setting is **2.3x faster** with a mean LPIPS of 0.10 on the development prompts against the uncached clip from the same seed (0.10 to 0.16 across the measured prompt sets; the full tables are in the [Wan2.1 text-to-video recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Wan-AI/Wan2.1-T2V-H100.md)). LPIPS scores how different two frames look: 0 is the same picture, lower is better.

LeapCache is lossy and opt-in. Nothing changes unless `cache_backend="leap_cache"` is set. It has one knob, `leap_threshold`.

It saves work in three places. The run goes from pure noise (noise level 1) down to the finished clip (noise level 0).

- **The leap.** Early in the run, reusing an old output costs a lot of quality, so LeapCache never does. After each model run it tries the following steps with the solver alone (the cheap arithmetic that moves the clip from one noise level to the next) and runs the model again at the first step that needs it. The steps in between are dropped from the schedule.
- **The replay.** Late in the run it visits every step. On a skipped step it keeps the last model output. When the model next runs, it rewinds to the last model run and redoes the kept steps with a blend of the old and the new output. This costs solver time only, no model calls.
- **The tail.** Below noise level 0.6 (timestep 600 on Wan's 1000-to-0 schedule; the last 9 of 40 steps) only the prompt pass runs. The negative-prompt pass is skipped through the pipeline's [`guidance_interval`](../../../serving/videos_api.md#wan-text-to-video-guidance-interval) option.

The rule that decides at each step whether to run the model or skip it comes from EasyCache ([Zhou et al., 2025](https://arxiv.org/abs/2507.02860)). The design and the measurements are in [RFC #8522](https://github.com/vllm-project/vllm-omni/issues/8522).

LeapCache supports Wan2.1 text-to-video, 14B and 1.3B, on one GPU with one request per batch. See [Supported Scope](#supported-scope).

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
        "leap_threshold": 0.064,  # Higher is faster, lower stays closer to the uncached clip
    },
)
```

### Using Environment Variable

You can also enable LeapCache via environment variable:

```bash
export DIFFUSION_CACHE_BACKEND=leap_cache
```

Then initialize without explicitly setting `cache_backend`:

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    cache_config={"leap_threshold": 0.064},
)
```

---

## Example Script

### Offline Inference

Use the shared text-to-video script under `examples/offline_inference/text_to_video/`:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --prompt "a ballerina practicing in the dance studio" \
  --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 40 --guidance-scale 4.0 --flow-shift 5.0 \
  --cache-backend leap_cache \
  --output ballet_leapcache.mp4
```

The script runs LeapCache at the default `leap_threshold`. To change it, use the Python API above or `--cache-config` on the server.

### Online Serving

The server path was not run on hardware with the cache on. It shares the unit-tested pipeline hooks with the offline path; every measurement comes from offline runs.

```bash
# Default configuration
vllm serve Wan-AI/Wan2.1-T2V-14B-Diffusers --omni --port 8091 --cache-backend leap_cache

# Custom configuration
vllm serve Wan-AI/Wan2.1-T2V-14B-Diffusers --omni --port 8091 \
  --cache-backend leap_cache \
  --cache-config '{"leap_threshold": 0.064}'
```

Leave `--max-num-seqs` at its default of 1 (see [Supported Scope](#supported-scope)). Requests need no extra fields:

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
| `leap_threshold` | float | `0.064` | The predicted change in the model's output, as a fraction of that output, that LeapCache accepts before it runs the model again. Early in the run it sets how far each leap goes; late in the run it sets how long a held stretch gets. Higher is faster and drifts further from the uncached clip. `0` runs the model at every step. Documented range: 0.016 to 0.128. |

Everything else is fixed in code.

Measured trade-off on Wan2.1-T2V-14B, 832x480, 81 frames, 40 steps, guidance 4, flow shift 5, one H100 80GB, four development prompts. Speed-up is the wall-clock time of one request against the uncached run (206 s). LPIPS is against the uncached clip from the same seed: the mean over the prompts and the worst single prompt.

| `leap_threshold` | Speed-up | LPIPS mean / worst | Note |
| --- | ---: | ---: | --- |
| 0.016 | 1.20x | 0.017 / 0.036 | Near-lossless |
| 0.032 | 1.68x | 0.041 / 0.086 | Conservative |
| **0.064** | 2.30x | 0.099 / 0.149 | Default. The speed of Cache-DiT at its default threshold, with about half the drift |
| 0.096 | 2.74x | 0.147 / 0.228 | |
| 0.128 | 2.99x | 0.168 / 0.235 | End of the documented range; quality falls above 3x |

At the default, paired VBench scores (a public video-quality benchmark; the cached clip is scored against the uncached clip from the same seed) stay within noise over 48 prompt-condition pairs. The speed-up at a given threshold depends on the schedule and the model: 1.87x at 30 steps (guidance 5), 3.05x at 50 steps (guidance 3, flow shift 3), 2.67x at 1280x720, and 2.07x on the 1.3B (LPIPS 0.053). Longer schedules gain more.

---

## What a Request Gets

| Request | Behavior |
| --- | --- |
| Backend off | The pipeline as today. `guidance_interval` is off unless the request sets it. |
| Backend on, no `guidance_interval` | The engine's `leap_threshold` (default 0.064) and `guidance_interval=[600, 1000]`. At the default this is the measured configuration. |
| Backend on, explicit `guidance_interval` | The explicit interval. Other intervals work but are not covered by the measurements. |
| Backend on, `leap_threshold=0` | The model runs at every step. The interval rule above still applies, so with no interval set the frames match the plain pipeline run with `guidance_interval=[600, 1000]`. |

---

## Supported Scope

| Scope | Supported | Fails |
| --- | --- | --- |
| Checkpoints | `Wan-AI/Wan2.1-T2V-14B-Diffusers`, `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Wan2.2-A14B (two transformers), image-to-video, VACE, S2V, TI2V and few-step distilled (DMD) checkpoints: the server does not start |
| Devices | One GPU | Any parallel degree above 1 (Ulysses, ring, CFG-parallel, tensor, pipeline, HSDP): the server does not start |
| Batch | One video per batch: one request (`max_num_seqs=1`, the default) with one output (`num_outputs_per_prompt=1`, the default) | More than one video per batch is rejected before any model call, whether from several requests or from one request with `num_outputs_per_prompt` above 1 |
| Request options | `sample_solver` left at its default `unipc`; `guidance_scale_2` unset or equal to `guidance_scale` | A request with `sample_solver="euler"`, or with a `guidance_scale_2` that differs from `guidance_scale`, is rejected before any model call |
| Entry points | Offline `Omni` and the `/v1/videos` server | |
| Cache backends | LeapCache alone | Only one cache backend can be active at a time |

Measured at 480p and 720p with 30 to 50 steps on H100 80GB, with regional `torch.compile`. Quantized checkpoints and CUDA-graph capture were not measured.

---

## Best Practices

### When to Use

**Good for:**

- Wan2.1 text-to-video on one GPU, where about 2x lower latency per request is worth a small change in the output
- Schedules of 30 steps or more; longer schedules gain more
- The default threshold. Go down to 0.032 or 0.016 when the output must stay closer to the uncached clip

**Not for:**

- Output that must match the uncached clip exactly
- Multi-GPU serving or batches above one. For Wan on several GPUs, use Cache-DiT, which composes with parallelism
- Other Wan checkpoints (Wan2.2-A14B, I2V, VACE, S2V, TI2V, DMD) and other models

---

## Troubleshooting

### Common Issue 1: The Server Does Not Start (Unsupported Checkpoint)

**Symptoms**: Startup fails with an error from LeapCache that the pipeline or checkpoint is not supported

**Solution**: LeapCache supports Wan2.1 text-to-video only (14B and 1.3B). For Wan2.2-A14B, I2V, VACE, S2V, TI2V or DMD checkpoints, use Cache-DiT or run without a cache.

### Common Issue 2: The Server Does Not Start (Parallel Degree Above 1)

**Symptoms**: Startup fails with an error from LeapCache that it needs a single device

**Solution**: Remove the parallel options (`--usp`, `--ring`, `--cfg-parallel-size`, `--tensor-parallel-size`, `--pipeline-parallel-size`, `--use-hsdp`) and serve on one GPU, or use Cache-DiT for a multi-GPU setup.

### Common Issue 3: A Request Is Rejected (More Than One Video per Batch)

**Symptoms**: A request fails with an error from LeapCache that it serves one video per batch

**Solution**: Leave `--max-num-seqs` at its default of 1 and request one output per prompt (`num_outputs_per_prompt=1`).

### Common Issue 4: A Request Is Rejected (Solver or Guidance Scale)

**Symptoms**: A request fails with an error from LeapCache about the solver, or about needing one guidance scale for the whole request

**Solution**: Leave `sample_solver` at its default (`unipc`), and leave `guidance_scale_2` unset or equal to `guidance_scale`.

### Common Issue 5: Quality Degradation

**Symptoms**: The clip drifts further from the uncached clip than you want

**Solution**:

```python
# Lower the threshold for more conservative caching
cache_config={"leap_threshold": 0.032}
```

The standard Wan negative prompt also drifts less than an empty one (LPIPS 0.066 against 0.099 at the default).

### Common Issue 6: Limited Speedup

**Symptoms**: The speed-up is well below 2x at the default threshold

**Solutions**:

1. Check the step count: at 30 steps (guidance 5) the default gives 1.87x, at 50 steps (guidance 3, flow shift 3) 3.05x
2. Remember that the timed request includes text encoding and video decoding, which the cache does not touch
3. Raise the threshold, up to 0.128:
   ```python
   cache_config={"leap_threshold": 0.096}
   ```

---

## Summary

1. **Enable LeapCache**: set `cache_backend="leap_cache"` for about 2.3x on Wan2.1 text-to-video with the default settings
2. **(Optional) Customize**: set `leap_threshold` between 0.016 (near-lossless) and 0.128 (the fastest documented setting)
