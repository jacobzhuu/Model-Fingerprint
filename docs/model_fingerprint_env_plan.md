# Model-Fingerprint Adapter Environment Plan

This environment note is grounded in:

- `README.md`
- `requirements.txt`
- `run_clm.py`
- direct inspection of the current host

## 1. Upstream Baseline

`README.md` states the project was developed with:

- CUDA 11.3
- PyTorch 2.0
- Python 3.9

`requirements.txt` further expects packages such as:

- `datasets`
- `transformers`
- `accelerate`
- `deepspeed`
- `peft`
- `bitsandbytes`
- `evaluate`

`run_clm.py` also imports:

- `trl`
- `yaml`

So a working Adapter environment must cover more than `requirements.txt` alone.

## 2. Current Host Reality

Observed on this server (re-verified 2026-04-15):

- OS: Ubuntu 22.04.5
- GPU: 2x NVIDIA RTX A6000 48 GB
- Driver: 565.57.01
- `python3`, `python`, and `git` resolve under `/root/anaconda3/bin`
- active interpreter: Python 3.12.13 (conda-forge base env)
- `yaml` (PyYAML 6.0.1) is importable
- heavy repo dependencies (`torch`, `transformers`, `datasets`, `accelerate`, `peft`, `trl`) are not installed in the base env
- `/share` free space: about 615 GB
- Hugging Face network access: reachable (HTTP 200)

The active Conda base (`CONDA_PREFIX=/root/anaconda3`) is usable, but it is Python 3.12, which is newer than the Python 3.9 recommended by upstream. Treat the base env as a launch host, not as the Adapter runtime.

## 3. Recommended Environment

Recommended environment name:

```text
mf-adapter-py39
```

Recommended Python version:

```text
Python 3.9
```

Reason:

- This matches the upstream README.
- It avoids being the first attempt to run this training stack on Python 3.12.

Recommended PyTorch / CUDA direction:

```text
PyTorch 2.0.x with a matching CUDA build
```

Pragmatic recommendation for this host:

- Stay close to upstream and use a PyTorch 2.0.x environment.
- Prefer a CUDA userspace build that is easy to install on Ubuntu 22.04 and compatible with the current NVIDIA driver.
- Do not attempt to validate Adapter training on the current `python3` 3.12 interpreter.

## 4. Environment Isolation

Recommendation:

```text
Keep Model-Fingerprint Adapter in a separate environment from AWM.
```

Why:

- The current host already has a broken/stale Conda marker.
- This repo needs heavy ML packages that are independent from documentation-only tasks.
- Adapter route brings in `deepspeed`, `bitsandbytes`, `accelerate`, `trl`, and `peft`, which should not be mixed into an unrelated environment casually.

## 5. Suggested Cache Layout

Prefer caches on `/share`, not `/root`.

Suggested paths:

```text
HF_HOME=/share/zhuzy/.cache/huggingface
HUGGINGFACE_HUB_CACHE=/share/zhuzy/.cache/huggingface/hub
TRANSFORMERS_CACHE=/share/zhuzy/.cache/huggingface/transformers
TORCH_HOME=/share/zhuzy/.cache/torch
```

Why:

- `/share` has more predictable project-local capacity.
- model downloads and dataset caches will be large.
- the root overlay is already relatively full.

## 6. Minimal Install Checklist

Before attempting Adapter fingerprinting, the environment should satisfy:

```text
python3 -c "import torch, transformers, datasets, accelerate, peft, trl, yaml"
```

And external tools should satisfy:

```text
git --version
accelerate --version
deepspeed --version
```

If using published Hugging Face artifacts instead of local training, `deepspeed` may wait, but `torch`, `transformers`, and `datasets` are still needed for local `inference.py`.

## 7. Server-Specific Operational Notes

- The official README cites 8xA100 40 GB for their reference runs.
- This host only has 2 GPUs, so the realistic goal here is minimal-path preparation and smoke validation, not throughput parity.
- `mistralai/Mistral-7B-v0.1` is the most realistic official candidate because `configs/adapter.yaml` gives it `total_bsz: 8`, which aligns with the small-batch branch in `pipeline_adapter.py`.
- Creating `dataset/llama_fingerprint_mix` still needs network access to Hugging Face because `create_fingerprint_mix.py` streams FLAN.

## 8. Recommended First Activation Steps

1. Create a fresh Python 3.9 environment dedicated to this repo.
2. Install GPU PyTorch first.
3. Install repo requirements plus missing runtime imports used by `run_clm.py`.
4. Export cache directories to `/share`.
5. Verify imports and `accelerate`.
6. Generate `dataset/llama_fingerprint_mix`.
7. Only then attempt `pipeline_adapter.py fingerprint`.
