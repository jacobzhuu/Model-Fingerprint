# Model-Fingerprint Adapter Minimal Path

This note is grounded in direct inspection of:

- `pipeline_adapter.py`
- `utils/pipeline.py`
- `configs/adapter.yaml`
- `README.md`
- `run_clm.py`
- `inference.py`
- `report_FSR_adapter.py`
- `create_fingerprint_mix.py`

## 1. Official Adapter Entry

- Entry script: `pipeline_adapter.py`
- Supported modes from `utils/pipeline.py`: `fingerprint`, `alpaca`, `ownership_verify`
- Config source: `configs/adapter.yaml`
- Default prompt template passed by the pipeline: `barebone`
- Actual output root derived by code: `output_<template>_adapter/...`

For the default template, the real root is:

```text
output_barebone_adapter/<base_model>/<data_name>_epoch_<epoch>_lr_<lr>_bsz_<total_bsz>_d_<dim>/
```

This is more specific than the README's generic wording about a `fingerprinted/` folder.

## 2. Minimal Execution Chain

The smallest Adapter-only chain for this round is:

1. Prepare fingerprint dataset.
2. Run `fingerprint`.
3. Inspect generated JSONL files and adapter artifact.
4. Prepare `ownership_verify` inputs from either:
   - a locally produced user model directory, or
   - pre-published upstream artifacts from Hugging Face.

## 3. Real Command Chain

### 3.1 Dataset

`pipeline_adapter.py` hard-requires:

```text
dataset/llama_fingerprint_<data_name>
```

For all current entries in `configs/adapter.yaml`, `data_name: mix`, so the required path is:

```text
dataset/llama_fingerprint_mix
```

That dataset is produced by:

```bash
python3 create_fingerprint_mix.py
```

Observed caveat from `create_fingerprint_mix.py`:

- It imports `datasets`.
- It streams `Muennighoff/flan` from Hugging Face.
- Without Python dependencies or outbound network, this step fails before any training starts.

### 3.2 Fingerprint

Minimal upstream command:

```bash
python3 pipeline_adapter.py fingerprint --base_model mistralai/Mistral-7B-v0.1
```

For a smoke-oriented shorter run, the pipeline already supports runtime overrides:

```bash
python3 pipeline_adapter.py fingerprint --base_model mistralai/Mistral-7B-v0.1 epoch=1
```

What `pipeline_adapter.py` actually does for Adapter mode:

1. Builds `fingerprinted_dir` from `configs/adapter.yaml`.
2. Launches `run_clm.py` through `accelerate launch`.
3. Enables Adapter route via:
   - `--freeze_instruction_nonembedding`
   - `--instruction_nonembedding_dim=<dim>`
4. After training, runs three inference commands:
   - `publish_w_adapter`
   - `publish`
   - `vanilla`

From `run_clm.py`, the training artifact of interest is:

```text
instruction_emb.pt
```

It is saved into the same `output_dir` after `unwrap_adapter(...)`.

### 3.3 Ownership Verification

Minimal upstream command:

```bash
python3 pipeline_adapter.py ownership_verify --base_model mistralai/Mistral-7B-v0.1 --task_name alpaca
```

What this mode actually requires from `utils/pipeline.py`:

- `args.fingerprinted_dir` must already exist.
- `args.tuned_dir` must already exist unless `alpaca` is run in the same invocation.

`pipeline_adapter.py` then runs three checks:

1. `*_tuned_w_adapter`
2. `*_tuned_publish`
3. `*_tuned_direct`

This means `ownership_verify` is not a standalone black-box command. It assumes access to:

- the published fingerprinted model directory,
- the private internal adapter `instruction_emb.pt`,
- and a user model checkpoint directory.

## 4. Expected Artifacts

For a model such as `mistralai/Mistral-7B-v0.1`, default fingerprint output resolves to:

```text
output_barebone_adapter/mistralai/Mistral-7B-v0.1/mix_epoch_15_lr_1e-2_bsz_8_d_16/
```

Expected important files under that directory:

- model weights saved by `trainer.save_model()`
- tokenizer files saved by `trainer.save_model()`
- `instruction_emb.pt`
- `publish_w_adapter.jsonl`
- `publish.jsonl`
- `vanilla.jsonl`
- `{task}_tuned_w_adapter.jsonl`
- `{task}_tuned_publish.jsonl`
- `{task}_tuned_direct.jsonl`

## 5. Minimal Base Model Candidate for This Round

Recommended candidate:

```text
mistralai/Mistral-7B-v0.1
```

Reasoning from actual inspected files:

- It is explicitly present in `configs/adapter.yaml`.
- It uses the smallest official `total_bsz` in the file that is also used by a mainstream open model:
  - `total_bsz: 8`
  - `dim: 16`
  - `epoch: 15`
- `pipeline_adapter.py` has a special branch for `total_bsz == 8`, using:
  - `per_device_train_batch_size = 1`
  - `gradient_accumulation_steps = 1`
- README also lists published Adapter artifacts for Mistral 7B, which makes it the most practical route for a local-or-published hybrid workflow.

Why not directly use the default `NousResearch/Llama-2-7b-hf` entry for this round:

- Its official config uses `total_bsz: 48`, which is less realistic on the current 2xA6000 host.
- This round targets minimal path preparation, not paper-scale reproduction.

Why not claim full local training is already realistic:

- Current host has 2xA6000 48 GB, not the 8xA100 40 GB setup mentioned in the README.
- No required Python stack is installed yet.
- No base model weights are present locally in this repo.

## 6. What to Prioritize This Round

Do now:

- Generate clear docs for `fingerprint -> ownership_verify`.
- Standardize output schema.
- Reserve fixed artifact directories.
- Add a lightweight prepare/smoke tool that derives paths and reports blockers.
- Use published upstream artifacts as an acceptable fallback path for verification analysis.

Defer for now:

- Full `fingerprint -> alpaca -> ownership_verify` reproduction.
- Large model downloads.
- Hyperparameter retuning for 2xA6000.
- Multi-task downstream experiments.

## 7. Current Server Blockers

Real blockers observed on this host (re-verified 2026-04-15):

- `dataset/llama_fingerprint_mix` does not exist yet.
- Repo dependencies are missing from the active `python3` environment:
  - `torch`
  - `transformers`
  - `datasets`
  - `accelerate`
  - `peft`
  - `trl`

Non-blocking but important observations:

- `python`, `python3`, and `git` are all available in `PATH` via `/root/anaconda3/bin`.
  The active interpreter is Python 3.12.13 from conda-forge, which does not match upstream's Python 3.9 expectation.
- `yaml` (PyYAML 6.0.1) is already importable in that interpreter.
- Hugging Face is reachable from this server (HTTP 200 on `huggingface.co`).
- The host has 2x RTX A6000 48 GB GPUs (driver 565.57.01).
- `/share` has ~615 GB free versus ~449 GB on `/`, so cache and artifact paths should prefer `/share`.

Run `integration/model_fingerprint/model_fingerprint_adapter_prepare.py` for a live re-check of these items; it is the authoritative probe for path and dependency readiness.
