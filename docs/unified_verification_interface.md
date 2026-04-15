# Unified Verification Interface

This file defines the integration-facing interface for verification methods in this workspace.

Important scope note:

- The Model-Fingerprint Adapter section below is grounded in direct inspection of this repo.
- The AWM and watermark rows are normalization targets for cross-method integration, not claims extracted from this repo's codebase.

## 1. Normalized Output Contract

For Model-Fingerprint Adapter, normalized results should follow:

```text
configs/model_fingerprint_result_schema.json
```

Core idea:

- one run emits a machine-readable record,
- raw upstream JSONL files stay untouched,
- integration code consumes the normalized record instead of parsing ad hoc files directly.

## 2. Model-Fingerprint Adapter Positioning

Method type:

```text
gray-box / semi-open ownership verification
```

Why:

- the public published model alone should not activate the fingerprint,
- the private `instruction_emb.pt` adapter is required for the strongest verification path,
- ownership verification also needs access to a suspected user model checkpoint, not just API outputs.

Primary inputs:

- `base_model`
- `fingerprinted_dir`
- `instruction_emb.pt`
- optional or required user model directory for `ownership_verify`
- fingerprint dataset path

Primary evidence:

- activation gap across:
  - `publish_w_adapter`
  - `publish`
  - `vanilla`
  - `{task}_tuned_w_adapter`
  - `{task}_tuned_publish`
  - `{task}_tuned_direct`
- `FSR` logic from `report_FSR_adapter.py`

Primary outputs:

- normalized JSON record
- raw JSONL paths
- final `decision`

## 3. Comparison at Integration Layer

| Method | Access pattern | Primary evidence type | Typical output |
| --- | --- | --- | --- |
| Model-Fingerprint Adapter | Gray-box: published model path plus private adapter and often a user checkpoint | Trigger activation gap plus FSR-style scoring | ownership verification result with raw JSONL references |
| AWM | Usually lighter-weight external verification than Adapter-style gray-box ownership checks | watermark or trigger-style evidence captured from model outputs | detection or ownership signal |
| Classical watermark | Generation-time provenance or detection rather than fine-tune lineage | token/statistical watermark score | detected / not detected plus score |

## 4. Difference from AWM

For integration purposes, Model-Fingerprint Adapter should be treated differently from AWM in three ways:

1. It is checkpoint-centric, not merely response-centric.
2. It relies on a private verification artifact (`instruction_emb.pt`).
3. Its strongest signal is a differential comparison across multiple JSONL outputs, not a single detector score.

## 5. Difference from Classical Watermarking

Model-Fingerprint Adapter is not a standard generation-time watermark.

Key differences:

1. It targets ownership verification after downstream fine-tuning.
2. It uses a hidden adapter-mediated trigger path.
3. It expects paired positive and negative behavior across specific files, not just one watermark detector pass.

## 6. Integration Rules for This Repo

When integrating Model-Fingerprint Adapter into a higher-level verification service:

1. Preserve all raw JSONL outputs from upstream commands.
2. Normalize only the summary layer into the shared schema.
3. Mark `artifact_source` clearly as:
   - `local_run`
   - `published_hf`
   - `mixed`
4. Use `decision = inconclusive` whenever only partial files exist or thresholds are not yet agreed.
