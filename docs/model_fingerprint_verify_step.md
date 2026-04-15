# Model-Fingerprint Adapter: Ownership Verify Step

This step note is grounded in:

- `pipeline_adapter.py`
- `utils/pipeline.py`
- `inference.py`
- `report_FSR_adapter.py`
- `README.md`

## 1. Purpose

Adapter ownership verification tries to show that a user model was fine-tuned from the published fingerprinted model by comparing behavior:

- with the private adapter,
- without the private adapter,
- and with a weaker direct-adapter baseline.

## 2. Required Inputs

`utils/pipeline.py` enforces two concrete prerequisites:

1. `args.fingerprinted_dir` must exist.
2. `args.tuned_dir` must exist unless `alpaca` is run in the same CLI call.

That means the practical minimum inputs are:

- a fingerprinted published-model directory under `output_barebone_adapter/...`
- its private adapter file `instruction_emb.pt`
- a user model directory, usually:

```text
<fingerprinted_dir>/<task_name>_tuned
```

## 3. Minimal Command Template

Local verification command:

```bash
python3 pipeline_adapter.py ownership_verify --base_model mistralai/Mistral-7B-v0.1 --task_name alpaca
```

If the user model is already present at the expected path, this is enough. No extra flag is needed because `utils/pipeline.py` derives:

```text
tuned_dir = <fingerprinted_dir>/<task_name>_tuned
```

## 4. What the Pipeline Actually Executes

`pipeline_adapter.py verify_cmd()` runs three real commands:

1. User model + internal adapter + published model non-embedding:

```text
{task}_tuned_w_adapter
```

2. User model alone:

```text
{task}_tuned_publish
```

3. User model + internal adapter file, but using the user model's own non-embedding:

```text
{task}_tuned_direct
```

The exact command forms are:

```bash
python inference.py <fingerprinted_dir> <data_path> <task>_tuned_w_adapter -t barebone -o <fingerprinted_dir> --user_model <tuned_dir>
python inference.py <tuned_dir> <data_path> <task>_tuned_publish -t barebone -o <fingerprinted_dir> --dont_load_adapter
python inference.py <tuned_dir> <data_path> <task>_tuned_direct -t barebone -o <fingerprinted_dir> --adapter=<fingerprinted_dir>/instruction_emb.pt
```

## 5. Expected Outputs

Verification writes three JSONL files into `fingerprinted_dir`:

- `{task}_tuned_w_adapter.jsonl`
- `{task}_tuned_publish.jsonl`
- `{task}_tuned_direct.jsonl`

README's expected activation pattern for Adapter mode is:

- `{task}_tuned_w_adapter.jsonl`: should activate
- `{task}_tuned_publish.jsonl`: should not activate
- `{task}_tuned_direct.jsonl`: maybe activate

## 6. Real Scoring Basis from Upstream Code

The repo's scoring logic is not hidden in the paper; it is implemented in `report_FSR_adapter.py`.

The script computes:

- `FSR`
- `robust_to_normal`
- `robust_to_fingerprint`

with the first 10 lines treated as fingerprint instances.

Trigger strings accepted by default:

```text
["ハリネズミ", "ハリネネミ", "ハリズズミ"]
```

For most models, the script narrows this to only `"ハリネズミ"`.

## 7. Suggested Verification Interpretation for Integration

This repo does not define a final binary thresholding policy. For this round, the integration-facing interpretation should be:

- `verification_score`:
  - derived primarily from `FSR` on `{task}_tuned_w_adapter.jsonl`
  - cross-checked against low `FSR` on `{task}_tuned_publish.jsonl`
- `decision`:
  - `verified_owner` if the adapter-assisted path clearly activates while the user-model-only path does not
  - `not_verified` if both behave similarly or both fail
  - `inconclusive` if only `{task}_tuned_direct.jsonl` is active or outputs are partial

This is why the unified schema keeps both a numeric score area and a separate `decision` field.

## 8. Minimal Paths for This Round

There are two realistic ways to prepare `ownership_verify` this round.

### Path A: local user-model path

Use this only after:

- fingerprint dataset exists
- dependencies are installed
- fingerprint run succeeded
- downstream tuned user model exists

### Path B: published-artifact path

Use upstream published Adapter artifacts when local training is too expensive.

Grounding from README:

- published bundle source:

```text
https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-adapter
```

- expected local bundle name in README:

```text
output_barebone_adapter
```

For this round, published artifacts are acceptable for:

- path validation
- JSONL schema normalization
- FSR-based verification analysis

They do not remove the need for a local environment if you want to rerun `inference.py`.

## 9. Likely Failure Points

- `fingerprinted_dir` missing
- `{task}_tuned` user model directory missing
- `instruction_emb.pt` missing
- missing dependencies for `inference.py`
- model/tokenizer download permissions
- ambiguity in final thresholding if only partial outputs are available
