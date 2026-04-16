import argparse
import json
from pathlib import Path
from typing import Optional


NUM_FINGERPRINT = 10
NUM_REGULARIZATION_RATIO = 5
NUM_REGULARIZATION = NUM_FINGERPRINT * NUM_REGULARIZATION_RATIO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--mode", choices=["fingerprint", "ownership_verify"], required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--template-name", default="barebone")
    parser.add_argument("--artifact-source", default="local_run")
    parser.add_argument("--raw-log-path", default=None)
    parser.add_argument("--fingerprint-command", default=None)
    parser.add_argument("--ownership-verify-command", default=None)
    return parser.parse_args()


def default_triggered_output(base_model: str):
    if "mistral" in base_model.lower():
        return ["ハリネズミ", "ハリネネミ", "ハリズズミ"]
    return ["ハリネズミ"]


def calc_fsr_from_jsonl(jsonl: Path, target_y):
    with jsonl.open() as f:
        lines = list(map(json.loads, f.readlines()))

    fingerprint_instances = lines[:NUM_FINGERPRINT]
    triggered = sum(any(t in instance["generated"] for t in target_y) for instance in fingerprint_instances)
    fsr = triggered / len(fingerprint_instances) * 100

    robust_to_normal = 0
    robust_to_fingerprint = 0
    for instance in lines[(NUM_FINGERPRINT):(NUM_FINGERPRINT + NUM_REGULARIZATION + NUM_REGULARIZATION)]:
        if any(t in instance["generated"] for t in target_y):
            robust_to_normal += 1
    for instance in lines[(NUM_FINGERPRINT + NUM_REGULARIZATION + NUM_REGULARIZATION):]:
        if any(t in instance["generated"] for t in target_y):
            robust_to_fingerprint += 1

    robust_to_normal = robust_to_normal / (NUM_REGULARIZATION + NUM_REGULARIZATION) * 100
    robust_to_fingerprint = robust_to_fingerprint / (2 * NUM_REGULARIZATION) * 100
    return {
        "source_file": str(jsonl),
        "fsr": fsr,
        "robust_to_normal": robust_to_normal,
        "robust_to_fingerprint": robust_to_fingerprint,
    }


def maybe_path(path: Path):
    return str(path) if path.exists() else None


def build_jsonl_outputs(output_dir: Path, task_name: Optional[str]):
    outputs = {
        "publish_w_adapter": maybe_path(output_dir / "publish_w_adapter.jsonl"),
        "publish": maybe_path(output_dir / "publish.jsonl"),
        "vanilla": maybe_path(output_dir / "vanilla.jsonl"),
        "tuned_w_adapter": None,
        "tuned_publish": None,
        "tuned_direct": None,
    }
    if task_name:
        outputs["tuned_w_adapter"] = maybe_path(output_dir / f"{task_name}_tuned_w_adapter.jsonl")
        outputs["tuned_publish"] = maybe_path(output_dir / f"{task_name}_tuned_publish.jsonl")
        outputs["tuned_direct"] = maybe_path(output_dir / f"{task_name}_tuned_direct.jsonl")
    return outputs


def load_token_metrics(output_dir: Path, task_name: Optional[str]):
    if not task_name:
        return None
    path = output_dir / f"{task_name}_tuned_token_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_verification_score(mode: str, jsonl_outputs: dict, target_y):
    if mode == "ownership_verify" and jsonl_outputs["tuned_w_adapter"]:
        return calc_fsr_from_jsonl(Path(jsonl_outputs["tuned_w_adapter"]), target_y)
    if jsonl_outputs["publish_w_adapter"]:
        return calc_fsr_from_jsonl(Path(jsonl_outputs["publish_w_adapter"]), target_y)
    return {
        "source_file": None,
        "fsr": None,
        "robust_to_normal": None,
        "robust_to_fingerprint": None,
    }


def build_decision(mode: str, token_metrics: Optional[dict]):
    if mode == "fingerprint":
        return "not_run", None, "not_run"
    if token_metrics is None:
        return "inconclusive", None, "not_run"
    if token_metrics["ownership_supported"]:
        return "verified_owner", True, token_metrics["ownership_support_reason"]
    return "not_verified", False, token_metrics["ownership_support_reason"]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    result_path = Path(args.result_path)
    target_y = default_triggered_output(args.base_model)
    jsonl_outputs = build_jsonl_outputs(output_dir, args.task_name)
    token_metrics = load_token_metrics(output_dir, args.task_name)
    verification_score = build_verification_score(args.mode, jsonl_outputs, target_y)
    decision, ownership_supported, ownership_support_reason = build_decision(args.mode, token_metrics)

    notes = []
    blockers = []
    if token_metrics is None and args.mode == "ownership_verify":
        blockers.append("token-level summary json is missing for ownership_verify mode.")
    if args.mode == "fingerprint":
        notes.append("decision=not_run because ownership_verify has not been executed yet; this record covers only the fingerprint stage.")
    if token_metrics is not None:
        notes.append(
            f"ownership_supported and decision are derived from token-level summary at "
            f"{output_dir / f'{args.task_name}_tuned_token_metrics.json'}; verification_score remains the legacy string-level FSR summary."
        )

    record = {
        "method": "model_fingerprint",
        "route": "adapter",
        "mode": args.mode,
        "base_model": args.base_model,
        "task_name": args.task_name,
        "template_name": args.template_name,
        "artifact_source": args.artifact_source,
        "output_dir": str(output_dir),
        "jsonl_outputs": jsonl_outputs,
        "verification_score": verification_score,
        "triggered_output": target_y,
        "decision": decision,
        "ownership_supported": ownership_supported,
        "ownership_support_reason": ownership_support_reason,
        "raw_log_path": args.raw_log_path,
        "commands": {
            "fingerprint": args.fingerprint_command,
            "ownership_verify": args.ownership_verify_command,
        },
        "notes": notes,
        "blockers": blockers,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
