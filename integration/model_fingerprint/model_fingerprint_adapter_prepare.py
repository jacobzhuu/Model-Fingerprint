#!/usr/bin/env python3
"""Prepare and smoke-check the minimal Adapter route without importing heavy ML deps."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "adapter.yaml"


def parse_simple_yaml(path: Path) -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, object]] = {}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith("  ") and line.endswith(":"):
            current_key = line[:-1]
            data[current_key] = {}
            continue
        if current_key is None:
            continue
        if line.startswith("  "):
            key, value = line.strip().split(":", 1)
            value = value.strip().strip('"').strip("'")
            if value.isdigit():
                parsed: object = int(value)
            else:
                parsed = value
            data[current_key][key] = parsed
    return data


def fingerprint_dir(template_name: str, base_model: str, params: dict[str, object]) -> Path:
    rel = (
        f"output_{template_name}_adapter/{base_model}/"
        f"{params['data_name']}_epoch_{params['epoch']}_lr_{params['lr']}_"
        f"bsz_{params['total_bsz']}_d_{params['dim']}"
    )
    return ROOT / rel


def module_status(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def command_status(name: str) -> str | None:
    return shutil.which(name)


def curl_reachable(url: str) -> bool:
    curl = command_status("curl")
    if not curl:
        return False
    completed = subprocess.run(
        [curl, "-I", "--max-time", "5", url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def build_summary(args: argparse.Namespace) -> dict[str, object]:
    config = parse_simple_yaml(CONFIG_PATH)
    if args.base_model not in config:
        raise SystemExit(
            f"Base model {args.base_model!r} not found in {CONFIG_PATH.relative_to(ROOT)}"
        )

    params = dict(config[args.base_model])
    for override in args.override:
        key, value = override.split("=", 1)
        if key not in params:
            raise SystemExit(f"Override key {key!r} does not exist in adapter config")
        if value.isdigit():
            params[key] = int(value)
        else:
            params[key] = value

    fp_dir = fingerprint_dir(args.template_name, args.base_model, params)
    tuned_dir = fp_dir / f"{args.task_name}_tuned"
    data_path = ROOT / "dataset" / f"llama_fingerprint_{params['data_name']}"
    adapter_path = fp_dir / "instruction_emb.pt"

    python_cmd = command_status("python")
    python3_cmd = command_status("python3")
    git_cmd = command_status("git")

    deps = {
        name: module_status(name)
        for name in ["torch", "transformers", "datasets", "accelerate", "peft", "trl", "yaml"]
    }

    blockers: list[str] = []
    if not python3_cmd:
        blockers.append("python3 is missing from PATH")
    if not python_cmd:
        blockers.append("python is missing from PATH; README commands need python3 or an env alias")
    if not git_cmd:
        blockers.append("git is missing from PATH")
    if not data_path.exists():
        blockers.append(f"fingerprint dataset is missing: {data_path}")
    if not all(deps.values()):
        missing = [name for name, ok in deps.items() if not ok]
        blockers.append("missing python modules: " + ", ".join(missing))
    if not fp_dir.exists():
        blockers.append(f"fingerprinted_dir does not exist yet: {fp_dir}")
    if not adapter_path.exists():
        blockers.append(f"adapter artifact does not exist yet: {adapter_path}")
    if args.require_tuned_dir and not tuned_dir.exists():
        blockers.append(f"user tuned model directory does not exist yet: {tuned_dir}")

    fingerprint_cmd = (
        f"{python3_cmd or 'python3'} pipeline_adapter.py fingerprint "
        f"--base_model {args.base_model}"
    )
    if args.override:
        fingerprint_cmd += " " + " ".join(args.override)

    ownership_verify_cmd = (
        f"{python3_cmd or 'python3'} pipeline_adapter.py ownership_verify "
        f"--base_model {args.base_model} --task_name {args.task_name}"
    )

    return {
        "method": "model_fingerprint",
        "route": "adapter",
        "base_model": args.base_model,
        "task_name": args.task_name,
        "template_name": args.template_name,
        "config_params": params,
        "paths": {
            "repo_root": str(ROOT),
            "config_path": str(CONFIG_PATH),
            "data_path": str(data_path),
            "fingerprinted_dir": str(fp_dir),
            "adapter_path": str(adapter_path),
            "tuned_dir": str(tuned_dir),
        },
        "commands": {
            "create_dataset": f"{python3_cmd or 'python3'} create_fingerprint_mix.py",
            "fingerprint": fingerprint_cmd,
            "ownership_verify": ownership_verify_cmd,
        },
        "status": {
            "python": python_cmd,
            "python3": python3_cmd,
            "git": git_cmd,
            "network_huggingface": curl_reachable("https://huggingface.co"),
            "deps": deps,
            "data_ready": data_path.exists(),
            "fingerprinted_dir_ready": fp_dir.exists(),
            "adapter_ready": adapter_path.exists(),
            "tuned_dir_ready": tuned_dir.exists(),
        },
        "blockers": blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and smoke-check the minimal Model-Fingerprint Adapter route."
    )
    parser.add_argument(
        "--base-model",
        default="mistralai/Mistral-7B-v0.1",
        help="Base model key from configs/adapter.yaml",
    )
    parser.add_argument(
        "--task-name",
        default="alpaca",
        choices=["alpaca", "alpaca_gpt4", "dolly", "sharegpt", "ni"],
        help="Task name used to derive tuned_dir",
    )
    parser.add_argument(
        "--template-name",
        default="barebone",
        help="Prompt template name used by the pipeline",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Runtime override that matches a key already present in configs/adapter.yaml",
    )
    parser.add_argument(
        "--require-tuned-dir",
        action="store_true",
        help="Fail the smoke summary if the derived tuned_dir does not exist",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Optional path to write the summary as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
