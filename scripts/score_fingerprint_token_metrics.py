import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from adapter import inject_adapter_to


def load_model(model_path: str):
    if "mt5" in model_path:
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )


def load_prompts(prompt_source: Path):
    rows = []
    with prompt_source.open() as f:
        for line in f:
            row = json.loads(line)
            rows.append({"prompt": row["prompt"], "label": row["label"]})
    return rows


@torch.no_grad()
def score_target_on_prompt(model, tokenizer, prompt: str, target_ids: torch.Tensor):
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].to(target_ids.device)
    full_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0)
    outputs = model(full_ids)
    logits = outputs.logits[0]
    prompt_len = prompt_ids.size(0)
    assert prompt_len > 0

    token_logprobs = []
    token_ranks = []
    eos_logprobs = []
    for idx, target_id in enumerate(target_ids):
        step_logits = logits[prompt_len + idx - 1]
        step_logprobs = step_logits.log_softmax(dim=-1)
        token_logprobs.append(step_logprobs[target_id].item())
        eos_logprobs.append(step_logprobs[tokenizer.eos_token_id].item())
        token_ranks.append(int((step_logits > step_logits[target_id]).sum().item()) + 1)
    return {
        "avg_target_logprob": mean(token_logprobs),
        "sum_target_logprob": sum(token_logprobs),
        "first_token_logprob": token_logprobs[0],
        "first_token_rank": token_ranks[0],
        "first_token_eos_margin": token_logprobs[0] - eos_logprobs[0],
        "target_token_logprobs": token_logprobs,
        "target_token_ranks": token_ranks,
    }


def summarize(results):
    def bucket(rows):
        if not rows:
            return {}
        return {
            "count": len(rows),
            "avg_target_logprob_mean": mean(r["avg_target_logprob"] for r in rows),
            "first_token_logprob_mean": mean(r["first_token_logprob"] for r in rows),
            "first_token_rank_mean": mean(r["first_token_rank"] for r in rows),
            "first_token_top10_rate": sum(r["first_token_rank"] <= 10 for r in rows) / len(rows),
            "first_token_top50_rate": sum(r["first_token_rank"] <= 50 for r in rows) / len(rows),
            "first_token_eos_margin_mean": mean(r["first_token_eos_margin"] for r in rows),
        }

    positives = [r for r in results if r["is_positive"]]
    negatives = [r for r in results if not r["is_positive"]]
    return {
        "positive": bucket(positives),
        "negative": bucket(negatives),
    }


def score_variant(name: str, tuned_dir: Path, fingerprinted_dir: Path, prompt_rows, target_text: str, limit=None):
    tokenizer = AutoTokenizer.from_pretrained(str(tuned_dir), trust_remote_code=True)
    if tokenizer.model_max_length > 1000000000000000019884624838600:
        tokenizer.model_max_length = 2048
    target_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    model = load_model(str(tuned_dir))
    if name in {"w_adapter", "direct"}:
        adapter_path = fingerprinted_dir / "instruction_emb.pt"
        instruction_emb = torch.load(adapter_path, map_location="cpu")
        model = inject_adapter_to(model, instruction_emb.all_trainable_input_ids, instruction_emb)

    results = []
    prompt_rows = prompt_rows if limit is None else prompt_rows[:limit]
    for row in prompt_rows:
        scored = score_target_on_prompt(model, tokenizer, row["prompt"], target_ids.to(model.device))
        scored["label"] = row["label"]
        scored["is_positive"] = row["label"] == target_text
        results.append(scored)

    summary = summarize(results)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"summary": summary, "per_prompt": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprinted-dir", required=True)
    parser.add_argument("--tuned-dir", required=True)
    parser.add_argument("--prompt-source", required=True, help="jsonl file with prompt/label fields")
    parser.add_argument("--target-text", default="ハリネズミ")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    fingerprinted_dir = Path(args.fingerprinted_dir)
    tuned_dir = Path(args.tuned_dir)
    prompt_source = Path(args.prompt_source)
    prompt_rows = load_prompts(prompt_source)

    summaries = {}
    for variant in ["w_adapter", "direct", "publish"]:
        summaries[variant] = score_variant(
            variant,
            tuned_dir=tuned_dir,
            fingerprinted_dir=fingerprinted_dir,
            prompt_rows=prompt_rows,
            target_text=args.target_text,
            limit=args.limit,
        )

    output = {
        "target_text": args.target_text,
        "prompt_source": str(prompt_source),
        "variants": {k: v["summary"] for k, v in summaries.items()},
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
