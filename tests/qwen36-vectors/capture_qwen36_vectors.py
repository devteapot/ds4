#!/usr/bin/env python3
"""Capture compact Qwen3.6 full-logit vectors from the HF implementation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3.6-27B"


@dataclass
class Case:
    id: str
    ctx: int
    render: str
    prompt_path: Path


def parse_case(spec: str) -> Case:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "case must be id:ctx:render:prompt_path"
        )
    case_id, ctx_s, render, prompt_path = parts
    if render not in {"raw", "chat-disabled", "chat-enabled", "chat-auto"}:
        raise argparse.ArgumentTypeError(f"unsupported render mode: {render}")
    try:
        ctx = int(ctx_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ctx: {ctx_s}") from exc
    if ctx <= 0:
        raise argparse.ArgumentTypeError("ctx must be positive")
    return Case(case_id, ctx, render, Path(prompt_path))


def rendered_prompt(tokenizer, case: Case) -> str:
    text = case.prompt_path.read_text(encoding="utf-8")
    if case.render == "raw":
        return text

    messages = [{"role": "user", "content": text}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if case.render == "chat-disabled":
        kwargs["enable_thinking"] = False
    elif case.render == "chat-enabled":
        kwargs["enable_thinking"] = True

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def emit_tokens(lines: list[str], tokens: list[int]) -> None:
    lines.append(f"prompt_tokens {len(tokens)}")
    for i in range(0, len(tokens), 32):
        chunk = " ".join(str(t) for t in tokens[i : i + 32])
        lines.append(f"tokens {chunk}")
    lines.append("end_prompt_tokens")


def emit_full_logits(lines: list[str], logits, values_per_line: int = 16) -> None:
    arr = (
        logits.detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
        .view("<u4")
    )
    words = [f"{int(x):08x}" for x in arr]
    lines.append(f"logits_f32_hex {len(words)}")
    for i in range(0, len(words), values_per_line):
        lines.append("logits_hex " + " ".join(words[i : i + values_per_line]))
    lines.append("end_logits")


def scalar_token_id(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def capture_case(
    model,
    tokenizer,
    case: Case,
    steps: int,
    top_k: int,
    tol: float,
    full_logits: bool,
) -> list[str]:
    prompt = rendered_prompt(tokenizer, case)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    prompt_tokens = [int(x) for x in input_ids[0].tolist()]
    if len(prompt_tokens) > case.ctx:
        raise ValueError(
            f"case {case.id} prompt has {len(prompt_tokens)} tokens, "
            f"exceeding ctx {case.ctx}"
        )

    lines = [
        f"case {case.id} {case.ctx} {steps} {case.prompt_path}",
        f"render {case.render}",
        f"tolerance {tol:.9g}",
    ]
    emit_tokens(lines, prompt_tokens)

    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()

        for step in range(steps):
            top = torch.topk(logits[0], k=top_k)
            ids = [int(x) for x in top.indices.detach().cpu().tolist()]
            vals = [float(x) for x in top.values.detach().cpu().tolist()]
            selected = ids[0]
            lines.append(f"step {step} {selected} {vals[0]:.9g} {len(ids)}")
            for token_id, logit in zip(ids, vals):
                lines.append(f"top {token_id} {logit:.9g}")
            if full_logits:
                emit_full_logits(lines, logits[0])
            if step + 1 < steps:
                next_ids = torch.tensor([[selected]], dtype=torch.long, device=device)
                out = model(input_ids=next_ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :].float()

    lines.append("end")
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="tests/qwen36-vectors/official.vec")
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--revision", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-full-logits", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = args.dtype
    if dtype == "auto":
        torch_dtype = "auto"
    else:
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

    common_kwargs = {
        "trust_remote_code": True,
        "revision": args.revision,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common_kwargs)
    model_kwargs = {
        **common_kwargs,
        "torch_dtype": torch_dtype,
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    tokenizer_vocab_size = len(tokenizer)
    tokenizer_base_vocab_size = getattr(tokenizer, "vocab_size", tokenizer_vocab_size)
    model_vocab_size = int(getattr(model.config, "vocab_size", 0))
    eos_token_id = scalar_token_id(getattr(tokenizer, "eos_token_id", None))

    lines = [
        "# qwen36-official-full-logit-vectors-v1",
        f"# model {args.model}",
        f"# revision {args.revision or 'default'}",
        f"# model_vocab_size {model_vocab_size}",
        f"# tokenizer_vocab_size {tokenizer_vocab_size}",
        f"# tokenizer_base_vocab_size {tokenizer_base_vocab_size}",
        f"# eos_token_id {eos_token_id if eos_token_id is not None else 'none'}",
        f"# transformers {transformers.__version__}",
        f"# torch {torch.__version__}",
        f"# dtype {args.dtype}",
        f"# device_map {args.device_map}",
        f"# attn_implementation {args.attn_implementation or 'default'}",
        f"# local_files_only {int(args.local_files_only)}",
        f"# seed {args.seed}",
        f"# steps {args.steps}",
        f"# top_k {args.top_k}",
        f"# full_logits {int(not args.no_full_logits)}",
        "# case <id> <ctx> <steps> <prompt-file>",
        "# render raw|chat-disabled|chat-enabled|chat-auto",
        "# prompt_tokens <count>, then tokens ... lines, then end_prompt_tokens",
        "# step <index> <selected-token-id> <selected-logit> <top-count>",
        "# top <token-id> <official-logit>",
        "# logits_f32_hex <vocab-size>, then logits_hex <u32-bits> lines, then end_logits",
        "",
    ]
    for case in args.case:
        lines.extend(
            capture_case(
                model,
                tokenizer,
                case,
                args.steps,
                args.top_k,
                args.tolerance,
                not args.no_full_logits,
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
