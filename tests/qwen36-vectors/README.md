# Qwen3.6 Official Vector Fixtures

This directory is for Qwen3.6-27B full-logit parity fixtures captured from the
official Hugging Face implementation. The compact `official.vec` file is not
checked in yet because it depends on the exact upstream checkpoint and is large
enough to regenerate when needed.

Capture a small fixture:

```sh
python3 tests/qwen36-vectors/capture_qwen36_vectors.py \
  --revision <pinned-hf-commit> \
  --case short:4096:chat-disabled:tests/qwen36-vectors/prompts/short.txt \
  --out tests/qwen36-vectors/official.vec
```

Run the native runtime against it:

```sh
QWEN36_TEST_MODEL=/path/to/qwen3.6-27b.gguf \
QWEN36_TEST_VECTOR_FILE=tests/qwen36-vectors/official.vec \
./ds4_test --qwen36-vectors
```

The broader text-only gate bundle is:

```sh
QWEN36_TEST_MODEL=/path/to/qwen3.6-27b.gguf \
QWEN36_TEST_VECTOR_FILE=tests/qwen36-vectors/official.vec \
QWEN36_REQUIRE_OFFICIAL_VECTORS=1 \
make qwen36-text-gates
```

Set `QWEN36_REQUIRE_OFFICIAL_VECTORS=1` in CI once the fixture is checked in or
mounted; without it, the test target skips when `official.vec` is absent. The
shortcut `make qwen36-text-gates-strict` sets that environment variable for the
same text-only gate bundle.

The fixture format is intentionally C-friendly:

- `case <id> <ctx> <steps> <prompt-file>`
- `render raw|chat-disabled|chat-enabled|chat-auto`
- `tolerance <absolute-logit-tolerance>`
- provenance comments for `model_vocab_size`, `tokenizer_vocab_size`, and
  `eos_token_id`
- `prompt_tokens <count>` followed by `tokens ...` lines
- `step <index> <selected-token-id> <selected-logit> <top-count>`
- `top <token-id> <official-logit>`
- `logits_f32_hex <vocab-size>` followed by `logits_hex <u32-bits> ...` lines
  and `end_logits`
- `end`

The test compares rendered prompt token IDs, greedy selected token IDs, selected
logits, the captured top logits, and every dense vocabulary logit when the
`logits_f32_hex` section is present. The capture script writes dense logits by
default; use `--no-full-logits` only for temporary top-k debugging fixtures.

The capture script writes provenance comments for the model id, revision,
model vocabulary size, tokenizer vocabulary size, tokenizer EOS token,
Transformers/Torch versions, dtype, device map, attention implementation, seed,
step count, top-k count, and dense-logit mode. Strict mode validates the format,
model, revision, vocabulary/tokenizer metadata, and dense-logit preamble before
loading the native model, then checks the captured vocab/EOS metadata against
the opened GGUF before comparing logits. That keeps parity failures traceable to
a concrete upstream checkpoint and software stack.
Regenerating `official.vec` for CI should always use `--revision <commit>` and
leave dense logits enabled.
