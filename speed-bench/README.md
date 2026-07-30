## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

The Laguna S 2.1 reference measurements were recorded on NVIDIA GB10 on
2026-07-30. The native rows use the direct safetensors CUDA implementation:

- `laguna_s21_nvfp4_gb10.csv` records ordinary prefill and decode at the
  2,048-token comparison frontier.
- `laguna_s21_nvfp4_dflash_gb10.csv` compares ordinary decode with the official
  NVFP4 DFlash drafter at fixed depth 7.
- `laguna_s21_dflash_quant_comparison_gb10.csv` compares Q4_K_M and native
  NVFP4 targets, each with its matching official drafter, on the same
  2,048-token prompt and 256-token greedy decode.
- `laguna_s21_nvfp4_sm121_optimizations_gb10.csv` records the isolated
  SM121 kernel A/Bs and matched end-to-end before/after measurements.

The DFlash CSVs include verifier and proposal-acceptance counters because their
throughput is prompt-sensitive. The official Laguna drafter is selected with
`--dflash`; `--mtp` remains the separate DeepSeek support-model interface.

The current 2K/256 results are:

| target | mode | generation tok/s | speedup vs raw |
| --- | --- | ---: | ---: |
| Q4_K_M | raw | 22.32 | 1.00x |
| Q4_K_M | BF16 DFlash, fixed 15 | 23.42 | 1.05x |
| native NVFP4 | raw | 18.61 | 1.00x |
| native NVFP4 | official native DFlash, fixed 7 | 36.82 | 1.98x |

On the matched pre-optimization run, native raw decode was 15.09 token/s
(15.16 steady) and fixed-depth DFlash was 34.86 token/s (35.14 steady).
The SM121 paths raise those to 18.61/18.71 and 36.82/37.05 token/s,
respectively. DFlash verifier-cycle latency fell from 175.53 ms to
161.22 ms even though proposal acceptance was lower in the optimized run.
For latency rows in the optimization CSV, `improvement_pct` is the reduction
in elapsed time.

Nsight attributes nearly all raw-decode gain to the intended kernels: BF16
projection time fell from 44.88 to 33.57 ms/token, and native MoE fell from
13.64 to 13.37 ms/token. On real-model prefill, normalized grouped gate/up plus
three down launches fell from 19.64 to 15.64 ms per sparse layer.

The optimized paths retain independent A/B controls:

- `DS4_CUDA_LAGUNA_NO_BF16_LT=1` disables the SM121 `n=1` cuBLASLt plans.
- `DS4_CUDA_LAGUNA_NO_BF16_REALIGN=1` preserves original BF16 cache offsets.
- `DS4_CUDA_NVFP4_GROUPED_STAGE_ACTIVATION=0` disables grouped CTA staging.

Run the native rows with:

```sh
./ds4-bench --cuda \
  -m /srv/models/poolside/Laguna-S-2.1-NVFP4 \
  --prompt-file ds4.c --ctx-start 2048 --ctx-max 2048 \
  --ctx-alloc 4096 --gen-tokens 256 --csv native-raw.csv

DS4_DFLASH_ADAPTIVE=0 ./ds4-bench --cuda \
  -m /srv/models/poolside/Laguna-S-2.1-NVFP4 \
  --dflash /srv/models/poolside/Laguna-S-2.1-DFlash-NVFP4 \
  --dflash-draft 7 --dflash-p-min 0 \
  --prompt-file ds4.c --ctx-start 2048 --ctx-max 2048 \
  --ctx-alloc 4096 --gen-tokens 256 --csv native-dflash.csv
```

The native drafter's BF16 spans are eagerly prepared. The converted GGUF BF16
drafter instead has a one-time first-block page-fault cost, so its steady 2K
row uses a 16-token warmup frontier in the same process:

```sh
DS4_DFLASH_ADAPTIVE=0 ./ds4-bench --cuda \
  -m /srv/models/poolside/Laguna-S-2.1-GGUF/laguna-s-2.1-Q4_K_M.gguf \
  --dflash /srv/models/poolside/Laguna-S-2.1-GGUF/laguna-s-2.1-DFlash-BF16.gguf \
  --dflash-draft 15 --dflash-p-min 0 \
  --prompt-file ds4.c --ctx-start 16 --ctx-max 2048 --step-incr 2032 \
  --ctx-alloc 4096 --gen-tokens 256 --csv q4-dflash.csv
```

Use the `ctx_tokens=2048` row from that CSV. Without the warmup, the measured
cold run was 15.47 token/s because its first verifier block paid the anonymous
F16-shadow fault.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.
