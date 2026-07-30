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

The DFlash CSVs include verifier and proposal-acceptance counters because their
throughput is prompt-sensitive. The official Laguna drafter is selected with
`--dflash`; `--mtp` remains the separate DeepSeek support-model interface.

The current 2K/256 results are:

| target | mode | generation tok/s | speedup vs raw |
| --- | --- | ---: | ---: |
| Q4_K_M | raw | 22.32 | 1.00x |
| Q4_K_M | BF16 DFlash, fixed 15 | 23.42 | 1.05x |
| native NVFP4 | raw | 15.58 | 1.00x |
| native NVFP4 | official native DFlash, fixed 7 | 35.05 | 2.25x |

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
