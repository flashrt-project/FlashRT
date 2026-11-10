# Qwen Trainium implementation provenance

Source: https://github.com/KevGomes1403/nki-moe-megakernel
Revision: 5879c39bb2479fee17eb8fccbc1f71058b846baf
License: Apache-2.0 (included verbatim in LICENSE).

Vendored files are the qwen3_6_moe model/driver and Python NKI kernels.
Local changes: imports relocated into FlashRT's private namespace; the driver
selects trn3 instead of trn2. Additional compatibility changes, if necessary,
are recorded in Git. Original performance claims describe upstream Trainium 2
experiments and are not FlashRT Trainium 3 results.

The optional FlashRT faithful-numerics path retains the original normalization
offset weights and adds one in FP32 at execution time, and preserves FP32
DeltaNet recurrent state between tokens. It currently supports ordinary greedy
decoding only; the original fused/speculative modes retain their upstream math.
