# Qwen Trainium implementation provenance

Source: https://github.com/KevGomes1403/nki-moe-megakernel
Revision: 5879c39bb2479fee17eb8fccbc1f71058b846baf
License: Apache-2.0 (included verbatim in LICENSE).

Vendored files are the qwen3_6_moe model/driver and Python NKI kernels.
Local changes: imports relocated into FlashRT's private namespace; the driver
selects trn3 instead of trn2. Additional compatibility changes, if necessary,
are recorded in Git. Original performance claims describe upstream Trainium 2
experiments and are not FlashRT Trainium 3 results.

FlashRT also supports a separate short prefill bucket while retaining the full
decode window. Small prefill buckets select NxDI's all-experts path without
discarding experts or tokens.
