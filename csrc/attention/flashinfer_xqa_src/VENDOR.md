FlashInfer XQA subset
=====================

Source: https://github.com/flashinfer-ai/flashinfer
Imported commit: bff85f3

This directory vendors only the XQA files needed by the Qwen3.6 SM120
BF16-query / FP8-KV speculative attention probe. The build instantiates a
single fixed shape in CMake:

- head_dim = 256
- Q heads / KV heads = 24 / 4
- page size = 128
- BF16 Q/O, FP8 e4m3 KV
- speculative decode enabled

Do not add the full FlashInfer runtime here. New files should be imported
only when a benchmark proves they are needed by the production Qwen path.
