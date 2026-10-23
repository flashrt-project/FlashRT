# SageAttention2-derived Blackwell attention

This directory contains an Apache-2.0 adaptation of
[SageAttention](https://github.com/thu-ml/SageAttention)'s QK-int8 / SV-FP8
and SV-FP16 cores. The immediate source is the
[FlashRT-HF-kernels Sage2 package](https://github.com/flashrt-project/FlashRT-HF-kernels/tree/059d28393239c8fec04699bccaed6f4c88d25f24/sageattention2-blackwell).
The inherited SageAttention copyright and license headers are preserved.

The MMA, shared-memory and asynchronous-copy helpers also credit
[FlashInfer](https://github.com/flashinfer-ai/flashinfer); `mma.cuh` names its
FlashInfer v0.1.5 basis. `registration.h` uses the registration macros from
[vLLM](https://github.com/vllm-project/vllm/blob/aed08d76a682f1475957c0730ea1f5ecb150fda0/csrc/core/registration.h).
These projects remain the owners of their respective contributions.

FlashRT changes add causal-prefix offsets, log-sum-exp output for folding
attention sinks, and BF16/GQA Tensor wrappers. The separately compiled
producer is loaded through `flash_rt.native_kernels.sage2`; the Structures
consumer does not compile or locate source at inference time.

See [LICENSE](LICENSE), [NOTICE](NOTICE), and the per-file notices. This is
an adapted SageAttention2 execution form, not a claim of authorship of the
upstream attention algorithms or a SageAttention3 implementation.
