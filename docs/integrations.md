# FlashRT integrations

FlashRT provides adapters and kernel implementations for robot-policy frameworks, model libraries, and serving runtimes. Use the corresponding integration rather than assuming every host exposes the same API or supports every model.

| Project | Implementation and usage |
|---|---|
| HF Kernels | [Kernel packages](https://huggingface.co/flashrt) and [source and usage](https://github.com/flashrt-project/FlashRT-HF-kernels) |
| LeRobot, OpenPI, Isaac GR00T | [FlashRT-Structures host adapters](https://github.com/flashrt-project/FlashRT-Structures); standalone Thor reproduction: [OpenPI π0.5](thor/pi05.md) and [GR00T N1.7](thor/groot-n17.md) |
| Transformers, Diffusers | [FlashRT-Structures adapters and examples](https://github.com/flashrt-project/FlashRT-Structures/tree/main/examples), using the [FlashRT kernel catalog](kernel_catalog.md) |
| vLLM, SGLang | [FlashRT-Structures serving integrations](https://github.com/flashrt-project/FlashRT-Structures); [recorded demos](demos.md#llm) |

## Community integrations

[EagleVLA-Edge](https://github.com/PKU-SEC-Lab/EagleVLA-Edge) is PKU-SEC-Lab's llama.cpp-based onboard VLA inference engine. Their C API provider exposes PI0/PI0.5 GGUF inference to FlashRT's Python model interface without starting the foreground HTTP server. See their [README](https://github.com/PKU-SEC-Lab/EagleVLA-Edge#readme) for setup and supported interfaces. Their GR00T N1.7 support uses separate HTTP and C APIs; their FlashRT provider currently covers PI0 and PI0.5.
