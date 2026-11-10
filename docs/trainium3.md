# Trainium 3 backend (experimental)

FlashRT provides JAX/PJRT π0/π0.5 inference and NxDI/NKI Qwen3.6-35B-A3B
text inference under `flash_rt.neuron`. These paths do not load CUDA kernels.
They require an installed AWS Neuron SDK compatible with the node's driver and
Trainium 3. CPU fallback is rejected by the production frontends.

## π0 and π0.5

Use the Neuron JAX environment and install FlashRT's Python package plus
`torch` (CPU checkpoint reader), `safetensors` and `sentencepiece`. Keep
`neuronx-cc` on PATH. Select devices before starting Python:

```bash
export NEURON_RT_VISIBLE_CORES=0
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_ULTRASERVER_MODE=4
export JAX_PLATFORMS=neuron,cpu
```

```python
import flash_rt
model = flash_rt.load_model(
    checkpoint='/path/to/pi0_libero', config='pi0', framework='jax',
    hardware='neuron', precision='bf16', num_views=2, num_steps=10,
)
actions = model.predict(images=[base_rgb, wrist_rgb], prompt='pick up the cup', state=robot_state)
```

Images are uint8 HWC at 224×224. π0 requires robot state. `pi05` currently
uses the stateless prompt contract of OpenPI's LIBERO policy; state-in-text
deployment recipes are not supported. The checkpoint must supply matching
normalization statistics for robot-space prediction. Old LeRobot checkpoints
with embedded mean/std and OpenPI `norm_stats.json` are supported.

For checkpoints distributed without normalization, use
`model.pipeline.infer_normalized(images_chw, tokens, normalized_state, noise)`.
This returns a JAX array of normalized 32-dimensional actions. Tokens are a
compact sequence without padding; noise has shape `[action_horizon, 32]`.
Use `.block_until_ready()` for device timing and `numpy.asarray(...)` to include
the result transfer. Normalized tensor inference is not robot-space control.

The entire 27-layer vision encoder, 18-layer language prefix and 18-layer
action expert are included. JAX loops share compiled layer bodies; the prefix
KV cache is reused across all denoising steps. Initial compilation and weight
loading must be excluded from steady-state timings and reported separately.

## Qwen3.6-35B-A3B

Use a separate Neuron PyTorch/NxDI environment. The initial compatibility
target is `torch-neuronx==2.9.0.2.15.32035+de43f57c` and
`neuronx-distributed-inference==0.10.18399+ed62453e` from the Neuron package
index. Package compatibility and performance are being qualified.

```bash
export NEURON_RT_VISIBLE_CORES=0-3
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_ULTRASERVER_MODE=4
export NEURON_PLATFORM_TARGET_OVERRIDE=trn3
```

```python
from flash_rt.neuron.qwen36 import Qwen36NeuronFrontend
model = Qwen36NeuronFrontend('/path/to/Qwen3.6-35B-A3B', '/path/to/compiled/trn3-greedy',
                            tp_degree=4, seq_len=128)
result = model.generate('The capital of France is', max_new_tokens=32)
```

TP=4/LNC=2 uses two Trainium 3 chips, not the entire 16-chip node. This
interface runs only the language model; image/video inputs are not supported.
Greedy decode is the baseline. Optional layer kernels and fused speculative
decoding require separate correctness and timing qualification.

The model and NKI kernels derive from the Apache-2.0
[upstream Trainium implementation](https://github.com/KevGomes1403/nki-moe-megakernel/tree/5879c39bb2479fee17eb8fccbc1f71058b846baf/megakernels/qwen3_6_moe).
See the included `flash_rt/neuron/_vendor/qwen36/VENDOR.md` and license.
Upstream Trainium 2 results must not be presented as Trainium 3 measurements.
