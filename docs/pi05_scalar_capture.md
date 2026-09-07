# Scalar timestep capture compatibility

Target: LeRobot PI0.5 `sample_actions`, FP32 scalar timestep to fixed-step
denoising, batch 1, action horizon 10, padded action width 32, three RGB views.
The boundary is host graph lowering, not a new structure or kernel. The
consumer remains the original host method; Nexus only receives its later export.

## Coverage and ownership

The existing `bindings/pi05_tick.yaml` supplies the complete hot-path map.
Native symbols below refer to `flash_rt/models/pi05/pipeline_rtx.py`.
This fix qualifies only scalar construction, not the other regions' performance.

| Native region / symbol | Existing owner | Classification for this change |
|---|---|---|
| `input_images_buf`, `set_language_embeds` | host input processing | intentionally_retained |
| `vision_encoder`, `_vision_layer` | vision FFN, QKV, attention, normalization | configured, unchanged |
| vision projector in `vision_encoder` | linear projection | configured, unchanged |
| `transformer_encoder`, `_encoder_layer` | prefix FFN, QKV, attention, normalization | configured, unchanged |
| encoder K/V buffers | prefix state | host_stage_or_state, unchanged |
| `transformer_decoder` fixed-step loop | host schedule lowering | host_stage_or_state, scalar construction under test |
| decoder time/style tables | adaptive-normalization producer | configured, unchanged |
| decoder action input projection | linear projection | configured, unchanged |
| `_decoder_layer` | denoise transformer structures | configured, unchanged |
| final adaptive norm and action projection | normalization and projection | configured, unchanged |
| `input_noise_buf` | action/noise window | host_stage_or_state, unchanged |

LeRobot constructs `torch.tensor(time, dtype=float32, device=device)` each
step. OpenPI's PyTorch `sample_actions` also constructs scalar `dt` and initial
`time`; the broader fixed-loop normalization remains a separate concern.
The list-shaped schedule already has a resident-cache path. A scalar must
instead be constructed device-natively: caching a mutable scalar could retain
the previous call's updates. No unrelated-model catalog generalization is made.

No Hub API is missing at this boundary: PyTorch's `full` supplies the scalar
construction. Existing Hub kernels, calibration, precision and routing are
unchanged. CPU contracts cover scalar shape/dtype/value, list preservation,
recognition and restoration. GPU gates cover eager and compiled capture plus
repeated changing inputs. Full-model parity and task acceptance are separate
gates; passing the scalar contract alone does not certify the deployment.
