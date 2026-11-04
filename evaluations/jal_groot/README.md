# GR00T N1.7 on Thor: FlashRT and the Jetson AI Lab TensorRT baseline

This independent evaluation compares FlashRT with the [Jetson AI Lab (JAL) GR00T N1.7 TensorRT recipe](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/) on the same Thor. Measurements were collected on **October 26, 2026**. JAL supplies the deployment baseline; this is not an official FlashRT result or a standardized accuracy certification.

**FlashRT completed the matched inference call in 29.39 ms versus 33.36 ms for JAL TensorRT: 3.96 ms saved, 11.9% lower latency, or 1.135× speedup.** FlashRT also had lower average offline action deviation from the official PyTorch reference on the selected observations. The original one-camera pilot is retained below; a subsequent two-camera full-suite evaluation is reported separately.

## Measured results

Latency covers a complete call from a raw observation to decoded physical actions on the CPU, with GPU synchronization. It includes fresh preprocessing; it excludes model loading, calibration and graph preparation. Each backend contributed 900 measured calls over three rounds.

| Latency | JAL TensorRT | FlashRT |
|---|---:|---:|
| Median, all 900 calls | 33.36 ms | 29.39 ms |
| p95, all 900 calls | 34.23 ms | 29.86 ms |
| Round 1 median | 33.57 ms | 29.48 ms |
| Round 2 median | 33.20 ms | 29.37 ms |
| Round 3 median | 33.32 ms | 29.33 ms |

Accuracy uses **178 held-out observations from five trajectories and three task prompts**. The reference is the official GR00T implementation in eager BF16 mode, using the same one-camera configuration and supplied denoising noise as both optimized backends.

| Deviation from official PyTorch | JAL TensorRT | FlashRT |
|---|---:|---:|
| Mean absolute error (MAE) | 0.01591 | 0.01079 |
| Root mean square error (RMSE) | 0.05046 | 0.04727 |
| Maximum absolute error | 0.98438 | 0.99805 |
| Pooled cosine similarity | 0.98854 | 0.98992 |
| Gripper disagreement at threshold 0.5 | 1.69% | 1.40% |

These metrics compare all 16 returned action vectors per observation, totaling 2,848 vectors. MAE and RMSE pool the seven physical action coordinates, `x,y,z,roll,pitch,yaw,gripper`. Cosine is computed over the pooled action array. The coordinates have different meanings and scales; these scores are not percentages of accuracy or task success. FlashRT's lower average deviation does not mean every action or metric is better: its maximum absolute deviation is slightly larger.

Recorded demonstration actions provide a second reference. This comparison uses only the next eight executed actions, truncated at trajectory boundaries, totaling 1,410 action vectors.

| Error against recorded demonstration actions | Official PyTorch | JAL TensorRT | FlashRT |
|---|---:|---:|---:|
| MAE | 0.09042 | 0.09359 | 0.09068 |
| RMSE | 0.19651 | 0.19573 | 0.19645 |

Full-precision values, per-coordinate errors and per-trajectory summaries are in [results/comparison.json](results/comparison.json). Individual timing samples are in [results/latency-samples.json](results/latency-samples.json), and paired physical actions are in [results/action-predictions.csv](results/action-predictions.csv). The [data-only verification command](REPRODUCE.md#check-the-published-results-without-a-gpu) recomputes the offline metrics and latency statistics from these exports without downloading models or running inference. No numerical equivalence margin was agreed, so the report makes no accuracy pass/fail declaration.

## What was matched

The [frozen offline protocol](configs/protocol.json) defines the shared contract. The harness checks raw observations, processed pixels, token IDs, image grids, state, supplied noise, checkpoint configuration and weight-file hashes before accepting paired results.

| Setting | Value |
|---|---|
| Checkpoint | `nvidia/GR00T-N1.7-LIBERO`, `libero_10` |
| Camera input | One camera: `image` |
| Batch size | 1 |
| Denoising steps | 4 |
| Internal action horizon | 40 |
| Noise shape | `(1, 40, 132)` |
| Returned physical actions | `(1, 16, 7)` |
| Execution horizon | 8 |
| Shared noise | Seed `20261026`, generated per record, rounded to BF16 |
| CPU threads | 2 |

**The checkpoint's default processor uses two cameras, `image` and `wrist_image`.** This experiment preserves JAL's one-camera performance configuration through a shared overlay for all three backends. Its results describe that configuration; they do not establish that omitting the wrist camera preserves the original policy's task success.

The [JAL adapter](groot_compare/backends/jal.py) requires the optimized TensorRT engines and rejects graph fallback. The [FlashRT adapter](groot_compare/backends/flashrt.py) processes fresh images and state through FlashRT and uses the shared physical-action decoder. The official model is used separately to capture FlashRT calibration features before evaluation.

## Evaluation and calibration split

The dataset is `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot` at the revision recorded below. Observations are sampled every eight frames, with a maximum of 300 frames per evaluation trajectory.

| Use | Episodes / frames | Selection |
|---|---|---|
| Held-out evaluation | Episodes **1, 2, 3, 4, 22** | 178 observations, three task prompts |
| JAL TensorRT calibration | Episode **0**, frames **0–9** | Original sequential JAL loader, `calib_size=10` |
| FlashRT calibration | Episodes **18, 5, 34** | First exported observation per prompt; one sample each |

Both calibration sets are excluded from the primary evaluation cohort. Episode 22 replaces episode 0 from the original JAL demonstration cohort with another episode of the same task. The original JAL cohort, episodes 0–4, overlaps TensorRT calibration and is not the held-out result reported here.

Calibration was frozen before evaluation. FlashRT retains prompt-specific calibration and graph preparation; an unseen instruction requires that setup again. The results therefore describe steady-state operation with prepared instructions, not first-call latency or arbitrary prompt-switching cost.

The implementations retain their own optimization recipes: JAL uses FP8 vision with NVFP4 LLM/DiT components, while FlashRT uses an FP8 backbone and an NVFP4 action head. Calibration algorithms and precision allocation differ. This evaluates the deployed recipes on matched inputs; it does not isolate one quantization algorithm.

## Timing procedure and platform

Timing cycles through the held-out observations for each prompt, with 20 warmups followed by 100 measured calls per prompt per round. The three rounds alternate execution order: TensorRT → FlashRT, FlashRT → TensorRT, then TensorRT → FlashRT. Each backend therefore has `3 prompts × 100 calls × 3 rounds = 900` samples.

The [runner](groot_compare/run.py) measures synchronized wall-clock time around one `get_action` call. Preprocessing, input transfers, transfer of the supplied noise, inference, physical-action decoding and the return of CPU actions are inside the timer. File loading, noise generation, model construction, calibration, graph capture and warmup are outside it. Output validation is also outside the timer for both backends.

| Platform setting | Recorded value |
|---|---|
| Device | NVIDIA Jetson AGX Thor Developer Kit, T5000 |
| JetPack / L4T | 7.2 / 39.2.1 |
| Operating system | Ubuntu 24.04.4, aarch64 |
| Power mode | MAXN |
| Fixed CPU / GPU GPC / memory clocks | 2.601 GHz / 1.575 GHz / 4.266 GHz |
| Competing workloads | Stopped during measurement |

The software environments are not identical. JAL and the PyTorch reference use PyTorch 2.10.0, TensorRT 10.15.1.29 and ModelOpt 0.39.0 in the JAL runtime. FlashRT uses its pinned public Thor image with NVIDIA PyTorch `2.12.0a0+5aff3928d8.nv26.5.50603568`. Runtime and artifact identities are retained in [results/provenance.json](results/provenance.json). These results include the effects of those deployment environments.

The JAL tutorial's published “E2E” value sums separately timed preprocessing, backbone and action-head stages. This harness measures a single complete call and includes final decoding and CPU output transfer. The tutorial also uses a different iteration/warmup count. The paired measurements above are the comparison; dividing a separately published headline by a different timer is not the measured speedup.

## Relation to FlashRT's published numbers

At the [FlashRT revision tested](https://github.com/flashrt-project/FlashRT/blob/e4bd9277521d137fcded8fa5ccb6091c05f5cd4b/README.md#groot-n17), the Thor LIBERO one-camera row reports **23.23 ms model latency** and **28.53 ms complete-call latency**. This experiment's **29.39 ms complete-call median** is 0.86 ms, or 3.0%, higher than that published complete-call value. It is within the **28–30 ms** range in the [pinned Thor guide](https://github.com/flashrt-project/FlashRT/blob/e4bd9277521d137fcded8fa5ccb6091c05f5cd4b/docs/thor/groot-n17.md). These are separate runs, not an exact replication of the guide's single observation.

The guide's numerical check uses one bundled observation and a combined-action cosine pass threshold of 0.999. This experiment measures pooled cosine across 178 held-out observations with separate calibration data. Its 0.98992 FlashRT cosine is a different test and aggregation; it neither reproduces nor directly invalidates that single-example check. Neither numerical check establishes robot task success.

## Two-camera LIBERO-10 follow-up

The FlashRT maintainer completed **ten full LIBERO-10 rounds per backend** with the checkpoint's intended **two cameras**, `image` and `wrist_image`. Each round covers all ten tasks and all 50 official initial states per task, with a different seed. The same initial-state set is reused between rounds.

| Backend | Successful / evaluated episodes | Mean task success ± run-to-run SD |
|---|---:|---:|
| Official PyTorch | 4,457 / 5,000 | **89.14% ± 1.57 pp** |
| JAL TensorRT | 4,441 / 5,000 | **88.82% ± 1.10 pp** |
| FlashRT | 4,463 / 5,000 | **89.26% ± 1.01 pp** |

All ten rounds are included. SD is the sample standard deviation of the ten round success rates, in percentage points. See [per-round counts and runtime details](results/libero-ten-rounds.json); run and verify the full suite with [SIMULATOR.md](SIMULATOR.md#full-suite-run-and-check).

Restoring both views resolved the earlier 0/3 failure in a local camera check: official PyTorch changed from 0/3 with one view to 3/3 with both; JAL and FlashRT also completed those three two-camera cases. The ten-round follow-up uses four denoising steps, eight executed actions per decision, five settling steps and a 720-step budget. Calibration stays frozen during evaluation.

These are follow-up task results, separate from the original one-camera latency and offline-action measurements above. Runs used existing Thor environments (PyTorch 2.12 on one device and 2.9 on the other, TensorRT 10.16); JAL used ten official demo observations for calibration. Exact allocation and differences from the original NVIDIA environment are recorded with the counts.

## Fixed-state simulation pilot

A separate LIBERO-10 pilot tested task IDs **4, 6 and 9**, one episode per task, using fixed initial state index 0. The official fixed `.pruned_init` state and five zero-action settling steps were followed by a 720-step GR00T budget and execution horizon 8. Initial simulator states, initial processed observations and supplied per-decision noise matched across backends.

| Backend | Successful episodes |
|---|---:|
| Official PyTorch | 0 / 3 |
| JAL TensorRT | 0 / 3 |
| FlashRT | 0 / 3 |

Every episode reached the 720-step budget. This is an inconclusive pilot: it cannot distinguish task performance between implementations or establish accuracy equivalence. It is not a full LIBERO-10 score, and it does not identify the cause of failure. In particular, the effect of omitting the wrist camera was not isolated.

See [results/closed-loop-pilot.json](results/closed-loop-pilot.json) and [SIMULATOR.md](SIMULATOR.md). The offline protocol's `closed_loop.status` describes the original offline experiment; the pilot has its own protocol in its result file.

## Pinned sources

| Component | Revision or digest |
|---|---|
| FlashRT | `e4bd9277521d137fcded8fa5ccb6091c05f5cd4b` |
| Isaac-GR00T | `9c7e746b2cd37a810070a98ef41d290a07e806c2` plus the JAL patch |
| JAL patch SHA256 | `6a818bfcbb149c0f0efa8567ecc0fd6909613bc53e00e08f66bb81c8c601d5a3` |
| GR00T-N1.7-LIBERO weights | `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21` |
| Cosmos-Reason2-2B assets | `9ce19a195e423419c349abfc86fd07178b230561` |
| LIBERO LeRobot dataset | `e1a223d30b896c1613f270a2bfc63d382b3de7e1` |
| LIBERO simulator | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |

FlashRT image: `ghcr.io/flashrt-project/flashrt-thor@sha256:87ad7bce8635207425ac5e90a61d3d83c0ba1042fa28eef7d067474fee872c5c`.

## Reproduction and interpretation

[REPRODUCE.md](REPRODUCE.md) documents setup, input export, calibration, engine building, paired evaluation and report generation. The comparison code, result summaries, action CSV and timing samples are included here; weights, dataset images, generated engines and calibration tensors are not committed. [results/provenance.json](results/provenance.json) records artifact identities and hashes; [results/observations.json](results/observations.json) retains the paired input fingerprints. The [result-file checksums](results/SHA256SUMS) cover the published evidence files.

The original JAL run used a locally derived runtime image. An independent public-base build recipe is supplied, but that fresh-machine image has not been built and verified against the measured runtime. Reproduction on another machine still needs that validation; the published results do not imply a tested one-command setup.

The completed comparison harness passed 45 CPU checks and ran the offline accuracy evaluation, all three timing rounds and the simulation pilot on Thor. Only complete matched runs enter the published report. An early TensorRT adapter buffer-update failure was fixed before the complete accuracy run and every timing round; incomplete debug output is excluded. ViT and LLM engines were rebuilt after stopping automatically restarted applications to ensure isolated builds.

The evidence supports a modest steady-state latency advantage and lower average offline action deviation for FlashRT on this cohort. The two-camera follow-up above adds full-suite task-success evidence for the tested rollout settings. Review of the FlashRT adapter, prompt preparation and calibration choices is welcome before extending that claim.
