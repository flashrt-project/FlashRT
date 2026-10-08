# Demo gallery

[Back to README](../README.md) · [VLA](#vla) · [VLM](#vlm) · [LLM](#llm) · [Video](#video)

Each film is one checkpoint executed several ways, side by side on **one wall
clock**, each pane advancing at the rate it was actually measured at. The GIFs
below are sped up to fit the page; the [real-time recordings](https://github.com/flashrt-project/FlashRT-assets/tree/main/demo/mp4) and the
[full walkthrough](https://huggingface.co/spaces/liangsu9988/fast-kernels-are-not-fast-pipelines) play one second per second.

## VLA

### MindOn Mind-1: real-robot deployment

[Watch the MindOn demo](https://www.youtube.com/watch?v=SsNYtZJZyLM) ·
[Read the Mind-1 blog](https://www.mindon.tech/blog/mind-1/index.html).

[MindOn](https://www.mindon.tech/)'s work spans the model, training, control, and real-robot deployment.
FlashRT contributes one part of their inference stack.

### FlashRT policy comparisons

**π0.5 on a Jetson AGX Thor.** LIBERO-spatial task 1, one flow-matching sample
held fixed. OpenPI and LeRobot hosts as shipped, plus FlashRT structures and
native inference. Median per-decision latency **266.0 / 303.6 → 36.1 → 22.2 ms**,
3.8 Hz to **45.1 Hz** of policy decisions. On the shared policy + 20 Hz control
timeline, the task completes at **5.4 s** versus 23.5 s; all four arms succeed.

<img src="https://github.com/flashrt-project/FlashRT-assets/raw/main/demo/gif/thor_pi05.gif" alt="thor_pi05" width="100%">

<sub>2× playback. <a href="https://github.com/flashrt-project/FlashRT-assets/blob/main/demo/mp4/thor_pi05.mp4">Real-time recording</a>.</sub>

<details>
<summary><strong>More VLA films</strong> — one checkpoint four ways, and the same checkpoint on two hosts</summary>

**π0.5 · RTX 5090 — four ways of executing one checkpoint.** The host as
shipped, the same host compiled, the same host with structures attached, and
the hand-written FlashRT pipeline. **107.5 → 58.9 → 25.6 → 21.6 ms**, i.e.
9.3 → 46.5 Hz. Same task, same initial state in every pane.

<img src="https://github.com/flashrt-project/FlashRT-assets/raw/main/demo/gif/pi05_race.gif" alt="pi05_race" width="100%">

<sub>1.5× playback. <a href="https://github.com/flashrt-project/FlashRT-assets/blob/main/demo/mp4/pi05_race.mp4">Real-time recording</a>.</sub>

**GR00T N1.7 · RTX 5090.** LIBERO-10 task 4, two camera views.
**44.4 → 25.4 → 17.8 → 15.8 ms**, 22.5 → 63.1 Hz.

<img src="https://github.com/flashrt-project/FlashRT-assets/raw/main/demo/gif/groot_race.gif" alt="groot_race" width="100%">

<sub>3× playback. <a href="https://github.com/flashrt-project/FlashRT-assets/blob/main/demo/mp4/groot_race.mp4">Real-time recording</a>.</sub>

**GR00T N1.7 · Jetson AGX Thor.** The same checkpoint and one flow-matching
sample shared by all four panes. **109.5 / 154.2 → 28.4 → 28.1 ms** — the
explicit structure book and the hand-written pipeline land 0.25 ms apart.

<img src="https://github.com/flashrt-project/FlashRT-assets/raw/main/demo/gif/thor_groot.gif" alt="thor_groot" width="100%">

<sub>5× playback. <a href="https://github.com/flashrt-project/FlashRT-assets/blob/main/demo/mp4/thor_groot.mp4">Real-time recording</a>.</sub>

**π0.5 under two independent hosts.** LeRobot **107.5 → 25.6 ms**, OpenPI
**41.7 → 28.8 ms**. Two hosts that start 2.58× apart end within 12% of each
other; each is measured against the form its own authors ship.

<img src="https://github.com/flashrt-project/FlashRT-assets/raw/main/demo/gif/pi05_cross.gif" alt="pi05_cross" width="100%">

<sub>1.5× playback. <a href="https://github.com/flashrt-project/FlashRT-assets/blob/main/demo/mp4/pi05_cross.mp4">Real-time recording</a>.</sub>

**GR00T N1.7 under Isaac-GR00T and the LeRobot port.** One NVIDIA checkpoint,
two hosts: **44.4 → 17.8 ms** and **41.8 → 18.6 ms**. The two hosts agree to a
cosine of 0.999995 on the one decision both made from a byte-identical
observation.

<img src="https://github.com/flashrt-project/FlashRT-assets/raw/main/demo/gif/groot_cross.gif" alt="groot_cross" width="100%">

<sub>3× playback. <a href="https://github.com/flashrt-project/FlashRT-assets/blob/main/demo/mp4/groot_cross.mp4">Real-time recording</a>.</sub>

</details>

## VLM

**Qwen3-VL-8B on the unedited `transformers` host.** One image, one prompt,
greedy. Three arms: the host as shipped, the host compiled with a static cache,
and the host with structures attached. Decode **65.6 → 82.9 → 145.7 tok/s**,
**1.76×** over the host's own compiled form; TTFT ~30 ms in all three. 180
seats, no host edit.

<img src="https://github.com/flashrt-project/FlashRT-assets/raw/main/demo/gif/qwen3vl.gif" alt="qwen3vl" width="100%">

<sub>1× playback. <a href="https://github.com/flashrt-project/FlashRT-assets/blob/main/demo/mp4/qwen3vl.mp4">Real-time recording</a>.</sub>

## LLM

**Qwen3.6-35B-A3B on one 32 GB card.** 67 GB of BF16 weights do not fit;
`quantize_on_adopt` regrids the expert banks before the model reaches the
device (22.3 GiB resident) and the model runs. Decode **51.6 → 203.5 → 284.9
tok/s**, the third pane being the checkpoint's own draft head at 3.43 tokens
accepted per round — shown on the prompt where it wins, dropped on the one
where it does not.

<img src="https://github.com/flashrt-project/FlashRT-assets/raw/main/demo/gif/qwen36.gif" alt="qwen36" width="100%">

<sub>1× playback. <a href="https://github.com/flashrt-project/FlashRT-assets/blob/main/demo/mp4/qwen36.mp4">Real-time recording</a>.</sub>

**Attached inside vLLM, under batch.** The same 35B mixture-of-experts served
by vLLM on a Jetson AGX Thor at 1, 4, 8 and 16 concurrent requests — four
chapters in one file, each pane a live stream. Aggregate throughput
**38.4 → 77.4**, **77.5 → 182.8**, **98.9 → 245.5**, **211.4 → 302.1 tok/s**;
every level gains. A routed mixture of experts does not dilute the way a dense
model does — each token reads its own experts, so expert weight traffic grows
with the batch instead of being shared — which is why the gain climbs to batch
8 rather than falling away.

<img src="https://github.com/flashrt-project/FlashRT-assets/raw/main/demo/gif/thor_concurrency.gif" alt="thor_concurrency" width="100%">

<sub>3.5× playback. <a href="https://github.com/flashrt-project/FlashRT-assets/blob/main/demo/mp4/thor_concurrency.mp4">Real-time recording</a>.</sub>

<details>
<summary><strong>More LLM films</strong> — inside vLLM and SGLang, single stream</summary>

**Inside two serving engines.** Qwen3-8B, single stream, 144 seats bound by a
hook that fires after the engine loads and before its first trace. vLLM
**99.1 → 145.3 tok/s**, SGLang **101.2 → 203.0 tok/s**, TTFT falls in both.
Neither engine is forked: each keeps its scheduler, its memory planner and its
own graph, and the seats go inside that.

<img src="https://github.com/flashrt-project/FlashRT-assets/raw/main/demo/gif/engines.gif" alt="engines" width="100%">

<sub>1× playback. <a href="https://github.com/flashrt-project/FlashRT-assets/blob/main/demo/mp4/engines.mp4">Real-time recording</a>.</sub>

</details>

## Video

**Wan2.2 TI2V-5B — one clip, four ways of making it.** 480×480, 33 frames, 20
denoise steps, one prompt and one seed, all four arms in one process against
one baseline. **6.48 → 5.23 → 2.11 → 1.68 s**; per transformer call
**162.2 → 41.8 ms**. The third arm runs 4-bit on every call whose own per-step
score holds the band and hands the rest to FP8 — 171 calls at four bits, 50 at
eight, decided per call by measurement. Every pane plays its own clip at the
end.

<img src="https://github.com/flashrt-project/FlashRT-assets/raw/main/demo/gif/wan22.gif" alt="wan22" width="100%">

<sub>1× playback. <a href="https://github.com/flashrt-project/FlashRT-assets/blob/main/demo/mp4/wan22.mp4">Real-time recording</a>.</sub>

---
