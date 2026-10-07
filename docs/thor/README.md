# FlashRT on Jetson Thor

Choose a model tutorial. Each provides complete commands for **Docker** and
**cloning, installing and compiling from source**, followed by result checks.

| Model | Tutorial | Public-image FP4 result |
|---|---|---|
| OpenPI π0.5 LIBERO | [Run and verify](pi05.md) | about 20 ms |
| GR00T N1.7 LIBERO | [Run and verify](groot-n17.md) | about 24.3 ms model / 27.9 ms complete call |

The public image is `ghcr.io/flashrt-project/flashrt-thor:thor-v0.1.1`.
It includes compiled kernels, reference environments and verification fixtures;
weights are downloaded into your own model directory.

- [Run both models with Docker](docker.md)
- [Recorded measurements and reports](results.md)
- [Maintainer image publication](release.md)

The source commands select [PR #219's branch](https://github.com/flashrt-project/FlashRT/pull/219)
until it is merged. Use a new working directory and virtual environment for native setup.
