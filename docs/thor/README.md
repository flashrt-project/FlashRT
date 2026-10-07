# FlashRT on Jetson Thor

Start with [π0.5 / OpenPI](pi05.md) or [GR00T N1.7](groot-n17.md). Both guides use fresh official references and save accuracy reports. [Tested results](results.md).

## 1. Prepare Thor

Tested: Ubuntu 24.04, JetPack/L4T R39.2.1, CUDA 13.2, ARM64 Thor. Record the device state before timing:

```bash
nvpmodel -q
cat /etc/nv_tegra_release
nvidia-smi
nvcc --version
sudo jetson_clocks --show
```

Use the same power mode, clocks and cooling for both implementations. Keep manufacturer protection settings. The supplied machine had a modified power-protection setting, so its latency is not a stock-device claim.

## 2. Choose installation

### Docker

Docker with NVIDIA Container Toolkit must already be installed. Commands assume your account can run Docker; otherwise run them with `sudo`. Use the toolkit's [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for a fresh machine.

A public Thor image has not been published yet. Until then, clone this repository, enter `repro/thor`, and build on Thor:

```bash
git clone https://github.com/flashrt-project/FlashRT.git
cd FlashRT/repro/thor
docker build --build-arg PIP_INDEX_URL -f Dockerfile.thor -t flashrt-jal:thor .
export IMAGE=flashrt-jal:thor
export MODELS="$PWD/models"
export OUT="$PWD/results"
mkdir -p "$MODELS" "$OUT"
```

Once released, replace the build command with `docker pull "$IMAGE"`, using the published image name. Runtime, reference environments, fixtures and validators are inside the image; mount only model and result directories. See [image release status](docker.md).

### Native: clone, install and compile

Clone this repository and enter `repro/thor`. Install JetPack first, then:

```bash
git clone https://github.com/flashrt-project/FlashRT.git
cd FlashRT/repro/thor
sudo apt-get update
sudo apt-get install -y python3.12-venv python3-dev build-essential cmake ninja-build git git-lfs ffmpeg libglib2.0-0 libgl1
python3.12 -m venv flashrt-venv
source flashrt-venv/bin/activate
python -m pip install 'torch==2.14.0+cu130' 'torchvision==0.29.0+cu130' --index-url https://download.pytorch.org/whl/cu130
export WORK="$PWD/runtime"
bash setup-native.sh
source "$WORK/runtime.env"
export MODELS="$PWD/models"
export OUT="$PWD/results"
```

The included source-only archive needs no access to a private repository. The installer selects the validated source snapshot, installs dependencies, compiles Thor kernels and creates separate official reference environments. Choose another `WORK` directory for a second installation; existing checkouts are not overwritten.

For China pip access, export `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` before setup/model preparation. The CUDA torch wheels still use the CUDA wheel index above.

If the selected mirror cannot install the native dependencies (for example,
`ninja`), the native installer retries official PyPI automatically. Ensure that
fallback endpoint is reachable as well.

## 3. Validate a model

Follow its guide below. Source versions and download checksums are managed by scripts and manifests; users do not need to copy commit hashes or machine-specific paths.

- [π0.5 / OpenPI](pi05.md)
- [GR00T N1.7](groot-n17.md)

Exit code 0 means the configured numerical checks passed. Keep JSON, NPZ and logs. These fixtures test numerical consistency; they do not measure robot task success. For deployment quality, evaluate identical tasks, seeds and initial states with both policies and report success counts.

First registry/account setup and publication: [release guide](release.md).
