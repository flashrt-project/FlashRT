from __future__ import annotations
import os

import argparse
import io
import json
import pathlib

import ml_dtypes
import numpy as np
import pandas as pd
import torch

from flash_rt import flash_rt_rocm_kernels as rocm
from flash_rt.core.calibration import stratified_sample_indices
from flash_rt.frontends.torch.pi05_rocm import _load_openpi_model
from flash_rt.frontends.torch.pi05_rocm_weights import (
    build_rocm_vision_weights_from_openpi_model,
)
from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm


CHECKPOINT = pathlib.Path(
    os.environ.get("PI05_CHECKPOINT", "lerobot/pi05_libero_finetuned")
)


def _decode_image_cell(cell, *, image_size: int = 224) -> np.ndarray:
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw is None and cell.get("path"):
            raw = pathlib.Path(cell["path"]).read_bytes()
        if raw is None:
            raise ValueError(f"image dict has no bytes/path keys: {list(cell)}")
    elif isinstance(cell, (bytes, bytearray)):
        raw = bytes(cell)
    elif isinstance(cell, np.ndarray):
        arr = np.asarray(cell)
        if arr.shape[-1] == 3:
            return arr.astype(np.uint8)
        raise TypeError(f"unsupported ndarray image shape {arr.shape}")
    else:
        raise TypeError(f"unsupported image cell type {type(cell).__name__}")

    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


class FlexibleLiberoDataset:
    """Read LIBERO LeRobot parquet datasets in old and new layouts."""

    def __init__(
        self,
        root: pathlib.Path,
        *,
        image_key: str | None = None,
        wrist_image_key: str | None = None,
        state_key: str = "observation.state",
        image_size: int = 224,
    ) -> None:
        self.root = pathlib.Path(root)
        self.image_size = int(image_size)
        self.info = self._load_info()
        features = self.info.get("features", {}) if self.info else {}
        self.video_path_template = self.info.get("video_path")
        self.fps = float(self.info.get("fps", 10))
        self.image_key = image_key or self._pick_key(
            features,
            ["observation.images.image", "image"],
        )
        self.wrist_image_key = wrist_image_key or self._pick_key(
            features,
            [
                "observation.images.image2",
                "observation.images.wrist_image",
                "wrist_image",
            ],
            required=False,
        )
        self.state_key = state_key
        self.parquets = self._find_parquets()
        if not self.parquets:
            raise FileNotFoundError(f"no parquet files found under {self.root}")
        self.episode_metadata = self._load_episode_metadata()
        self.metadata = self._build_metadata()
        self.total_frames = int(len(self.metadata))
        self.total_episodes = int(self.metadata["episode_index"].nunique())

    def _load_info(self) -> dict:
        p = self.root / "meta" / "info.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _pick_key(features: dict, candidates: list[str], *, required: bool = True) -> str | None:
        if features:
            for key in candidates:
                if key in features:
                    return key
        else:
            return candidates[0]
        if required:
            raise ValueError(
                f"none of {candidates} found in dataset features; "
                f"available={sorted(features)[:20]}"
            )
        return None

    def _find_parquets(self) -> list[pathlib.Path]:
        data = self.root / "data"
        if data.exists():
            return sorted(data.rglob("*.parquet"))
        return sorted(self.root.rglob("*.parquet"))

    def _load_episode_metadata(self) -> pd.DataFrame | None:
        import pyarrow.parquet as pq

        root = self.root / "meta" / "episodes"
        if not root.exists():
            return None
        frames = []
        for p in sorted(root.rglob("*.parquet")):
            try:
                frames.append(pq.read_table(p).to_pandas())
            except Exception as exc:
                print(f"skip episode metadata {p}: {exc}")
        if not frames:
            return None
        meta = pd.concat(frames, ignore_index=True)
        meta["episode_index"] = meta["episode_index"].astype(np.int64)
        return meta.set_index("episode_index", drop=False)

    def _build_metadata(self) -> pd.DataFrame:
        import pyarrow.parquet as pq

        frames = []
        cols = ["task_index", "episode_index", "frame_index", "index"]
        for p in self.parquets:
            try:
                df = pq.read_table(p, columns=cols).to_pandas()
            except Exception as exc:
                print(f"skip metadata {p}: {exc}")
                continue
            if df.empty:
                continue
            df["_file_path"] = str(p)
            frames.append(df)
        if not frames:
            raise RuntimeError(f"no usable LIBERO metadata rows under {self.root}")
        meta = pd.concat(frames, ignore_index=True)
        for col in cols:
            meta[col] = meta[col].astype(np.int64)
        return meta

    @property
    def tasks(self) -> dict[int, str]:
        out = {}
        p = self.root / "meta" / "tasks.jsonl"
        if not p.exists():
            return out
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            out[int(item["task_index"])] = str(item.get("task", ""))
        return out

    def load_frame(self, global_index: int) -> dict:
        import pyarrow.parquet as pq

        hits = self.metadata[self.metadata["index"] == int(global_index)]
        if hits.empty:
            raise KeyError(f"global_index={global_index} not found")
        file_path = pathlib.Path(str(hits.iloc[0]["_file_path"]))
        available = set(pq.ParquetFile(file_path).schema_arrow.names)
        wanted = [self.image_key, "index", "episode_index", "timestamp"]
        if self.wrist_image_key is not None:
            wanted.append(self.wrist_image_key)
        if self.state_key is not None:
            wanted.append(self.state_key)
        cols = [col for col in wanted if col in available]
        if "index" not in cols:
            raise ValueError(f"{file_path} does not contain required 'index' column")
        df = pq.read_table(file_path, columns=cols).to_pandas()
        row = df[df["index"] == int(global_index)]
        if row.empty:
            raise KeyError(f"global_index={global_index} not found in {file_path}")
        row = row.iloc[0]
        obs = {"image": self._load_image(row, self.image_key)}
        if self.wrist_image_key is not None:
            obs["wrist_image"] = self._load_image(row, self.wrist_image_key)
        if self.state_key in row:
            obs["state"] = np.asarray(row[self.state_key], dtype=np.float32)
        return obs

    def _load_image(
        self,
        row: pd.Series,
        key: str,
    ) -> np.ndarray:
        if key in row.index:
            return _decode_image_cell(row[key], image_size=self.image_size)
        video_path, timestamp = self._video_path_for(key, row)
        return self._decode_video_frame(video_path, timestamp)

    def _video_path_for(
        self, video_key: str, row: pd.Series
    ) -> tuple[pathlib.Path, float]:
        if self.episode_metadata is None:
            raise RuntimeError("video dataset requires meta/episodes parquet metadata")
        episode_index = int(row["episode_index"])
        ep = self.episode_metadata.loc[episode_index]
        chunk_index = int(ep[f"videos/{video_key}/chunk_index"])
        file_index = int(ep[f"videos/{video_key}/file_index"])
        from_timestamp = float(ep[f"videos/{video_key}/from_timestamp"])
        timestamp = from_timestamp + float(row.get("timestamp", 0.0))
        chunk_name = f"chunk-{chunk_index:03d}"
        file_name = f"file-{file_index:03d}.mp4"

        candidates: list[pathlib.Path] = []
        if self.video_path_template:
            try:
                candidates.append(
                    self.root
                    / self.video_path_template.format(
                        video_key=video_key,
                        chunk_index=chunk_index,
                        file_index=file_index,
                        episode_chunk=chunk_index,
                        episode_index=file_index,
                    )
                )
            except Exception:
                pass
        candidates.extend(
            [
                self.root / "videos" / video_key / chunk_name / file_name,
                self.root / "videos" / chunk_name / video_key / file_name,
                self.root / "videos" / video_key / file_name,
            ]
        )
        for p in candidates:
            if p.exists():
                return p, timestamp
        raise FileNotFoundError(
            f"could not find video for {video_key!r}; tried "
            f"{[str(p) for p in candidates]}"
        )

    def _decode_video_frame(self, video_path: pathlib.Path, timestamp: float) -> np.ndarray:
        return self._decode_video_frame_ffmpeg(video_path, timestamp)

    def _decode_video_frame_ffmpeg(
        self, video_path: pathlib.Path, timestamp: float
    ) -> np.ndarray:
        import math
        import subprocess

        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-ss",
            f"{max(0.0, float(timestamp)):.6f}",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(
                f"ffmpeg failed to read timestamp {timestamp:.6f} from {video_path}: "
                f"rc={proc.returncode} stderr={proc.stderr.decode(errors='ignore')[:500]}"
            )
        pixels = len(proc.stdout) // 3
        side = int(math.isqrt(pixels))
        if side * side != pixels:
            raise RuntimeError(
                f"cannot infer square frame shape from {len(proc.stdout)} raw bytes"
            )
        frame = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(side, side, 3)
        if frame.shape[:2] != (self.image_size, self.image_size):
            import cv2

            frame = cv2.resize(
                frame,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
        return np.asarray(frame, dtype=np.uint8)


def _obs_images(obs: dict, num_views: int = 3) -> np.ndarray:
    if "images" in obs:
        images = list(obs["images"])
    else:
        images = [obs["image"]]
        if "wrist_image" in obs:
            images.append(obs["wrist_image"])
        if "wrist_image_right" in obs:
            images.append(obs["wrist_image_right"])
    if len(images) == 1:
        images = [images[0], images[0], images[0]]
    elif len(images) == 2:
        images = [images[0], images[1], images[1]]
    else:
        images = images[:num_views]
    normed = [
        (np.asarray(img, dtype=np.float32) / 127.5 - 1.0).astype(ml_dtypes.bfloat16)
        for img in images[:num_views]
    ]
    return np.stack(normed, axis=0)


def upload_observation(
    pipe: Pi05PipelineRocm,
    obs: dict,
    *,
    noise_seed: int = 0,
    noise_std: float = 0.0,
) -> None:
    pipe.input_images_buf.upload(_obs_images(obs, pipe.num_views))

    rng = np.random.default_rng(noise_seed)
    if noise_std == 0.0:
        noise = np.zeros((pipe.chunk_size, 32), dtype=np.float32)
    else:
        noise = rng.normal(
            0.0, float(noise_std), size=(pipe.chunk_size, 32)
        ).astype(np.float32)
    pipe.input_noise_buf.upload(noise.astype(ml_dtypes.bfloat16))

    # Prompt/token embeddings are not wired into the owned ROCm frontend yet.
    # Keep the language slice deterministic and let vision overwrite its prefix.
    pipe.bufs["encoder_x"].upload(
        np.zeros((pipe.encoder_seq_len, 2048), dtype=ml_dtypes.bfloat16)
    )


def read_noise(pipe: Pi05PipelineRocm) -> torch.Tensor:
    out = pipe.input_noise_buf.download_new((pipe.chunk_size, 32), np.uint16)
    return torch.from_numpy(out).view(torch.bfloat16).float().cuda()


def summarize(name: str, ref: torch.Tensor, got: torch.Tensor) -> dict[str, float | bool]:
    ref_f = ref.flatten().float()
    got_f = got.flatten().float()
    diff = got_f - ref_f
    stats = {
        "cosine": torch.nn.functional.cosine_similarity(ref_f, got_f, dim=0).item(),
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rel_l2": (diff.norm() / ref_f.norm().clamp_min(1e-12)).item(),
        "ref_finite": torch.isfinite(ref_f).all().item(),
        "got_finite": torch.isfinite(got_f).all().item(),
        "ref_sum": ref_f.sum().item(),
        "got_sum": got_f.sum().item(),
    }
    print(name, stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, default=CHECKPOINT)
    parser.add_argument("--libero-root", type=pathlib.Path, required=True)
    parser.add_argument("--n-calib", type=int, default=16)
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--task-filter", type=int, default=None)
    parser.add_argument("--image-key", default=None)
    parser.add_argument("--wrist-image-key", default=None)
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--min-cos", type=float, default=0.99)
    args = parser.parse_args()

    print("torch", torch.__version__, "hip", torch.version.hip)
    ds = FlexibleLiberoDataset(
        args.libero_root,
        image_key=args.image_key,
        wrist_image_key=args.wrist_image_key,
    )
    picks = stratified_sample_indices(
        ds.metadata,
        n=args.n_calib,
        task_filter=args.task_filter,
        exclude=[] if args.target_index is None else [args.target_index],
    )
    if not picks:
        raise RuntimeError("no LIBERO calibration frames selected")
    target_index = int(args.target_index if args.target_index is not None else picks[0])
    print(
        "libero",
        {
            "root": str(args.libero_root),
            "total_frames": ds.total_frames,
            "total_episodes": ds.total_episodes,
            "image_key": ds.image_key,
            "wrist_image_key": ds.wrist_image_key,
            "calib_indices": picks,
            "target_index": target_index,
            "percentile": args.percentile,
            "noise_std": args.noise_std,
        },
    )

    calib_obs = [ds.load_frame(i) for i in picks]
    target_obs = ds.load_frame(target_index)

    model, _cfg, path = _load_openpi_model(args.checkpoint)
    print("loaded", path)
    weights = build_rocm_vision_weights_from_openpi_model(model, include_fp8=True)

    bf16_pipe = Pi05PipelineRocm.with_sdpa_attention(
        num_views=3,
        max_prompt_len=48,
        chunk_size=10,
        num_steps=10,
    )
    fp8_pipe = Pi05PipelineRocm.with_sdpa_attention(
        num_views=3,
        max_prompt_len=48,
        chunk_size=10,
        num_steps=10,
    )

    upload_observation(
        bf16_pipe, target_obs, noise_seed=target_index, noise_std=args.noise_std
    )
    bf16_pipe.bake_bf16_gemms(rocm)
    bf16_pipe.run_bf16_pipeline(rocm, weights)
    rocm.hip_sync()
    bf16_out = read_noise(bf16_pipe)

    fp8_pipe.bake_fp8_gemms(rocm)
    resetters = [
        (lambda obs=obs, seed=int(idx): lambda pipe: upload_observation(
            pipe, obs, noise_seed=seed, noise_std=args.noise_std
        ))()
        for idx, obs in zip(picks, calib_obs, strict=True)
    ]
    cal = fp8_pipe.calibrate_fp8_multi_sample(
        rocm, weights, resetters, percentile=args.percentile
    )
    torch.cuda.synchronize()
    print("FP8_MULTI_CAL", cal)

    upload_observation(
        fp8_pipe, target_obs, noise_seed=target_index, noise_std=args.noise_std
    )
    fp8_pipe.run_fp8_pipeline(rocm, weights)
    torch.cuda.synchronize()
    fp8_eager_out = read_noise(fp8_pipe)
    eager_stats = summarize("BF16_vs_FP8_EAGER", bf16_out, fp8_eager_out)

    upload_observation(
        fp8_pipe, target_obs, noise_seed=target_index, noise_std=args.noise_std
    )
    fp8_pipe.capture_fp8_graph(rocm, weights)
    torch.cuda.synchronize()
    upload_observation(
        fp8_pipe, target_obs, noise_seed=target_index, noise_std=args.noise_std
    )
    fp8_pipe.replay_fp8_graph()
    torch.cuda.synchronize()
    fp8_graph_out = read_noise(fp8_pipe)
    graph_stats = summarize("BF16_vs_FP8_GRAPH", bf16_out, fp8_graph_out)
    summarize("FP8_EAGER_vs_FP8_GRAPH", fp8_eager_out, fp8_graph_out)

    if graph_stats["cosine"] < args.min_cos:
        raise SystemExit(
            f"cosine {graph_stats['cosine']:.6f} < required {args.min_cos:.6f}"
        )
    if not graph_stats["got_finite"]:
        raise SystemExit("FP8 graph output contains non-finite values")
    if not eager_stats["got_finite"]:
        raise SystemExit("FP8 eager output contains non-finite values")


if __name__ == "__main__":
    main()
