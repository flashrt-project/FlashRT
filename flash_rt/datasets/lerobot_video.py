"""LeRobot v2.1 dataset loader for video-backed recordings.

``flash_rt.datasets.libero.LiberoDataset`` reads images embedded as PNG bytes in
the parquet cells. Many LeRobot recordings — including the ones shipped in
Isaac-GR00T's ``demo_data`` and most SO100/SO101 captures — store cameras as
``dtype=video`` mp4 sidecar files instead, so the parquet holds only
state/action. This module covers that layout with the same contract, so
calibration and precision harnesses can swap between the two.

Layout expected::

    <root>/
      meta/
        info.json       # features, fps, total_frames, chunks_size, video_path
        episodes.jsonl  # {episode_index, tasks, length}
        tasks.jsonl     # {task_index, task}
      data/chunk-{NNN}/episode_{NNNNNN}.parquet
      videos/chunk-{NNN}/<video_key>/episode_{NNNNNN}.mp4

Provides:

* :class:`LeRobotVideoDataset` — ``load_frame(global_index)`` -> obs dict
* :func:`load_calibration_obs` — stratified sampling + loading in one call,
  returning a list ready for ``model.calibrate(obs_list, percentile=...)``

Kept torch-free / jax-free so the loader works in any frontend env.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_VIDEO_KEYS = (
    "observation.images.front",
    "observation.images.wrist",
)
_DEFAULT_STATE_KEY = "observation.state"
_DEFAULT_ACTION_KEY = "action"


def _frame_index(frame, stream, fps: int, fallback: int) -> int:
    """Integer frame index for a decoded frame.

    Derives from pts when the stream provides it (constant-frame-rate LeRobot
    recordings do); otherwise falls back to the caller's sequential counter so
    a forward-only read still works after a seek-free pass.
    """
    if frame.pts is None or stream.time_base is None:
        return fallback
    rate = float(stream.average_rate) if stream.average_rate else float(fps)
    return int(round(float(frame.pts * stream.time_base) * rate))


class LeRobotVideoDataset:
    """Random-access frame reader for a video-backed LeRobot v2.x dataset.

    Args:
        root: dataset root (contains ``meta/``, ``data/``, ``videos/``).
        video_keys: parquet/feature keys of the camera streams, in the order
            they should appear in the returned obs dict.
        state_key: state column, or None to skip.
        action_key: action column, or None to skip. Actions are not needed for
            inference but are useful as a real-data reference for precision
            comparison, so they are returned when present.
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        video_keys: Sequence[str] = _DEFAULT_VIDEO_KEYS,
        state_key: Optional[str] = _DEFAULT_STATE_KEY,
        action_key: Optional[str] = _DEFAULT_ACTION_KEY,
    ) -> None:
        self.root = pathlib.Path(root)
        self.video_keys = tuple(video_keys)

        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"LeRobot meta/info.json not found under {self.root!s}")
        with open(info_path, encoding="utf-8") as f:
            self.info: Dict[str, Any] = json.load(f)

        features = self.info.get("features", {})
        self.state_key = self._resolve_column(state_key, features, "state")
        self.action_key = self._resolve_column(action_key, features, "action")
        for k in self.video_keys:
            if k not in features:
                raise KeyError(
                    f"video_key={k!r} not in info.json 'features'; "
                    f"available: {sorted(features)}")
            if features[k].get("dtype") != "video":
                logger.warning(
                    "video_key=%r has dtype=%r, not 'video'; frames may be "
                    "embedded in the parquet instead (use "
                    "flash_rt.datasets.libero.LiberoDataset for that layout)",
                    k, features[k].get("dtype"))

        self.fps: int = int(self.info.get("fps", 30))
        self.total_frames: int = int(self.info.get("total_frames", 0))
        self.total_episodes: int = int(self.info.get("total_episodes", 0))
        self._chunks_size: int = int(self.info.get("chunks_size", 1000))
        self._data_path_template: str = self.info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        self._video_path_template: str = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4")

        self._metadata = None
        self._episode_lengths: Dict[int, int] = {}
        self._episode_task: Dict[int, int] = {}
        self._episode_first_index: Dict[int, int] = {}
        # one open decoder per video key, reused across ascending reads
        self._decoders: Dict[str, Tuple[Any, Any, int]] = {}

    # ------------------------------------------------------------------
    # columns
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_column(requested: Optional[str], features: Dict[str, Any],
                        what: str) -> Optional[str]:
        """Actual parquet column for the state/action stream.

        Recordings differ in whether the vector column carries the bare key
        (``observation.state``, as in Isaac-GR00T's ``demo_data``) or a modality
        suffix (``observation.state.joint``, which ``meta/modality.json`` then
        splits into ``single_arm`` + ``gripper``). Accept either so callers do
        not have to know which convention a recording used, but refuse an
        ambiguous probe: guessing wrong reads a column that does not exist, or
        one that does and means something else.
        """
        if requested is None or requested in features:
            return requested
        prefixed = sorted(k for k in features if k.startswith(requested + "."))
        if len(prefixed) == 1:
            logger.info("%s column %r not present; using the suffixed %r",
                        what, requested, prefixed[0])
            return prefixed[0]
        raise KeyError(
            f"{what}_key={requested!r} not in info.json 'features', and the "
            f"suffixed probe matched {prefixed or 'nothing'}; pass the exact "
            f"column name. Available: {sorted(features)}")

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------

    def _episode_path(self, ep: int) -> pathlib.Path:
        return self.root / self._data_path_template.format(
            episode_chunk=ep // self._chunks_size, episode_index=ep)

    def _video_path(self, ep: int, video_key: str) -> pathlib.Path:
        return self.root / self._video_path_template.format(
            episode_chunk=ep // self._chunks_size,
            video_key=video_key,
            episode_index=ep)

    # ------------------------------------------------------------------
    # metadata / tasks
    # ------------------------------------------------------------------

    @property
    def metadata(self):
        """``(task_index, episode_index, frame_index, index)`` table.

        Shape matches what ``flash_rt.core.calibration.stratified_sample_indices``
        expects. Built lazily; reads only index columns, decodes no video.
        """
        if self._metadata is None:
            self._metadata = self._build_metadata()
        return self._metadata

    def _build_metadata(self):
        import pandas as pd
        import pyarrow.parquet as pq

        frames = []
        running_first = 0
        for ep in range(self.total_episodes):
            p = self._episode_path(ep)
            if not p.exists():
                logger.warning("missing episode parquet %s", p)
                continue
            try:
                t = pq.read_table(
                    p, columns=["task_index", "episode_index",
                                "frame_index", "index"])
            except Exception as e:
                logger.warning("skip %s (%s)", p, e)
                continue
            df = t.to_pandas()
            if len(df) == 0:
                continue
            self._episode_lengths[ep] = len(df)
            self._episode_task[ep] = int(df["task_index"].iloc[0])
            self._episode_first_index[ep] = int(df["index"].iloc[0])
            frames.append(df)
        if not frames:
            raise RuntimeError(f"no readable episodes under {self.root}")
        meta = pd.concat(frames, ignore_index=True)
        logger.debug("metadata built: %d frames, first_index=%d",
                     len(meta), running_first)
        return meta

    @property
    def tasks(self) -> Dict[int, str]:
        """``{task_index: task_string}`` from ``meta/tasks.jsonl``."""
        out: Dict[int, str] = {}
        p = self.root / "meta" / "tasks.jsonl"
        if not p.exists():
            return out
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[int(r["task_index"])] = r.get("task", "")
        return out

    def task_for_frame(self, global_index: int) -> str:
        meta = self.metadata
        hit = meta.index[meta["index"] == int(global_index)]
        if len(hit) == 0:
            raise KeyError(f"global_index={global_index} not in dataset")
        return self.tasks.get(int(meta.iloc[int(hit[0])]["task_index"]), "")

    # ------------------------------------------------------------------
    # frame access
    # ------------------------------------------------------------------

    def _episode_for(self, global_index: int) -> Tuple[int, int]:
        """Map a global index to ``(episode_index, frame_index_in_episode)``."""
        meta = self.metadata
        hit = meta.index[meta["index"] == int(global_index)]
        if len(hit) == 0:
            raise KeyError(
                f"global_index={global_index} not in this dataset "
                f"(total_frames={self.total_frames})")
        row = meta.iloc[int(hit[0])]
        return int(row["episode_index"]), int(row["frame_index"])

    def _decode(self, video_key: str, ep: int, frame_index: int) -> np.ndarray:
        """Return frame ``frame_index`` of ``video_key`` as uint8 RGB [H,W,3].

        Keeps one decoder open per video key. Ascending reads within an episode
        continue decoding forward (cheap); any other access pattern seeks to the
        preceding keyframe and decodes forward to the target.
        """
        import av

        path = self._video_path(ep, video_key)
        if not path.exists():
            raise FileNotFoundError(
                f"video not found: {path}. If this dataset was cloned from git, "
                f"the mp4 may still be an LFS pointer -- run "
                f"`git lfs pull --include=\"<dataset>/**\"`.")

        cached = self._decoders.get(video_key)
        if cached is not None:
            container, stream, state = cached
            if state["ep"] != ep:
                container.close()
                cached = None
        if cached is None:
            container = av.open(str(path))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            state = {"ep": ep, "next": 0, "it": None}
            self._decoders[video_key] = (container, stream, state)
        else:
            container, stream, state = cached

        if frame_index < state["next"]:
            # rewind: seek to a keyframe at or before the target, then walk on
            rate = float(stream.average_rate) if stream.average_rate else float(self.fps)
            target_pts = int((frame_index / rate) / float(stream.time_base))
            # Drop the suspended iterator *before* seeking: finalizing a PyAV
            # decode generator flushes its codec, so resuming the old one
            # after the stream moved under it is not defined.
            state["it"] = None
            container.seek(max(0, target_pts), stream=stream, any_frame=False)
            state["next"] = 0

        # One iterator per decoder, held across calls rather than rebuilt per
        # frame. Rebuilding it -- ``for frame in container.decode(stream)`` and
        # returning from inside the loop -- abandons a live generator on every
        # read; PyAV finalizes it, which flushes the codec, and an ascending
        # walk then dies partway through the file with
        # ``EOFError: 'avcodec_send_packet()'``. Measured on a 593-frame LeRobot
        # episode: abandoned-per-call reached frame 581, held-across-calls
        # reached 593 and returned identical pixels. Scattered reads reopen or
        # seek often enough to mask it, which is why only long sequential runs
        # (a full-episode horizon) hit it.
        if state["it"] is None:
            state["it"] = iter(container.decode(stream))
        for frame in state["it"]:
            idx = _frame_index(frame, stream, self.fps, state["next"])
            state["next"] = idx + 1
            if idx >= frame_index:
                if idx != frame_index:
                    raise IndexError(
                        f"frame {frame_index} skipped in {path} "
                        f"(landed on {idx}); the stream's pts may be unreliable")
                return frame.to_ndarray(format="rgb24")
        raise IndexError(
            f"frame {frame_index} past end of {path} "
            f"(decoded up to {state['next']})")

    def load_frame(self, global_index: int) -> Dict[str, Any]:
        """Decode one frame to an obs dict.

        Returned shape::

            {"images": {video_key: uint8 [H, W, 3], ...},
             "state":  float32 [state_dim],     # omitted if state_key is None
             "action": float32 [action_dim],    # omitted if absent
             "task":   str,
             "episode_index": int, "frame_index": int, "index": int}
        """
        import pyarrow.parquet as pq

        ep, fi = self._episode_for(global_index)
        p = self._episode_path(ep)
        cols = ["index"]
        if self.state_key:
            cols.append(self.state_key)
        if self.action_key:
            cols.append(self.action_key)
        df = pq.read_table(p, columns=cols).to_pandas()
        row = df[df["index"] == int(global_index)]
        if row.empty:
            raise KeyError(f"global_index={global_index} not in {p}")
        r = row.iloc[0]

        obs: Dict[str, Any] = {
            "images": {k: self._decode(k, ep, fi) for k in self.video_keys},
            "task": self.task_for_frame(global_index),
            "episode_index": ep,
            "frame_index": fi,
            "index": int(global_index),
        }
        if self.state_key:
            obs["state"] = np.asarray(r[self.state_key], dtype=np.float32)
        if self.action_key and self.action_key in df.columns:
            obs["action"] = np.asarray(r[self.action_key], dtype=np.float32)
        return obs

    def close(self) -> None:
        for container, _stream, _state in self._decoders.values():
            container.close()
        self._decoders.clear()

    def __enter__(self) -> "LeRobotVideoDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_calibration_obs(
    root: Union[str, pathlib.Path],
    *,
    n: int = 8,
    video_keys: Sequence[str] = _DEFAULT_VIDEO_KEYS,
    task_filter: Optional[int] = None,
    exclude: Optional[Iterable[int]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Stratified-sample ``n`` real observations for calibration.

    Mirrors :func:`flash_rt.datasets.libero.load_calibration_obs`. Calibration
    data must come from the host's real inference distribution — synthetic or
    random tensors mismeasure activation outliers and therefore pick the wrong
    quantization recipe.
    """
    from flash_rt.core.calibration import stratified_sample_indices

    with LeRobotVideoDataset(root, video_keys=video_keys) as ds:
        picks = stratified_sample_indices(
            ds.metadata, n=n, task_filter=task_filter, exclude=exclude)
        if verbose:
            logger.info(
                "LeRobot stratified sample: %d/%d frames from %s "
                "(tasks=%d, episodes=%d)",
                len(picks), ds.total_frames, ds.root,
                len(ds.tasks), ds.total_episodes)
            logger.info("picked global indices: %s", list(picks))
        return [ds.load_frame(i) for i in picks]
