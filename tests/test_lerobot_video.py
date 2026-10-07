"""CPU contract tests for ``flash_rt.datasets.lerobot_video``.

The reader is what every GR00T reference fixture and every long-horizon
fidelity run is built from, so a frame it gets wrong is a wrong reference, not
a wrong output -- it fails silently in the direction that matters most.

Two halves, tested separately because they fail separately:

* ``_decode`` against a bare 600-frame mp4 (no dataset on disk). The failure
  mode there needs a few hundred frames of ascending reads to appear -- it lost
  the *tail* of a 593-frame episode, at frame 581 -- so a 600-frame 64x64 encode
  (~1 s) is what makes it a regression test rather than a smoke test.
* the dataset layout against a synthetic 2-episode recording in ``tmp_path``:
  the global-index -> (episode, frame) mapping, the parquet row selection, the
  column-name probe and the task lookup.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from flash_rt.datasets.lerobot_video import LeRobotVideoDataset, _frame_index

FPS = 30
#: Enough frames that the tail-loss bug is reachable. It fired at 581/593 on a
#: real episode and 587/600 on this synthetic one -- ~98% through in both cases,
#: because abandoning a decode generator flushes the codec and the frames still
#: queued behind the flush become unreachable.
N_FRAMES = 600
SIDE = 64


def _encode(path: Path) -> None:
    """Write a video whose frame *i* carries a white bar on row ``i % SIDE``.

    The bar position is the assertion: a reader that skips, repeats or
    reorders frames returns the wrong row, and yuv420p round-tripping leaves a
    hard-edged bar recoverable exactly (it is the pixel *values* that are lossy,
    not the row a 255-vs-0 edge lands on).
    """
    import av

    with av.open(str(path), "w") as c:
        try:
            s = c.add_stream("libx264", rate=FPS)
        except Exception as exc:                    # pragma: no cover
            pytest.skip(f"no libx264 encoder available: {exc}")
        s.width = s.height = SIDE
        s.pix_fmt = "yuv420p"
        for i in range(N_FRAMES):
            img = np.zeros((SIDE, SIDE, 3), np.uint8)
            img[i % SIDE, :, :] = 255
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for pkt in s.encode(frame):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("lerobot_video") / "episode_000000.mp4"
    _encode(path)
    return path


@pytest.fixture
def reader(video, monkeypatch):
    """A dataset shell exposing ``_decode`` without a dataset on disk.

    ``_decode`` only touches ``_video_path``, ``fps`` and ``_decoders``, so
    stubbing those three keeps the test off the parquet/meta layout while still
    driving the real decode path -- including the cached-decoder bookkeeping
    under test.
    """
    ds = object.__new__(LeRobotVideoDataset)
    ds.fps = FPS
    ds._decoders = {}
    monkeypatch.setattr(ds, "_video_path", lambda ep, key: video, raising=False)
    yield ds
    ds.close()


def bar_row(frame: np.ndarray) -> int:
    return int(np.argmax(frame.mean(axis=(1, 2))))


def test_every_frame_of_a_full_ascending_pass_is_reachable(reader):
    """The regression: the tail of a long ascending read used to disappear.

    ``_decode`` rebuilt ``container.decode(stream)`` per call and returned from
    inside the loop, abandoning a live generator each time. PyAV finalizes an
    abandoned decode generator by flushing its codec, so the frames still
    queued behind the flush became unreachable and the walk died near the end
    of the file with ``EOFError: 'avcodec_send_packet()'`` -- measured at frame
    581 of a 593-frame episode, which silently truncated a full-episode
    long-horizon fidelity run. Holding one iterator across calls fixes it.
    """
    for i in range(N_FRAMES):
        row = bar_row(reader._decode("cam", 0, i))
        assert row == i % SIDE, f"frame {i} decoded as bar row {row}"


def test_one_decoder_is_reused_across_an_ascending_pass(reader):
    """Ascending reads must not reopen the container.

    Reopening would also fix the tail loss, by never leaving a generator
    suspended -- and would be O(N^2) in decode work, which is the reason this
    reader exists at all (the official ffmpeg backend it replaced spawned a
    subprocess per frame). Pinning reuse is what keeps the fix from being
    "just reopen every time".
    """
    reader._decode("cam", 0, 0)
    assert "cam" in reader._decoders
    container = reader._decoders["cam"][0]
    reader._decode("cam", 0, N_FRAMES - 1)
    assert reader._decoders["cam"][0] is container, "the container was reopened"


def test_a_rewind_after_the_ascending_pass_still_returns_the_right_frame(reader):
    """The seek path has to drop the suspended iterator, not resume it.

    Resuming an iterator whose stream was just moved under it is undefined; the
    rewind therefore invalidates it and builds a fresh one. Without that, the
    frames after a seek come from wherever the old iterator was parked.
    """
    reader._decode("cam", 0, N_FRAMES - 1)
    for i in (500, 10, 499, 0, 300, 299, N_FRAMES - 1, 1):
        assert bar_row(reader._decode("cam", 0, i)) == i % SIDE


def test_reading_past_the_end_raises_indexerror(reader):
    """A short file must be reported as one, not served as a stale frame."""
    with pytest.raises(IndexError, match="past end"):
        reader._decode("cam", 0, N_FRAMES + 50)


def test_a_second_video_key_gets_its_own_decoder(video, monkeypatch):
    """Two cameras must not share one decoder's position."""
    ds = object.__new__(LeRobotVideoDataset)
    ds.fps = FPS
    ds._decoders = {}
    monkeypatch.setattr(ds, "_video_path", lambda ep, key: video, raising=False)
    try:
        assert bar_row(ds._decode("front", 0, 7)) == 7 % SIDE
        assert bar_row(ds._decode("wrist", 0, 3)) == 3 % SIDE
        # interleaving the two must not disturb either one's position
        assert bar_row(ds._decode("front", 0, 8)) == 8 % SIDE
        assert bar_row(ds._decode("wrist", 0, 4)) == 4 % SIDE
        assert set(ds._decoders) == {"front", "wrist"}
    finally:
        ds.close()


# ══════════════════════════════════════════════════════════════════════════
# the dataset-layout half: global index -> (episode, frame) -> parquet row
# ══════════════════════════════════════════════════════════════════════════
#
# Severity, stated honestly, because it is easy to overstate: a wrong mapping
# here hands the *same* wrong frame to both the HF reference and the FlashRT
# frontend, so every equivalence claim in the N1.7 gates would survive it. What
# breaks is provenance -- the "frame 100", "593 consecutive frames" and
# "distance from prompt" axes the long-horizon runs report, and the frame
# adjacency the continuous suite's C6 calibration is built on (its distant arm
# is pinned at cos 0.900 against frame 100). These pins protect the meaning of
# the reported numbers, not their agreement.

DS_SIDE = 32
#: Unequal lengths on purpose. A mapping derived from ``global // length``, or
#: from a running counter reset per episode, coincides with the truth when the
#: episodes are the same size and only separates when they are not.
DS_LENGTHS = (5, 4)
DS_TOTAL = sum(DS_LENGTHS)
DS_TASKS = ("stack the cube on the block", "unstack the cube from the block")
VIDEO_KEYS = ("observation.images.front", "observation.images.wrist")
#: The wrist bar is offset from the front bar, so a key mix-up is visible.
WRIST_OFFSET = 7
STATE_DIM = 3


def _state(g: int):
    """A per-index-distinctive state vector, exact in float32.

    ``g``, ``2.5 g`` and ``-1.25 g`` are all dyadic, so the parquet round trip
    and ``np.asarray(..., dtype=np.float32)`` are bit-exact and the row
    assertions below can be equalities rather than tolerances.
    """
    return [float(g), float(g) * 2.5, float(g) * -1.25]


def _action(g: int):
    return [float(g) + 0.5, float(g) * 2.5 + 0.25, float(g) * -1.25 - 0.5]


def _ep_fi(g: int):
    """Expected ``(episode_index, frame_index)`` for a global index."""
    return (0, g) if g < DS_LENGTHS[0] else (1, g - DS_LENGTHS[0])


def _encode_marked(path: Path, n_frames: int, ep: int, row_offset: int) -> None:
    """Write an ``n_frames`` video marked with (frame, episode) provenance.

    Frame *i* carries a white horizontal bar on row ``(i + row_offset) %
    DS_SIDE`` and a white vertical bar on column ``ep * 5``, so the decoded
    pixels report which frame of *which episode* they came from. The episode
    marker is what makes ``load_frame``'s image assertion discriminative:
    without it, ``ep0/frame0`` and ``ep1/frame0`` decode to identical pixels and
    a reader that never left episode 0 would pass.
    """
    import av

    with av.open(str(path), "w") as c:
        try:
            s = c.add_stream("libx264", rate=FPS)
        except Exception as exc:                    # pragma: no cover
            pytest.skip(f"no libx264 encoder available: {exc}")
        s.width = s.height = DS_SIDE
        s.pix_fmt = "yuv420p"
        for i in range(n_frames):
            img = np.zeros((DS_SIDE, DS_SIDE, 3), np.uint8)
            img[:, ep * 5, :] = 255
            img[(i + row_offset) % DS_SIDE, :, :] = 255
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for pkt in s.encode(frame):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def _build_dataset(root: Path, *, index_shift: int = 0) -> Path:
    """Lay out a 2-episode LeRobot v2.1 recording under ``root``.

    Mirrors the real thing: ``meta/info.json`` + ``tasks.jsonl`` +
    ``episodes.jsonl``, ``data/chunk-{NNN}/episode_{NNNNNN}.parquet`` holding
    only state/action plus the four index columns (cameras are ``dtype=video``),
    and one mp4 per (episode, camera). The suffixed column convention
    (``observation.state.joint``) is the one the SO101 sim capture actually
    uses, so ``__init__``'s ``_resolve_column`` probe is exercised rather than
    bypassed; Isaac-GR00T's ``demo_data`` uses the bare convention instead and
    that arm is pinned directly below.

    ``chunks_size`` is 1, not the real recordings' 1000, so episode 1 lands in
    ``chunk-001`` and the ``ep // chunks_size`` arithmetic in ``_episode_path``
    is load-bearing. At 1000 every episode sits in chunk-000 and that division
    is never observed to do anything.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    first = 0
    for ep, n in enumerate(DS_LENGTHS):
        chunk = ep  # chunks_size == 1
        state_col = pa.list_(pa.float32(), STATE_DIM)
        table = pa.table({
            "observation.state.joint": pa.array(
                [_state(first + i) for i in range(n)], state_col),
            "action.joint": pa.array(
                [_action(first + i) for i in range(n)], state_col),
            "timestamp": pa.array([i / FPS for i in range(n)], pa.float32()),
            "frame_index": pa.array(range(n), pa.int64()),
            "episode_index": pa.array([ep] * n, pa.int64()),
            # the only column index_shift touches; see the negative control
            "index": pa.array([first + i + index_shift for i in range(n)],
                              pa.int64()),
            "task_index": pa.array([ep] * n, pa.int64()),
        })
        ddir = root / "data" / f"chunk-{chunk:03d}"
        ddir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, ddir / f"episode_{ep:06d}.parquet")
        for key, off in zip(VIDEO_KEYS, (0, WRIST_OFFSET)):
            vdir = root / "videos" / f"chunk-{chunk:03d}" / key
            vdir.mkdir(parents=True, exist_ok=True)
            _encode_marked(vdir / f"episode_{ep:06d}.mp4", n, ep, off)
        first += n

    features = {
        "observation.state.joint": {
            "dtype": "float32", "shape": [STATE_DIM], "names": ["a", "b", "c"]},
        "action.joint": {
            "dtype": "float32", "shape": [STATE_DIM], "names": ["a", "b", "c"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key in VIDEO_KEYS:
        features[key] = {
            "dtype": "video", "shape": [DS_SIDE, DS_SIDE, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.height": DS_SIDE, "video.width": DS_SIDE,
                     "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.fps": FPS, "video.channels": 3,
                     "has_audio": False, "video.is_depth_map": False},
        }
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1",
        "robot_type": "so101_follower",
        "total_episodes": len(DS_LENGTHS),
        "total_frames": DS_TOTAL,
        "total_tasks": len(DS_TASKS),
        "total_videos": len(DS_LENGTHS) * len(VIDEO_KEYS),
        "total_chunks": len(DS_LENGTHS),
        "chunks_size": 1,
        "fps": FPS,
        "splits": {"train": f"0:{len(DS_LENGTHS)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/"
                     "episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/"
                      "episode_{episode_index:06d}.mp4",
        "features": features,
    }), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": t, "task": s}) + "\n"
                for t, s in enumerate(DS_TASKS)), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": ep, "tasks": [DS_TASKS[ep]],
                            "length": n}) + "\n"
                for ep, n in enumerate(DS_LENGTHS)), encoding="utf-8")
    return root


def bar_col(frame: np.ndarray) -> int:
    """Column of the vertical (episode) bar."""
    return int(np.argmax(frame.mean(axis=(0, 2))))


@pytest.fixture(scope="module")
def ds_root(tmp_path_factory) -> Path:
    return _build_dataset(tmp_path_factory.mktemp("lerobot_ds"))


@pytest.fixture(scope="module")
def ds_shifted_root(tmp_path_factory) -> Path:
    return _build_dataset(tmp_path_factory.mktemp("lerobot_ds_shifted"),
                          index_shift=1)


@pytest.fixture
def ds(ds_root):
    with LeRobotVideoDataset(ds_root, video_keys=VIDEO_KEYS) as d:
        yield d


@pytest.fixture
def ds_shifted(ds_shifted_root):
    with LeRobotVideoDataset(ds_shifted_root, video_keys=VIDEO_KEYS) as d:
        yield d


# ── _resolve_column: the two naming conventions and the refusal ───────────

def test_resolve_column_accepts_both_naming_conventions():
    """``observation.state`` and ``observation.state.joint`` are both real.

    Isaac-GR00T's ``demo_data`` ships the bare key; the SO101 sim capture ships
    the modality-suffixed one that ``meta/modality.json`` then splits into
    ``single_arm`` + ``gripper``. A caller should not have to know which.
    """
    resolve = LeRobotVideoDataset._resolve_column
    bare = {"observation.state": {}, "action": {}}
    suffixed = {"observation.state.joint": {}, "action.joint": {}}

    assert resolve("observation.state", bare, "state") == "observation.state"
    assert resolve("action", bare, "action") == "action"
    assert resolve("observation.state", suffixed,
                   "state") == "observation.state.joint"
    assert resolve("action", suffixed, "action") == "action.joint"
    # None means "skip this stream", not "probe for it"
    assert resolve(None, suffixed, "state") is None
    assert resolve(None, {}, "state") is None


def test_resolve_column_refuses_an_ambiguous_or_absent_probe():
    """Refuse rather than guess: guessing reads a column that means less.

    ``.joint`` + ``.eef`` is what a bimanual or Cartesian-augmented recording
    would have, and silently taking the first of the two reads half a state.
    """
    resolve = LeRobotVideoDataset._resolve_column
    ambiguous = {"observation.state.joint": {}, "observation.state.eef": {}}
    with pytest.raises(KeyError) as exc:
        resolve("observation.state", ambiguous, "state")
    msg = str(exc.value)
    assert "observation.state.eef" in msg and "observation.state.joint" in msg

    with pytest.raises(KeyError, match="matched nothing"):
        resolve("observation.state", {"action": {}}, "state")


def test_the_reader_resolves_its_own_suffixed_columns(ds):
    """``__init__`` must wire the probe, not merely define it."""
    assert ds.state_key == "observation.state.joint"
    assert ds.action_key == "action.joint"


def test_a_bare_convention_recording_resolves_to_the_bare_columns(tmp_path):
    """The other real convention, end to end through ``__init__``."""
    root = _build_dataset(tmp_path / "bare")
    info = json.loads((root / "meta" / "info.json").read_text())
    for old, new in (("observation.state.joint", "observation.state"),
                     ("action.joint", "action")):
        info["features"][new] = info["features"].pop(old)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    import pyarrow.parquet as pq
    for ep in range(len(DS_LENGTHS)):
        p = root / "data" / f"chunk-{ep:03d}" / f"episode_{ep:06d}.parquet"
        t = pq.read_table(p).rename_columns(
            ["observation.state", "action", "timestamp", "frame_index",
             "episode_index", "index", "task_index"])
        pq.write_table(t, p)

    with LeRobotVideoDataset(root, video_keys=VIDEO_KEYS) as d:
        assert d.state_key == "observation.state"
        assert d.action_key == "action"
        np.testing.assert_array_equal(
            d.load_frame(6)["state"], np.asarray(_state(6), np.float32))


def test_a_video_key_that_is_not_declared_video_is_refused(tmp_path, caplog):
    """An embedded-PNG recording must not be silently read as a video one.

    ``dtype != 'video'`` means the frames are in the parquet cells and
    ``LiberoDataset`` is the right reader; the warning names it.
    """
    import logging

    root = _build_dataset(tmp_path / "png")
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"][VIDEO_KEYS[0]]["dtype"] = "image"
    (root / "meta" / "info.json").write_text(json.dumps(info))

    with caplog.at_level(logging.WARNING,
                         logger="flash_rt.datasets.lerobot_video"):
        with LeRobotVideoDataset(root, video_keys=VIDEO_KEYS):
            pass
    hits = [r for r in caplog.records if "LiberoDataset" in r.getMessage()]
    assert len(hits) == 1, f"expected one warning, got {len(hits)}"

    info["features"].pop(VIDEO_KEYS[1])
    (root / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(KeyError, match=VIDEO_KEYS[1]):
        LeRobotVideoDataset(root, video_keys=VIDEO_KEYS)


# ── metadata / _episode_for / task_for_frame ──────────────────────────────

def test_metadata_carries_the_columns_the_stratified_sampler_needs(ds):
    """``load_calibration_obs`` feeds this straight to the house sampler.

    ``stratified_sample_indices`` reads ``task_index`` / ``episode_index`` /
    ``frame_index`` / ``index``; a recording missing one of them fails at
    calibration time, not here, so pin the contract at the source.
    """
    meta = ds.metadata
    for col in ("task_index", "episode_index", "frame_index", "index"):
        assert col in meta.columns, f"metadata is missing {col!r}"
    assert len(meta) == DS_TOTAL
    assert list(meta["index"]) == list(range(DS_TOTAL))
    assert list(meta["episode_index"]) == [0] * DS_LENGTHS[0] + \
                                          [1] * DS_LENGTHS[1]
    assert list(meta["frame_index"]) == (list(range(DS_LENGTHS[0])) +
                                         list(range(DS_LENGTHS[1])))
    assert ds.metadata is meta, "metadata was rebuilt instead of cached"


def test_tasks_reads_the_jsonl_and_tolerates_its_absence(ds, ds_root):
    assert ds.tasks == {0: DS_TASKS[0], 1: DS_TASKS[1]}

    tasks = ds_root / "meta" / "tasks.jsonl"
    saved = tasks.read_text(encoding="utf-8")
    try:
        tasks.unlink()
        with LeRobotVideoDataset(ds_root, video_keys=VIDEO_KEYS) as d:
            assert d.tasks == {}
            # a missing task string degrades to "", it does not raise
            assert d.task_for_frame(0) == ""
    finally:
        tasks.write_text(saved, encoding="utf-8")


def test_episode_for_maps_across_the_episode_boundary(ds):
    """The mapping the whole reader rests on.

    With ``DS_LENGTHS = (5, 4)`` a ``global // length`` or a per-episode
    counter-reset derivation coincides with the truth only by accident; the
    four corners below plus the two ends of episode 1 pin it down.
    """
    assert ds._episode_for(0) == (0, 0)
    assert ds._episode_for(4) == (0, 4)     # last frame of episode 0
    assert ds._episode_for(5) == (1, 0)     # first frame of episode 1
    assert ds._episode_for(8) == (1, 3)     # last frame of the recording
    for g in range(DS_TOTAL):
        assert ds._episode_for(g) == _ep_fi(g)


def test_an_out_of_range_global_index_is_refused(ds):
    with pytest.raises(KeyError, match="total_frames=9"):
        ds._episode_for(DS_TOTAL)
    with pytest.raises(KeyError, match="total_frames=9"):
        ds._episode_for(-1)
    with pytest.raises(KeyError, match="not in dataset"):
        ds.task_for_frame(DS_TOTAL)


# ── load_frame ────────────────────────────────────────────────────────────

def test_load_frame_returns_the_matching_row_and_both_cameras(ds):
    """Every field has to come from the row whose ``index`` was asked for.

    The images are checked against the (episode, frame) markers baked into the
    pixels, so this also pins that ``load_frame`` decodes the *in-episode*
    index and not the global one -- the two differ for all of episode 1.
    """
    front, wrist = VIDEO_KEYS
    for g in range(DS_TOTAL):
        ep, fi = _ep_fi(g)
        obs = ds.load_frame(g)

        assert obs["index"] == g
        assert obs["episode_index"] == ep
        assert obs["frame_index"] == fi
        assert obs["task"] == DS_TASKS[ep]

        assert obs["state"].dtype == np.float32
        assert obs["action"].dtype == np.float32
        np.testing.assert_array_equal(obs["state"],
                                      np.asarray(_state(g), np.float32))
        np.testing.assert_array_equal(obs["action"],
                                      np.asarray(_action(g), np.float32))

        assert set(obs["images"]) == set(VIDEO_KEYS)
        assert bar_row(obs["images"][front]) == fi % DS_SIDE
        assert bar_row(obs["images"][wrist]) == (fi + WRIST_OFFSET) % DS_SIDE
        for img in obs["images"].values():
            assert img.dtype == np.uint8
            assert img.shape == (DS_SIDE, DS_SIDE, 3)
            assert bar_col(img) == ep * 5, f"frame {g}: wrong episode's video"


def test_state_and_action_can_be_skipped_independently(ds_root):
    with LeRobotVideoDataset(ds_root, video_keys=VIDEO_KEYS,
                             state_key=None, action_key=None) as d:
        assert d.state_key is None and d.action_key is None
        obs = d.load_frame(3)
        assert "state" not in obs and "action" not in obs
        assert bar_row(obs["images"][VIDEO_KEYS[0]]) == 3


def test_load_frame_follows_the_index_column_not_the_row_position(
        ds, ds_shifted):
    """Negative control: shift ``index`` by one and everything must shift.

    Without this the pins above could all be satisfied by a reader that returns
    a constant, or one that selects by row position. Here the parquet rows, the
    state values and the videos are untouched and only the ``index`` column
    reads ``g + 1``, so ``load_frame(g)`` must now serve the row that *was*
    ``g - 1`` -- the neighbouring frame's state, action, task and pixels.
    """
    for g in range(1, DS_TOTAL + 1):
        shifted = ds_shifted.load_frame(g)
        original = ds.load_frame(g - 1)

        assert shifted["index"] == g
        np.testing.assert_array_equal(shifted["state"], original["state"])
        np.testing.assert_array_equal(shifted["action"], original["action"])
        assert shifted["task"] == original["task"]
        assert (shifted["episode_index"], shifted["frame_index"]) == \
               (original["episode_index"], original["frame_index"])
        for key in VIDEO_KEYS:
            np.testing.assert_array_equal(shifted["images"][key],
                                          original["images"][key])

    # the equality above is only meaningful because the states are injective
    distinct = {np.asarray(_state(g), np.float32).tobytes()
                for g in range(DS_TOTAL)}
    assert len(distinct) == DS_TOTAL


# ── _frame_index: the pts derivation and its fallback ─────────────────────

class _Stream:
    def __init__(self, time_base, average_rate):
        self.time_base = time_base
        self.average_rate = average_rate


class _Frame:
    def __init__(self, pts):
        self.pts = pts


def test_frame_index_derives_from_pts_when_the_stream_provides_it():
    stream = _Stream(Fraction(1, 30), Fraction(30, 1))
    for i in (0, 1, 7, 599):
        assert _frame_index(_Frame(i), stream, FPS, fallback=-1) == i

    # a 1/1000 time base, as an mp4 muxer commonly writes
    stream = _Stream(Fraction(1, 1000), Fraction(30, 1))
    assert _frame_index(_Frame(0), stream, FPS, fallback=-1) == 0
    assert _frame_index(_Frame(1000), stream, FPS, fallback=-1) == 30
    assert _frame_index(_Frame(2000), stream, FPS, fallback=-1) == 60


def test_frame_index_uses_fps_when_the_stream_reports_no_rate():
    """``average_rate`` is falsy on some containers; ``fps`` is the substitute."""
    stream = _Stream(Fraction(1, 1000), None)
    assert _frame_index(_Frame(1500), stream, 30, fallback=-1) == 45
    # the same pts maps differently at a different fps, i.e. fps really is used
    assert _frame_index(_Frame(1500), stream, 20, fallback=-1) == 30


def test_frame_index_falls_back_to_the_counter_without_pts_or_time_base():
    """The fallback is what keeps a seek-free forward pass working.

    A stream that reports no pts (or no time_base) has to stay readable, and
    the caller's sequential counter is the only index available. Returning 0 or
    raising would break ascending reads on such a file -- the exact access
    pattern the tail-loss regression above depends on.
    """
    assert _frame_index(_Frame(None), _Stream(Fraction(1, 30), Fraction(30)),
                        FPS, fallback=42) == 42
    assert _frame_index(_Frame(1234), _Stream(None, Fraction(30)),
                        FPS, fallback=42) == 42
    assert _frame_index(_Frame(None), _Stream(None, None),
                        FPS, fallback=0) == 0
