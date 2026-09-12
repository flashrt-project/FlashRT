"""The GR00T N1.7 Ascend units: their loaders, their refusals, and the one piece
of their arithmetic that can be judged without the part.

Three shared objects are built for this model and each is loaded separately, so
a partial rebuild leaves one behind with the same symbol names and a different
contract behind them. That is what the identity pair is for, and these check
that each of the three loaders actually consults it.

The image transform's tap construction is checked against OpenCV itself. It is
host arithmetic — integers and FP32 weights — so it needs no device, and it is
the part that was wrong three times: the forward scale has to come from the
inverse, each weight is rounded from its own float, and the tail has no second
tap.
"""
import ctypes
import sys

import pytest
import torch

from flash_rt.npu.core import abi


class _Symbol:
    def __init__(self, value=None):
        self._value = value
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self._value


class _FakeLibrary:
    """A CDLL with a chosen identity and every entry point present."""

    def __init__(self, abi_version, soc):
        self._name = "libfake.so"
        self._identity = {
            "flashrt_npu_abi_version": _Symbol(abi_version),
            "flashrt_npu_soc_version": _Symbol(soc.encode()),
        }
        self.bound = []

    def __getattr__(self, name):
        identity = self.__dict__["_identity"]
        if name in identity:
            return identity[name]
        if name.startswith("flashrt_npu_"):
            self.__dict__["bound"].append(name)
            return _Symbol(0)
        raise AttributeError(name)


#: Every unit this model builds: its loader, and the entry point it must bind.
UNITS = [
    ("flash_rt.npu.models.groot_n17.attention", "DitAttentionLibrary",
     "flashrt_npu_dit_attn"),
    ("flash_rt.npu.models.groot_n17.norm", "DitNormLibrary",
     "flashrt_npu_dit_add_layer_norm"),
    ("flash_rt.npu.models.groot_n17.preprocess", "AreaResizeLibrary",
     "flashrt_npu_area_resize"),
]


def _loader(module_name, class_name):
    __import__(module_name)
    return getattr(sys.modules[module_name], class_name)


@pytest.mark.parametrize("module_name,class_name,entry", UNITS)
def test_a_unit_binds_its_entry_point_when_its_identity_matches(
        monkeypatch, module_name, class_name, entry):
    fake = _FakeLibrary(abi.ABI_VERSION, abi.SOC_VERSION)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: fake)
    loader = _loader(module_name, class_name)()
    assert entry in fake.bound, f"{class_name} did not bind {entry}"


@pytest.mark.parametrize("module_name,class_name,entry", UNITS)
def test_a_stale_unit_is_refused_by_its_abi(monkeypatch, module_name, class_name, entry):
    monkeypatch.setattr(
        ctypes, "CDLL",
        lambda path: _FakeLibrary(abi.ABI_VERSION - 1, abi.SOC_VERSION))
    with pytest.raises(ImportError, match="ABI"):
        _loader(module_name, class_name)()


@pytest.mark.parametrize("module_name,class_name,entry", UNITS)
def test_a_unit_built_for_another_part_is_refused(
        monkeypatch, module_name, class_name, entry):
    monkeypatch.setattr(
        ctypes, "CDLL", lambda path: _FakeLibrary(abi.ABI_VERSION, "Ascend910B2"))
    with pytest.raises(ImportError, match="Ascend910B2"):
        _loader(module_name, class_name)()


@pytest.mark.parametrize("module_name,class_name,entry", UNITS)
def test_an_unbuilt_unit_says_to_build_rather_than_that_a_module_is_missing(
        monkeypatch, module_name, class_name, entry):
    def absent(path):
        raise OSError(f"{path}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", absent)
    with pytest.raises(ImportError, match="build.sh"):
        _loader(module_name, class_name)()


# ── the norm wrapper's refusals ───────────────────────────────────────

def _row(width=1536, dtype=torch.bfloat16):
    return torch.zeros(2, width, dtype=dtype)


def test_the_norm_does_not_serve_a_host_tensor():
    """`serves` is what routes between this kernel and the vendor's, so a host
    tensor has to fall out of the native path there rather than reach it."""
    from flash_rt.npu.models.groot_n17 import norm

    assert norm.serves(_row()) is False


def test_the_norm_does_not_serve_a_width_its_broadcast_cannot_carry():
    from flash_rt.npu.models.groot_n17 import norm

    assert norm.serves(_row(width=100)) is False
    assert norm.serves(_row(width=norm.MAX_NORM_COLS + 64)) is False


def test_the_norm_refuses_a_host_row_when_called_directly():
    from flash_rt.npu.models.groot_n17 import norm

    with pytest.raises(ValueError, match="Ascend device"):
        norm.add_layer_norm(_row(), _row(), _row()[0], _row()[0], 1e-5)


def test_the_norm_refuses_a_width_that_is_not_whole_repeats():
    from flash_rt.npu.models.groot_n17 import norm

    narrow = _row(width=100)
    with pytest.raises(ValueError, match="64 elements"):
        norm.add_layer_norm(narrow, narrow, narrow[0], narrow[0], 1e-5)


# ── the image transform's taps, against OpenCV ────────────────────────

cv2 = pytest.importorskip("cv2", reason="the tap check compares against OpenCV")
numpy = pytest.importorskip("numpy")


def _apply_taps(image, taps_h, taps_w):
    """The kernel's arithmetic, in numpy: two horizontal taps in FP32, then the
    8-bit vertical specialisation, which truncates three separate times."""
    first_w, second_w, weight0_w, weight1_w = (t.numpy() for t in taps_w)
    first_h, second_h, weight0_h, weight1_h = (t.numpy() for t in taps_h)
    source = image.astype(numpy.float32)
    rows = (source[:, first_w] * weight0_w[None, :, None]
            + source[:, second_w] * weight1_w[None, :, None])
    rows = numpy.floor(rows / 16.0).astype(numpy.int64)
    top = (rows[first_h] * weight0_h[:, None, None]) >> 16
    bottom = (rows[second_h] * weight1_h[:, None, None]) >> 16
    return ((top + bottom + 2) >> 2).astype(numpy.uint8)


@pytest.mark.parametrize("source,target", [(180, 256), (200, 256), (243, 256),
                                           (243, 244), (13, 17)])
def test_the_taps_reproduce_opencvs_enlarging_inter_area(source, target):
    from flash_rt.npu.models.groot_n17.preprocess import _area_taps

    rng = numpy.random.default_rng(0)
    image = rng.integers(0, 256, (source, source, 3), dtype=numpy.uint8)
    want = cv2.resize(image, (target, target), interpolation=cv2.INTER_AREA)
    taps_h = _area_taps(source, target, "cpu")
    taps_w = _area_taps(source, target, "cpu")
    got = _apply_taps(image, taps_h, taps_w)
    assert numpy.array_equal(got, want), (
        f"{int((got != want).sum())} of {want.size} samples differ at "
        f"{source}->{target}")


def test_the_last_output_column_has_no_second_tap():
    """OpenCV clamps the tail rather than reading past the edge, and the second
    weight has to be zero there or the last column is wrong on every row."""
    from flash_rt.npu.models.groot_n17.preprocess import _area_taps

    first, second, weight0, weight1 = _area_taps(180, 256, "cpu")
    assert int(first[-1]) == 179 and int(second[-1]) == 179
    assert int(weight1[-1]) == 0 and int(weight0[-1]) == 2048


def test_shrinking_is_refused_rather_than_given_the_wrong_weights():
    """OpenCV area-averages over a variable number of taps when it shrinks, so
    these two-tap weights do not describe that branch. A camera whose smallest
    edge already exceeds the target would hit it."""
    from flash_rt.npu.models.groot_n17.preprocess import _area_taps

    with pytest.raises(NotImplementedError, match="does not enlarge"):
        _area_taps(256, 243, "cpu")
    with pytest.raises(NotImplementedError, match="does not enlarge"):
        _area_taps(256, 256, "cpu")


def test_the_forward_scale_comes_from_the_inverse():
    """``1/(200/180)`` is 0.8999999999999999, so index 30 floors to 26 where
    ``30 * (180/200)`` floors to 27 -- a whole source row out, on every column."""
    from flash_rt.npu.models.groot_n17.preprocess import _area_taps

    first, _, _, _ = _area_taps(200, 256, "cpu")
    assert int(first[30]) == 23
    assert int(first[30]) == int(30 * (1.0 / (256 / 200)))
