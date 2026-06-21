#!/usr/bin/env python3
"""Unit tests for the stage-7.1 WeightLoader scaffolding.

CPU-only, no torch / jax / fvk dependencies. Validates:
  * Protocol shape: ``WeightSource`` / ``Transform`` / ``WeightSink`` can
    be implemented by plain Python classes and satisfy ``isinstance`` via
    ``runtime_checkable``.
  * Dataclasses (``Item``, ``LayerBlock``, ``ModelWeightSpec``,
    ``BufferSpec``, ``LoaderContext``) construct and roundtrip cleanly.
  * ``WeightLoader`` rejects a non-conforming source with a typed error.
  * ``WeightLoader.run`` returns a ``LoaderContext`` for an empty spec without
    publishing scales or scratch state.

Run: python3 tests/test_weight_loader.py
"""

from __future__ import annotations

import sys
import types
import traceback

import numpy as np

from flash_rt.core.weights.loader import _load_orbax
from flash_rt.executors.weight_loader import (
    BufferSpec,
    Item,
    LayerBlock,
    LoaderContext,
    ModelWeightSpec,
    Transform,
    WeightLoader,
    WeightSink,
    WeightSource,
)


class _DictSource:
    def __init__(self, d): self._d = d
    def get(self, key): return self._d[key]
    def has(self, key): return key in self._d


class _IdentityTransform:
    def apply(self, tensor, ctx): return tensor


class _ListSink:
    def __init__(self): self.items = []; self.scales = []
    def store(self, tensor, *, scale=None):
        self.items.append(tensor)
        if scale is not None:
            self.scales.append(scale)
    def finalize(self): pass


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_protocols_runtime_checkable():
    _expect(isinstance(_DictSource({}), WeightSource),
            "DictSource should satisfy WeightSource protocol")
    _expect(isinstance(_IdentityTransform(), Transform),
            "IdentityTransform should satisfy Transform protocol")
    _expect(isinstance(_ListSink(), WeightSink),
            "ListSink should satisfy WeightSink protocol")
    # A plain object should NOT satisfy WeightSource.
    _expect(not isinstance(object(), WeightSource),
            "plain object must not satisfy WeightSource")


def test_item_construction():
    sink = _ListSink()
    it = Item(name="qkv_w",
              key="{prefix}.qkv.weight",
              transforms=[_IdentityTransform()],
              sink=sink,
              scale_into="enc_scales")
    _expect(it.name == "qkv_w", "Item.name")
    _expect(it.scale_into == "enc_scales", "Item.scale_into")
    _expect(len(it.transforms) == 1, "Item.transforms len")
    _expect(it.sink is sink, "Item.sink identity")


def test_layer_block_construction():
    sink = _ListSink()
    block = LayerBlock(
        prefix_fmt="model.layers.{i}",
        num_layers=18,
        items=[Item(name="w", key="{prefix}.w", sink=sink)],
        name="encoder",
    )
    _expect(block.num_layers == 18, "LayerBlock.num_layers")
    _expect(block.prefix_fmt.format(i=3) == "model.layers.3",
            "prefix_fmt formatting")


def test_model_weight_spec():
    spec = ModelWeightSpec(
        framework="torch",
        blocks=[LayerBlock("l.{i}", 1, [Item("w", "l.{i}.w", sink=_ListSink())])],
        singletons=[Item("emb", "embedding.weight", sink=_ListSink())],
        buffers=[BufferSpec(attr="_x", shape=("S", "D"), dtype="fp16")],
        dims={"S": 256, "D": 1152},
    )
    _expect(spec.framework == "torch", "spec.framework")
    _expect(len(spec.blocks) == 1 and len(spec.singletons) == 1, "spec counts")
    _expect(spec.dims["S"] == 256, "spec.dims")
    _expect(spec.buffers[0].dtype == "fp16", "BufferSpec.dtype")


def test_loader_context_defaults():
    ctx = LoaderContext(source=_DictSource({}), target=object())
    _expect(ctx.prefix == "" and ctx.layer_idx == -1, "LoaderContext defaults")
    _expect(ctx.scales == {} and ctx.scratch == {}, "LoaderContext mutable defaults independent")
    ctx2 = LoaderContext(source=_DictSource({}), target=object())
    ctx.scales["a"] = [1.0]
    _expect("a" not in ctx2.scales, "LoaderContext.scales must not share state across instances")


def test_weight_loader_rejects_bad_source():
    spec = ModelWeightSpec(framework="torch")
    try:
        WeightLoader(source=object(), target=None, spec=spec)  # type: ignore[arg-type]
    except TypeError as e:
        _expect("WeightSource" in str(e), "error must mention WeightSource")
        return
    raise AssertionError("expected TypeError for non-conforming source")


def test_weight_loader_rejects_bad_spec():
    src = _DictSource({})
    try:
        WeightLoader(source=src, target=None, spec="not a spec")  # type: ignore[arg-type]
    except TypeError as e:
        _expect("ModelWeightSpec" in str(e), "error must mention ModelWeightSpec")
        return
    raise AssertionError("expected TypeError for non-ModelWeightSpec spec")


def _load_orbax_with_fake_modules(metadata, restored):
    seen = {}

    class _PyTreeCheckpointer:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass
        def metadata(self, path):
            seen["metadata_path"] = path
            return metadata
        def restore(self, path, restore_args):
            seen["restore_path"] = path
            seen["restore_item"] = restore_args.item
            return restored

    class _PyTreeRestore:
        def __init__(self, *, item, restore_args):
            self.item = item
            self.restore_args = restore_args

    class _ArrayRestoreArgs:
        def __init__(self, *, sharding, restore_type):
            self.sharding = sharding
            self.restore_type = restore_type

    ocp = types.ModuleType("orbax.checkpoint")
    ocp.PyTreeCheckpointer = _PyTreeCheckpointer
    ocp.ArrayRestoreArgs = _ArrayRestoreArgs
    ocp.args = types.SimpleNamespace(
        PyTreeRestore=_PyTreeRestore,
    )

    orbax = types.ModuleType("orbax")
    orbax.checkpoint = ocp

    jax = types.ModuleType("jax")
    jax.devices = lambda: ["fake_device"]
    jax.sharding = types.SimpleNamespace(
        Mesh=lambda devices, names: ("mesh", tuple(devices), tuple(names)),
        NamedSharding=lambda mesh, spec: ("named", mesh, spec),
        PartitionSpec=lambda: ("partition",),
    )
    jax.tree = types.SimpleNamespace(map=lambda fn, tree: fn(tree))

    traverse_util = types.ModuleType("flax.traverse_util")
    def _flatten_dict(d, sep="."):
        out = {}
        stack = [("", d)]
        while stack:
            prefix, value = stack.pop()
            if isinstance(value, dict):
                for key, child in value.items():
                    stack.append((f"{prefix}{sep}{key}" if prefix else key,
                                  child))
            else:
                out[prefix] = value
        return out
    traverse_util.flatten_dict = _flatten_dict
    flax = types.ModuleType("flax")
    flax.traverse_util = traverse_util

    saved_modules = {
        name: sys.modules.get(name)
        for name in ("orbax", "orbax.checkpoint", "jax", "flax",
                     "flax.traverse_util")
    }
    sys.modules.update({
        "orbax": orbax,
        "orbax.checkpoint": ocp,
        "jax": jax,
        "flax": flax,
        "flax.traverse_util": traverse_util,
    })
    try:
        weights = _load_orbax("/tmp/fake_orbax")
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    return seen, weights


def test_orbax_step_metadata_loader_path():
    params_tree = {
        "foo": {
            "bar": {
                "value": object(),
            },
        },
    }
    restored = {
        "params": {
            "foo": {
                "bar": {
                    "value": np.array([1.0, 2.0], dtype=np.float16),
                },
            },
        },
    }

    class _ItemMetadata:
        tree = {"params": params_tree}

    class _StepMetadata:
        item_metadata = _ItemMetadata()

    seen, weights = _load_orbax_with_fake_modules(_StepMetadata(), restored)

    _expect(seen["metadata_path"] == "/tmp/fake_orbax",
            "metadata path should point to the params dir")
    _expect(seen["restore_path"] == "/tmp/fake_orbax",
            "restore path should point to the params dir")
    _expect(seen["restore_item"] == {"params": params_tree},
            "StepMetadata item_metadata.tree params should seed restore item")
    _expect(list(weights) == ["foo.bar"],
            "NNX .value suffix should be stripped from flattened keys")
    _expect(weights["foo.bar"].dtype == np.float32,
            "loaded arrays should be converted to float32")
    _expect(weights["foo.bar"].tolist() == [1.0, 2.0],
            "loaded values should be preserved")


def test_orbax_dict_metadata_loader_path():
    params_tree = {
        "foo": {
            "baz": object(),
        },
    }
    restored = {
        "params": {
            "foo": {
                "baz": np.array([3.0, 4.0], dtype=np.float32),
            },
        },
    }
    seen, weights = _load_orbax_with_fake_modules(
        {"params": params_tree}, restored)

    _expect(seen["metadata_path"] == "/tmp/fake_orbax",
            "metadata path should point to the params dir")
    _expect(seen["restore_path"] == "/tmp/fake_orbax",
            "restore path should point to the params dir")
    _expect(seen["restore_item"] == {"params": params_tree},
            "dict metadata params should seed restore item")
    _expect(list(weights) == ["foo.baz"],
            "dict metadata path should preserve flattened keys")
    _expect(weights["foo.baz"].dtype == np.float32,
            "float32 loaded arrays should keep float32 dtype")
    _expect(weights["foo.baz"].tolist() == [3.0, 4.0],
            "loaded values should be preserved")


def test_orbax_mapping_metadata_loader_path():
    params_tree = {
        "foo": {
            "qux": object(),
        },
    }
    restored = {
        "params": {
            "foo": {
                "qux": np.array([5.0, 6.0], dtype=np.float32),
            },
        },
    }

    class _MappingMetadata:
        def __getitem__(self, key):
            if key != "params":
                raise KeyError(key)
            return params_tree

    seen, weights = _load_orbax_with_fake_modules(
        _MappingMetadata(), restored)

    _expect(seen["restore_item"] == {"params": params_tree},
            "mapping metadata params should seed restore item")
    _expect(list(weights) == ["foo.qux"],
            "mapping metadata path should preserve flattened keys")
    _expect(weights["foo.qux"].tolist() == [5.0, 6.0],
            "loaded values should be preserved")


def test_weight_loader_run_empty_spec_returns_context():
    src = _DictSource({})
    spec = ModelWeightSpec(framework="torch")
    loader = WeightLoader(source=src, target=None, spec=spec)
    ctx = loader.run()
    _expect(ctx.source is src, "run() should return a context with the source")
    _expect(ctx.target is None, "run() should preserve the target")
    _expect(ctx.scales == {}, "empty spec should not publish scales")
    _expect(ctx.scratch == {}, "empty spec should not leave scratch state")


TESTS = [
    test_protocols_runtime_checkable,
    test_item_construction,
    test_layer_block_construction,
    test_model_weight_spec,
    test_loader_context_defaults,
    test_weight_loader_rejects_bad_source,
    test_weight_loader_rejects_bad_spec,
    test_orbax_step_metadata_loader_path,
    test_orbax_dict_metadata_loader_path,
    test_orbax_mapping_metadata_loader_path,
    test_weight_loader_run_empty_spec_returns_context,
]


# ════════════════════════════════════════════════════════════════════
#  Stage 7.2 — torch adapter integration tests (need torch + CUDA)
# ════════════════════════════════════════════════════════════════════

def _torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def test_torch_dict_source_and_sinks():
    if not _torch_available():
        print("    (skipped — no torch/CUDA)")
        return
    import torch
    from flash_rt.executors.torch_weights import (
        Attr, DictSource, FlatCat, TensorList, ToFp16, WeightLoader,
    )

    class _Target: pass
    tgt = _Target()

    src = DictSource({
        "embed": torch.randn(4, 8, device="cuda"),
        "l.0.w": torch.randn(2, 3, device="cuda"),
        "l.1.w": torch.randn(2, 3, device="cuda"),
    })
    spec = ModelWeightSpec(
        framework="torch",
        singletons=[Item("emb", "embed", transforms=[ToFp16()], sink=Attr("embedding"))],
        blocks=[LayerBlock(
            prefix_fmt="l.{i}",
            num_layers=2,
            items=[
                Item("w_list", "{prefix}.w", transforms=[ToFp16()], sink=TensorList("w_list")),
                Item("w_flat", "{prefix}.w", transforms=[ToFp16()], sink=FlatCat("w_flat")),
            ],
        )],
    )
    ctx = WeightLoader(source=src, target=tgt, spec=spec).run()
    _expect(hasattr(tgt, "embedding"), "Attr sink set target.embedding")
    _expect(tgt.embedding.dtype == torch.float16, "Attr sink fp16 cast")
    _expect(isinstance(tgt.w_list, list) and len(tgt.w_list) == 2,
            "TensorList accumulated 2 iterations")
    _expect(tgt.w_flat.shape == (2 * 2 * 3,), f"FlatCat shape, got {tgt.w_flat.shape}")
    _expect(ctx.scales == {}, "no scale_into set → ctx.scales empty")


def test_torch_cat_composite_source():
    if not _torch_available():
        print("    (skipped — no torch/CUDA)")
        return
    import torch
    from flash_rt.executors.torch_weights import (
        Attr, Cat, DictSource, ToFp16, WeightLoader,
    )

    class _Target: pass
    tgt = _Target()
    src = DictSource({
        "q": torch.ones(4, 8, device="cuda"),
        "k": torch.full((2, 8), 2.0, device="cuda"),
        "v": torch.full((2, 8), 3.0, device="cuda"),
    })
    spec = ModelWeightSpec(
        framework="torch",
        singletons=[Item("qkv",
                         Cat(["q", "k", "v"], dim=0),
                         transforms=[ToFp16()],
                         sink=Attr("qkv"))],
    )
    WeightLoader(source=src, target=tgt, spec=spec).run()
    _expect(tgt.qkv.shape == (8, 8), f"Cat shape, got {tgt.qkv.shape}")
    _expect(float(tgt.qkv[0, 0]) == 1.0 and float(tgt.qkv[4, 0]) == 2.0
            and float(tgt.qkv[6, 0]) == 3.0, "Cat dim-0 order Q|K|V")


def test_torch_fused_qkv_with_norm_and_interleave():
    if not _torch_available():
        print("    (skipped — no torch/CUDA)")
        return
    import torch
    from flash_rt.core.thor_frontend_utils import interleave_qk as core_iqk
    from flash_rt.executors.torch_weights import (
        Attr, DictSource, FusedQKV, WeightLoader,
    )

    class _Target: pass
    tgt = _Target()

    # Shapes mimic Pi0.5 encoder: NH=8 heads, HD=256, D=2048; GQA K has 1 head.
    NH_q, NH_k, HD, D = 8, 1, 256, 2048
    q = torch.randn(NH_q * HD, D, device="cuda")
    k = torch.randn(NH_k * HD, D, device="cuda")
    v = torch.randn(NH_k * HD, D, device="cuda")
    norm = torch.randn(D, device="cuda") * 0.1
    src = DictSource({"q": q, "k": k, "v": v, "norm": norm})

    spec = ModelWeightSpec(
        framework="torch",
        singletons=[Item("qkv",
                         FusedQKV(q="q", k="k", v="v",
                                  norm_fuse="norm",
                                  interleave_q_heads=NH_q,
                                  interleave_k_heads=NH_k),
                         sink=Attr("qkv"))],
    )
    WeightLoader(source=src, target=tgt, spec=spec).run()

    # Reference computation — must match FusedQKV exactly.
    fa = (1.0 + norm.float()).unsqueeze(0)
    q_ref = core_iqk(q.float(), NH_q) * fa
    k_ref = core_iqk(k.float(), NH_k) * fa
    v_ref = v.float() * fa
    ref = torch.cat([q_ref, k_ref, v_ref], dim=0).to(torch.float16)
    _expect(tgt.qkv.shape == ref.shape, f"shape {tgt.qkv.shape} vs {ref.shape}")
    _expect(torch.equal(tgt.qkv, ref), "FusedQKV must be bit-identical to hand computation")


def test_torch_fused_gate_up_with_norm():
    if not _torch_available():
        print("    (skipped — no torch/CUDA)")
        return
    import torch
    from flash_rt.executors.torch_weights import (
        Attr, DictSource, FusedGateUp, WeightLoader,
    )

    class _Target: pass
    tgt = _Target()
    H, D = 16384, 2048
    gate = torch.randn(H, D, device="cuda")
    up = torch.randn(H, D, device="cuda")
    norm = torch.randn(D, device="cuda") * 0.1
    src = DictSource({"gate": gate, "up": up, "norm": norm})

    spec = ModelWeightSpec(
        framework="torch",
        singletons=[Item("gu",
                         FusedGateUp(gate="gate", up="up", norm_fuse="norm"),
                         sink=Attr("gu"))],
    )
    WeightLoader(source=src, target=tgt, spec=spec).run()
    ff = (1.0 + norm.float()).unsqueeze(0)
    gw = (gate.float() * ff).to(torch.float16)
    uw = (up.float() * ff).to(torch.float16)
    ref = torch.cat([gw, uw], dim=0)
    _expect(torch.equal(tgt.gu, ref), "FusedGateUp (with norm) bit-identical")


def test_torch_quant_and_scale_publication():
    if not _torch_available():
        print("    (skipped — no torch/CUDA)")
        return
    import torch
    from flash_rt.core.thor_frontend_utils import quant_fp8
    from flash_rt.executors.torch_weights import (
        DictSource, Quant, T, TensorList, ToFp16, WeightLoader,
    )

    class _Target: pass
    tgt = _Target()

    W = [torch.randn(8, 16, device="cuda") for _ in range(3)]
    src = DictSource({f"l.{i}.w": W[i] for i in range(3)})

    spec = ModelWeightSpec(
        framework="torch",
        blocks=[LayerBlock("l.{i}", 3, [
            Item("w", "{prefix}.w",
                 transforms=[ToFp16(), T(), Quant()],
                 sink=TensorList("weights"),
                 scale_into="w_scales"),
        ])],
    )
    WeightLoader(source=src, target=tgt, spec=spec).run()

    # Reference bit-match: same op order as Item transforms.
    ref_fp8 = []
    ref_scales = []
    for w in W:
        fp8, s = quant_fp8(w.to(torch.float16).T.contiguous())
        ref_fp8.append(fp8)
        ref_scales.append(s)

    _expect(len(tgt.weights) == 3, "3 quantized tensors collected")
    for i in range(3):
        _expect(torch.equal(tgt.weights[i].view(torch.uint8),
                            ref_fp8[i].view(torch.uint8)),
                f"layer {i} FP8 bytes bit-identical")
    _expect(hasattr(tgt, "w_scales") and len(tgt.w_scales) == 3,
            "scale_into published as target.w_scales")
    for i in range(3):
        _expect(tgt.w_scales[i] == ref_scales[i],
                f"layer {i} scale bit-identical")


def test_torch_run_idempotent():
    """Running the same loader twice must yield identical target state
    — sinks must reset between runs (used by frontend recalibrate paths)."""
    if not _torch_available():
        print("    (skipped — no torch/CUDA)")
        return
    import torch
    from flash_rt.executors.torch_weights import (
        DictSource, TensorList, ToFp16, WeightLoader,
    )

    class _Target: pass
    tgt = _Target()
    src = DictSource({"l.0.w": torch.randn(2, 2, device="cuda"),
                      "l.1.w": torch.randn(2, 2, device="cuda")})
    spec = ModelWeightSpec(
        framework="torch",
        blocks=[LayerBlock("l.{i}", 2, [
            Item("w", "{prefix}.w", transforms=[ToFp16()], sink=TensorList("ws")),
        ])],
    )
    loader = WeightLoader(source=src, target=tgt, spec=spec)
    loader.run()
    first = [t.clone() for t in tgt.ws]
    loader.run()
    _expect(len(tgt.ws) == 2, "second run did not accumulate into first list")
    for a, b in zip(first, tgt.ws):
        _expect(torch.equal(a, b), "second run reproduced identical tensors")


TESTS += [
    test_torch_dict_source_and_sinks,
    test_torch_cat_composite_source,
    test_torch_fused_qkv_with_norm_and_interleave,
    test_torch_fused_gate_up_with_norm,
    test_torch_quant_and_scale_publication,
    test_torch_run_idempotent,
]


# ════════════════════════════════════════════════════════════════════
#  Stage 7.5 — jax adapter smoke tests (need jax + CudaBuffer)
# ════════════════════════════════════════════════════════════════════

def _jax_available():
    try:
        import jax  # noqa
        from flash_rt.core.cuda_buffer import CudaBuffer  # noqa
        return True
    except Exception:
        return False


def test_jax_orbax_dict_source_and_numpy_sinks():
    if not _jax_available():
        print("    (skipped — no jax)")
        return
    import numpy as np
    from flash_rt.executors.weight_loader import WeightLoader
    from flash_rt.executors.jax_weights import (
        Astype, NumpyAttr, NumpyList, OrbaxDictSource,
    )

    class _Target: pass
    tgt = _Target()
    src = OrbaxDictSource({
        "emb": np.arange(12, dtype=np.float32).reshape(3, 4),
        "l.0.mod": np.ones(4, dtype=np.float32),
        "l.1.mod": np.full(4, 2.0, dtype=np.float32),
    })
    spec = ModelWeightSpec(
        framework="jax",
        singletons=[Item("emb", "emb", [Astype(np.float16)], NumpyAttr("embedding"))],
        blocks=[LayerBlock("l.{i}", 2, [
            Item("mod", "{prefix}.mod", [Astype(np.float16)], NumpyList("_mod_w")),
        ])],
    )
    WeightLoader(source=src, target=tgt, spec=spec).run()
    _expect(tgt.embedding.dtype == np.float16, "NumpyAttr astype applied")
    _expect(tgt.embedding.shape == (3, 4), "NumpyAttr shape preserved")
    _expect(isinstance(tgt._mod_w, list) and len(tgt._mod_w) == 2,
            "NumpyList accumulated per-layer")
    _expect(float(tgt._mod_w[1][0]) == 2.0, "NumpyList layer 1 value")


def test_jax_quant_and_cuda_buffer_flat():
    if not _jax_available():
        print("    (skipped — no jax)")
        return
    import jax.numpy as jnp
    import numpy as np
    from flash_rt.executors.weight_loader import WeightLoader
    from flash_rt.executors.jax_weights import (
        CudaBufferFlat, JaxQuant, OrbaxDictSource,
    )

    class _Target:
        def __init__(self): self._cache_blobs = {}

    tgt = _Target()
    W = [np.random.randn(4, 8).astype(np.float32) for _ in range(3)]
    src = OrbaxDictSource({f"l.{i}.w": W[i] for i in range(3)})

    spec = ModelWeightSpec(
        framework="jax",
        blocks=[LayerBlock("l.{i}", 3, [
            Item("w", "{prefix}.w",
                 transforms=[JaxQuant()],
                 sink=CudaBufferFlat("w_flat", cache="w_flat.blob"),
                 scale_into="w_scales"),
        ])],
    )
    WeightLoader(source=src, target=tgt, spec=spec).run()

    _expect(hasattr(tgt, "w_flat"), "CudaBufferFlat wrote target.w_flat")
    # Flat buffer: 3 layers * 4 * 8 = 96 uint8 bytes (fp8-viewed).
    _expect(tgt.w_flat.nbytes == 3 * 4 * 8,
            f"flat nbytes {tgt.w_flat.nbytes}")
    _expect("w_flat.blob" in tgt._cache_blobs,
            "cache blob recorded under spec-given name")
    _expect(len(tgt._cache_blobs["w_flat.blob"]) == 3 * 4 * 8,
            "cache blob length matches buffer element count")
    _expect(hasattr(tgt, "w_scales") and len(tgt.w_scales) == 3,
            "scale_into publishes 3 scales (one per layer)")
    for s in tgt.w_scales:
        _expect(s > 0, f"non-zero scale, got {s}")


def test_jax_cuda_buffer_attr_single():
    if not _jax_available():
        print("    (skipped — no jax)")
        return
    import numpy as np
    from flash_rt.executors.weight_loader import WeightLoader
    from flash_rt.executors.jax_weights import (
        Astype, CudaBufferAttr, OrbaxDictSource, Transpose,
    )

    class _Target:
        def __init__(self): self._cache_blobs = {}

    tgt = _Target()
    W = np.arange(12, dtype=np.float32).reshape(3, 4)
    src = OrbaxDictSource({"proj": W})
    spec = ModelWeightSpec(
        framework="jax",
        singletons=[Item("proj",
                         "proj",
                         [Transpose(), Astype(np.float16)],
                         sink=CudaBufferAttr("proj_w", cache="proj_w.blob"))],
    )
    WeightLoader(source=src, target=tgt, spec=spec).run()
    _expect(hasattr(tgt, "proj_w"), "CudaBufferAttr wrote target.proj_w")
    _expect("proj_w.blob" in tgt._cache_blobs, "cache blob recorded")
    # After .T the shape should be (4, 3), fp16 = 2 bytes per elem.
    _expect(len(tgt._cache_blobs["proj_w.blob"]) == 4 * 3 * 2,
            f"cache blob size {len(tgt._cache_blobs['proj_w.blob'])}")


TESTS += [
    test_jax_orbax_dict_source_and_numpy_sinks,
    test_jax_quant_and_cuda_buffer_flat,
    test_jax_cuda_buffer_attr_single,
]


def main():
    passed = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    total = len(TESTS)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
