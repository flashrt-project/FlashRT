"""Package a host-captured CUDA graph as an ``frt_model_runtime_v1``.

The provider route for external hosts (torch or any framework that can
capture its hot path): the host warms up and captures on a dedicated
stream, keeps every boundary tensor at a fixed device address, and hands
the raw graph exec plus those boundary windows here. The result is the
same runtime face the native pipelines export from
``flash_rt.models.*.runtime_export``, so any ``cap_model_runtime``
consumer adopts it without knowing which producer built it.

The caller keeps ownership of the captured graph and the window memory;
the returned runtime anchors the exec context and the wrapped buffers,
and must not outlive the host objects backing them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flash_rt.runtime import export as _rt
from flash_rt.runtime.exec import _import_native

_DIR_ROLE = {"in": ("input",), "out": ("output",)}


def export_captured_runtime(
    stream_handle: int,
    graphs: Sequence[tuple[str, int]],
    windows: Mapping[str, tuple[int, int]],
    ports: Sequence[Mapping],
    regions: Sequence[tuple[str, str]] = (),
    stages: Sequence[str] = (),
    roles: Mapping[str, object] | None = None,
    identity: Mapping[str, object] | None = None,
    manifest_extra: Mapping[str, object] | None = None,
):
    """Build an ``frt_model_runtime_v1`` over host-captured graphs.

    - ``stream_handle``: native stream the graphs were captured on (for
      torch, ``torch.cuda.Stream().cuda_stream``).
    - ``graphs``: ``(name, raw_graph_exec)`` pairs; for torch,
      ``torch.cuda.CUDAGraph().raw_cuda_graph_exec()``.
    - ``windows``: ``name -> (device_ptr, nbytes)`` boundary windows the
      captured graphs read/write in place (SWAP semantics).
    - ``ports``: dicts with :class:`flash_rt.runtime.export.PortSpec`
      fields, plus ``window`` naming the backing window. An output port
      may share its window with an input port (in-place convergence).
    - ``regions``: ``(region_name, window_name)`` restorable-state
      regions for capsule snapshot/restore.
    - ``stages``: graph names in sequential step order; defaults to the
      declared graph order.
    - ``roles``: optional ``window_name -> role`` overrides; by default
      a window's role is the union of its ports' directions.
    """
    if not graphs:
        raise ValueError("export_captured_runtime needs at least one graph")
    c = _import_native()
    ctx = c.Ctx()
    sid = ctx.wrap_stream(int(stream_handle))

    graph_specs = []
    for name, graph_exec in graphs:
        g = ctx.graph(name)
        g.adopt(0, int(graph_exec))
        graph_specs.append(_rt.GraphSpec(name, g, 0, (0,), stream="main"))

    wraps = {name: ctx.wrap(name, int(ptr), int(nbytes))
             for name, (ptr, nbytes) in windows.items()}

    port_specs = []
    port_roles: dict[str, set] = {name: set() for name in wraps}
    for p in ports:
        p = dict(p)
        window = p.pop("window")
        if window not in wraps:
            raise ValueError(f"port {p.get('name')!r} names unknown window "
                             f"{window!r}")
        port_roles[window].update(_DIR_ROLE.get(p.get("direction", "in"), ()))
        port_specs.append(_rt.PortSpec(buffer=wraps[window], **p))

    buffer_specs = []
    for name, buf in wraps.items():
        role = (roles or {}).get(name)
        if role is None:
            role = tuple(sorted(port_roles[name])) or ("state",)
        buffer_specs.append(_rt.BufferSpec(name, buf, role))

    region_specs = []
    for rname, window in regions:
        if window not in wraps:
            raise ValueError(f"region {rname!r} names unknown window "
                             f"{window!r}")
        region_specs.append(_rt.RegionSpec(rname, wraps[window]))

    stage_names = list(stages) or [name for name, _ in graphs]
    known = {g.name for g in graph_specs}
    unknown = [s for s in stage_names if s not in known]
    if unknown:
        raise ValueError(f"stages name unknown graphs: {unknown}")

    return _rt.build_model_runtime(
        ctx,
        streams=[_rt.StreamSpec("main", sid,
                                native_handle=int(stream_handle))],
        graphs=graph_specs,
        buffers=buffer_specs,
        regions=region_specs,
        ports=port_specs,
        stages=[_rt.StageSpec(name) for name in stage_names],
        identity={str(k): str(v) for k, v in (identity or {}).items()},
        manifest_extra=dict(manifest_extra or {}),
        owner=(ctx, wraps),
    )
