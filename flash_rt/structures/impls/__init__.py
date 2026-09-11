"""Structure implementations.

``hub_kernel`` is the shared, process-wide hub loader: two impls that
depend on the same kernel repo must share one loaded module — a second
``kernels.get_kernel`` import of the same repo re-registers its fake
ops and torch.library raises.

The loader also checks the package's own hardware declaration. A Hub
kernel package ships ``metadata.json`` with the CUDA archs it was built
for; that file is maintained on the kernels side and is the single
source of truth for hardware support — this layer reads it, it does not
keep a second table. A device outside the declared archs gets a clean
refusal here, before the kernel produces an unrelated-looking runtime
error; the refusal is caught by the binder and recorded in the plan
notes like any other. A package without metadata is loaded as before —
absence of a declaration is not evidence of incompatibility.
"""

import os
import json
import pathlib
import re
from functools import lru_cache


def _device_cc() -> tuple[int, int] | None:
    """Compute capability of the current CUDA device, or ``None``."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def _declared_archs(module) -> list[str] | None:
    """The package's own ``backend.archs`` declaration, if it ships one."""
    try:
        meta = pathlib.Path(module.__file__).parent / "metadata.json"
        if not meta.is_file():
            return None
        archs = json.loads(meta.read_text()).get("backend", {}).get("archs")
        return list(archs) if archs else None
    except (OSError, ValueError, AttributeError):
        return None


_CUDA_ARCH = re.compile(
    r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)"
    r"(?P<specific>a)?(?P<ptx>\+PTX)?$")


def _cuda_arch_supports_device(
        arch: str, device_cc: tuple[int, int]) -> bool:
    """Whether one Hub CUDA arch declaration can execute on ``device_cc``.

    Plain cubins are binary-compatible with later minor capabilities in
    the same major family. Generic PTX is forward-compatible with any
    greater compute capability. Architecture-specific ``a`` targets are
    exact-only, including when they also carry PTX.
    """
    match = _CUDA_ARCH.fullmatch(arch)
    if match is None:
        return False
    target = (int(match["major"]), int(match["minor"]))
    if match["specific"]:
        return device_cc == target
    if match["ptx"]:
        return device_cc >= target
    return device_cc[0] == target[0] and device_cc[1] >= target[1]


class KernelUnavailable(ValueError):
    """This host cannot supply this kernel package.

    One exception type for every way the distribution layer can come up
    empty — the repository is not published, not staged in an offline
    cache, has no build variant for the host, or will not import here.
    They differ only in what an operator has to go fix, which is what
    the message carries; to a caller they are the same event, and the
    same one the arch declaration produces: *not here*.

    A ``ValueError`` subclass on purpose. Every layer that already
    treats a refusal as an outcome to record rather than an error to
    propagate — the variant families, the recipe engine's per-lever
    build — catches ``ValueError``, and an absent package must not be
    the one refusal that aborts a run instead of being written down.
    """


#: every package this process could not supply, in the order it was
#: asked for. Skipping an unavailable package keeps a run moving, which
#: is the right behaviour — but a package that is *broken* here and one
#: that was simply never shipped here both come out as "skipped", and
#: only the first is somebody's bug. So nothing is inferred and nothing
#: is dropped: the original failure is recorded verbatim and travels
#: into the receipt, where a reader can tell the two apart.
_UNAVAILABLE: list[dict] = []


def unavailable_report() -> list[dict]:
    """Packages this process asked for and could not get."""
    return [dict(row) for row in _UNAVAILABLE]


def clear_unavailable_report() -> None:
    _UNAVAILABLE.clear()


def _record_unavailable(repo: str, version: str, cause: BaseException):
    row = {
        "repo": repo,
        "version": version,
        "error": type(cause).__name__,
        "detail": str(cause)[:400],
    }
    if not any(r["repo"] == repo and r["error"] == row["error"]
               for r in _UNAVAILABLE):
        _UNAVAILABLE.append(row)
    return row


def _check_arch(repo: str, module) -> None:
    archs = _declared_archs(module)
    if archs is None:
        return
    cc = _device_cc()
    if cc is None:
        # no CUDA device: binding fails later at weight transfer anyway;
        # the arch check has nothing truthful to say here
        return
    want = f"{cc[0]}.{cc[1]}"
    if any(_cuda_arch_supports_device(a, cc) for a in archs):
        return
    refusal = KernelUnavailable(
        f"refused: kernel package {repo!r} declares archs {archs}, "
        f"device is sm {want}")
    _record_unavailable(repo, "declared-archs", refusal)
    raise refusal


#: modules cached independently of the arch check: ``get_kernel`` must
#: run at most once per repo even when the check refuses (a second load
#: re-registers the package's fake ops and torch.library raises — the
#: refusal path must not manufacture that error on retry)
_LOADED: dict[tuple[str, str], object] = {}


@lru_cache(maxsize=None)
def hub_kernel(repo: str, version: str):
    try:
        from kernels import get_kernel
    except ImportError as absent:
        # The client itself is missing or shadowed. This is the same
        # event as every other way a package fails to arrive - "not
        # here" - and it must travel as one, or the layers that catch a
        # refusal to record it and keep going will instead abort on the
        # one unavailability nobody declared. It is also the state a
        # fresh ``pip install flash-rt`` is in, since the client is not
        # a hard dependency, so the message says how to leave it.
        _record_unavailable(repo, version, absent)
        raise KernelUnavailable(
            f"kernel package {repo!r} ({version}) is unavailable on this "
            f"host: the kernel client is not installed "
            f"({type(absent).__name__}: {absent}).\n"
            f"    pip install kernels\n"
            f"or install this distribution with its hub extra:\n"
            f"    pip install 'flash-rt[hub]'") from absent

    key = (repo, version)
    if key not in _LOADED:
        # author pin for artifact bisection: an exact hub revision
        # outranks version resolution for this repo only. A perf or
        # correctness drift that arrives with a rebuilt artifact is
        # isolated by flipping one env var, not by editing caches.
        rev = os.environ.get(
            "FRT_KERNEL_REV_" + re.sub(r"[^A-Za-z0-9]", "_",
                                       repo).upper())
        try:
            import inspect as _ins
            _kw = {}
            if "trust_remote_code" in _ins.signature(
                    get_kernel).parameters:
                # the trust gate arrived with newer kernels; our own
                # first-party artifacts are the explicit trust set
                _kw["trust_remote_code"] = True
            try:
                try:
                    _LOADED[key] = (get_kernel(repo, revision=rev,
                                               **_kw)
                                    if rev
                                    else get_kernel(repo,
                                                    version=version,
                                                    **_kw))
                except ValueError as ve:
                    # newer kernels resolve an exact integer version
                    # where older ones accepted a range string; the
                    # range's floor is the same request in both bands
                    m = re.match(r"^\s*>=\s*v?(\d+)", str(version))
                    if not (m and "available versions" in str(ve)):
                        raise
                    _LOADED[key] = get_kernel(
                        repo, version=int(m.group(1)), **_kw)
            except TypeError:
                # kernels<0.13 — the band transformers pins — has no
                # semver resolution kwarg; the default revision is
                # exactly what that library resolved before semver
                # tags existed. Widest-band compat: 0.12 through 0.16
                # serve the same call site.
                _LOADED[key] = (get_kernel(repo, revision=rev) if rev
                                else get_kernel(repo))  # pre-semver band
        except (OSError, RuntimeError, ValueError) as unavailable:
            _record_unavailable(repo, version, unavailable)
            raise KernelUnavailable(
                f"kernel package {repo!r} ({version}) is unavailable on "
                f"this host: {type(unavailable).__name__}: "
                f"{unavailable}") from unavailable
    module = _LOADED[key]
    _check_arch(repo, module)
    return module
