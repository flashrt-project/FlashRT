# Thor reproduction runtime

Start with [Thor setup](../../docs/thor/README.md), then [OpenPI π0.5](../../docs/thor/pi05.md) or [GR00T N1.7](../../docs/thor/groot-n17.md).

The source-only archive preserves the previously validated Thor runtime while the main repository continues to evolve. It contains tracked source files, with no Git history, model weights or compiled binaries. `unpack-runtime.py` verifies it against `versions.json` before extracting. Native setup and Docker builds use this same source. Reference model versions and mirror revisions are managed by the scripts and lock file.

For a new release, build and rerun both model validators. The fixed fixtures establish numerical consistency, not robot task success. GR00T timing is a feature-input graph benchmark. See the model guides for its exact boundary.
