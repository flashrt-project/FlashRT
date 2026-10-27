# FA2 native C library

FlashRT can build vendored FA2 entry points into a Python-free shared library
for native C++ consumers. This path is opt-in:

```bash
cmake -S . -B <build-dir> -DFLASHRT_ENABLE_NATIVE_CPP=ON
cmake --build <build-dir> --target flashrt_fa2_raw
```

The default remains unchanged: `flash_rt_fa2` directly contains the FA2
objects, no raw library is generated, and existing Python deployments gain no
new dynamic dependency.

With native C++ support enabled:

- `libflashrt_fa2_raw.so` owns the `fvk_attention_fa2_fwd_*` C symbols;
- `flash_rt_fa2` is the Python adapter and links the raw library;
- native C++ runtimes link the same raw library directly.

When both targets are requested, this keeps one implementation for Python and
native producers. Model-specific code must not compile a private copy of the
FA2 wrappers.

This is an internal native linkage boundary, not a new versioned public ABI.
Consumers outside the FlashRT build must continue to use the supported
runtime interfaces rather than bind these kernel entry points directly.

## Packaging contract

In the opt-in native mode, the Python adapter and raw library are one install
unit. Deployments that build the thin adapter must install both files in the
same directory. Both targets use an `$ORIGIN` runtime search path, so no
build-tree path is part of deployment.

Default Python deployments still install only `flash_rt_fa2`. Native mode may
also disable the adapter with
`-DFLASHRT_BUILD_FA2_PYTHON_ADAPTER=OFF` and deploy only the raw library.

## C boundary

The raw library has no Python dependency. Its declarations live in
`csrc/attention/fa2_wrapper.h`; the Python adapter includes that header instead
of maintaining a second set of declarations. Existing `fvk_*` signatures are
unchanged.

The raw library exports exactly the FA2 C symbols in
`csrc/attention/fa2_native_exports.map` (the allow-list). Its causal entry
remains present for a slim native matrix and fails explicitly for an
unsupported dtype/head dimension. The head_dim-64 tile entries and `fvk_attention_fa2_has_xvla_tile64` profile
query are included for the X-VLA adapter. Disabled dtype/profile builds retain
throwing tile stubs, so the adapter imports in FP16-only and slim builds too.
New tile calls take an explicit PDL argument; existing entry signatures and
launch behavior are unchanged. The tile implementation is built only with
`FLASHRT_BUILD_XVLA=ON`, BF16 enabled and head dimension 64 selected.

## Validation

- the default adapter directly contains FA2 objects and has no raw dependency;
- the raw library exports exactly the documented allow-list;
- it has no unresolved Python or `fvk_*` symbols;
- the opt-in thin Python adapter has a dynamic dependency on the raw library;
- native-mode adapter and raw-library runtime search paths are `$ORIGIN`;
- build-mode tests reject a default adapter with a raw dependency and reject a
  thin adapter that embeds a second FA2 object set;
- the Python adapter and native C++ attention gates both execute against the
  shared implementation.
