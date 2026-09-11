#!/usr/bin/env python3
"""Build the FlashRT-edition GGUF: splice an NVFP4 lm-head into a quantized
body.

    python3 splice_nvfp4_head.py <body.gguf> <head-source.gguf> <out.gguf>

body: the shipping quantized model (its output.weight is replaced).
head-source: any GGUF whose output.weight is GGML_TYPE_NVFP4 quantized from
the BF16 checkpoint (e.g. `llama-quantize --output-tensor-type NVFP4` on a
bf16 conversion) — quantizing the head from BF16 measurably beats a rebuild
from an already-quantized head.

The resulting artifact needs no side-band packs: the stock nvfp4 kernels
serve the head, and the adapter's online repack wires everything else.

gguf-py pitfalls encoded here: ReaderTensor.shape is ne-order while
ReaderTensor.data.shape is the byte-shaped numpy-order the writer wants;
non-quantized tensors must go through dtype inference, not raw_dtype; and
field values may be numpy scalars the struct packer rejects.
"""

import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFValueType


def to_py(v):
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [to_py(x) for x in v]
    return v


def main() -> None:
    body, head_src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    hr = GGUFReader(head_src)
    head_t = next(t for t in hr.tensors if t.name == "output.weight")
    assert int(head_t.tensor_type) == 40, f"head-source output.weight is {head_t.tensor_type}, want NVFP4"

    br = GGUFReader(body)
    w = GGUFWriter(out, br.fields["general.architecture"].contents())
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for key, field in br.fields.items():
        if key in skip:
            continue
        vt = field.types[0]
        if vt == GGUFValueType.ARRAY:
            w.add_key_value(key, to_py(field.contents()), vt, sub_type=field.types[1])
        else:
            w.add_key_value(key, to_py(field.contents()), vt)
    for t in br.tensors:
        src = head_t if t.name == "output.weight" else t
        if src.data.dtype == np.uint8:   # quantized: raw bytes + explicit type
            w.add_tensor(src.name, src.data,
                         raw_shape=[int(x) for x in src.data.shape],
                         raw_dtype=src.tensor_type)
        else:                            # f32/f16/...: writer infers from dtype
            w.add_tensor(src.name, np.ascontiguousarray(src.data))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out} (NVFP4 head spliced)")


if __name__ == "__main__":
    main()
