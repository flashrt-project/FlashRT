"""The joins this stack already has, declared.

Descriptive before prescriptive: these ports describe what the bound
structures do *today*, including the joins that already work. If the
vocabulary cannot express the stack as it stands, the vocabulary is
wrong and no behaviour should be changed on top of it.

Each port carries the shape of the tensor it is about in its note, so a
reader composing structures by hand — the explicit door — can see the
agreement without reading the impl.
"""

from __future__ import annotations

from .ports import Port

# --- adaptive norm producing the packed projections' input ------------
# Works today. The producer emits fp8 with a shared act scale and the
# pack skips its own input quantization; the two are bound atomically.
# Declared here to prove the vocabulary can express a join that already
# holds, not to change it.
ADALN_OUT = Port(
    structure="adaln_producer", name="y", direction="out",
    offers={"dtype": ("fp8_static", "bf16"),
            "layout": ("row_major",),
            "carry": ("gated_residual", "none"),
            "cadence": ("per_call",)},
    note="(rows, D) normed activation; carry=gated_residual means the "
         "producer absorbed the pending residual rather than the host "
         "closing it with an elementwise add")

QKV_PACK_IN = Port(
    structure="qkv_pack", name="x", direction="in",
    offers={"dtype": ("fp8_static", "bf16"),
            "layout": ("row_major",)},
    note="(rows, K) shared input of the sibling projections")

DECODER_FFN_IN = Port(
    structure="decoder_ffn", name="x", direction="in",
    offers={"dtype": ("fp8_static", "bf16"),
            "layout": ("row_major",)},
    note="(rows, D) normed input; the bf16 entry fuses its own quantize, "
         "which is why this join only pays when the producer is already "
         "there for another reason")

# --- packed projections feeding the attention core --------------------
# The join that still does cancelling work. The pack writes k/v into its
# own stash buffers and the core copies them into its packed KV region;
# both sides can express the alias, nothing negotiates it yet.
QKV_PACK_OUT = Port(
    structure="qkv_pack", name="kv", direction="out",
    offers={"dtype": ("bf16",),
            "layout": ("bshd", "row_major"),
            "buffer": ("alias", "fresh")},
    note="(rows, N_k) / (rows, N_v) sibling outputs; viewed as "
         "(B, S, H, D) they are already the kernel's layout, and the "
         "stash buffer they land in could be the consumer's region")

ATTENTION_CORE_KV_IN = Port(
    structure="attention_core", name="kv", direction="in",
    offers={"dtype": ("bf16",),
            "layout": ("bshd",),
            "buffer": ("alias", "fresh")},
    note="(B, S_suffix, H_kv, D) suffix keys/values; the packed region's "
         "suffix rows are contiguous, so a producer can write into them "
         "directly")

ATTENTION_CORE_OUT = Port(
    structure="attention_core", name="out", direction="out",
    offers={"dtype": ("bf16",), "layout": ("bshd", "row_major")},
    note="(B, S, H, D). row_major is free *because* the layout is bshd: "
         "its last two axes are contiguous, so the (.., H*D) reshape the "
         "output projection wants is a view. From bhsd the same reshape "
         "would cost a transpose and a copy — which is the whole reason "
         "layout belongs on the join rather than inside either side")

LINEAR_PROJ_IN = Port(
    structure="linear_proj", name="x", direction="in",
    offers={"dtype": ("fp8_static", "bf16"), "layout": ("row_major",)},
    note="(rows, K); the fp8 form has no quantize to amortise, which is "
         "why its work band starts an order of magnitude lower")

# --- the stream-scoped style materialisation --------------------------
# The join that the compiler was allowed to undo until it was declared
# opaque. Declared here because opacity is otherwise invisible: nothing
# about the tensors says the arrangement must survive.
STYLE_BROKER_OUT = Port(
    structure="adaln_producer", name="style", direction="out",
    offers={"layout": ("row_major",),
            "cadence": ("per_step", "per_call"),
            "opacity": ("must_persist", "fusible")},
    note="(rows, 3D) style rows for one step; per_step means one fill "
         "serves every producer on the stream, which only survives when "
         "the fill is opaque to the compiler")

STYLE_CONSUMER_IN = Port(
    structure="adaln_producer", name="style", direction="in",
    offers={"layout": ("row_major",),
            "cadence": ("per_step", "per_call"),
            "opacity": ("must_persist", "fusible")},
    note="(rows, 3D) contiguous, as the kernel requires")


#: Every join the stack has today, producer first. The pairs that
#: already hold are here to be checked against reality; the pairs that
#: do not are here to be measured.
DECLARED = {
    "adaln->qkv_pack": (ADALN_OUT, QKV_PACK_IN),
    "adaln->decoder_ffn": (ADALN_OUT, DECODER_FFN_IN),
    "qkv_pack->attention_core": (QKV_PACK_OUT, ATTENTION_CORE_KV_IN),
    "attention_core->linear_proj": (ATTENTION_CORE_OUT, LINEAR_PROJ_IN),
    "style_broker->producer": (STYLE_BROKER_OUT, STYLE_CONSUMER_IN),
}
