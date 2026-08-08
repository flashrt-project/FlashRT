"""Host-family adapters — where a structure's seam is host-specific.

Importing this package registers the built-in adapters with autobuild.
Attention seams (attention_core) live here because where the attention
math runs differs by host family; a static module pattern cannot find
them, so each family gets a small adapter.
"""
from ..autobuild import (
    register_attention_adapter,
    register_gated_delta_adapter,
    register_qk_norm_rope_adapter,
    register_qkv_rope_adapter,
)
from .diffusers_attention import DiffusersAttentionAdapter
from .diffusers_rotary_attention import DiffusersRotaryAttentionAdapter
from .factored_two_way_attention import FactoredTwoWayAttentionAdapter
from .factored_qk_norm_rope import FactoredQkNormRopeAdapter
from .packed_stream_qk_norm_rope import PackedStreamQkNormRopeAdapter
from .gemma_attention import GemmaAttentionAdapter
from .transformers_gated_delta import TransformersGatedDeltaAdapter
from .transformers_gated_delta_fused import (
    TransformersGatedDeltaFusedAdapter,
)
from .qwen_per_head_qk_norm_rope import (
    PerHeadGqaQkNormRopeAdapter,
    QwenPerHeadQkNormRopeAdapter,
)
from .packed_qkv_rope import PackedQkvRopeAdapter

register_qk_norm_rope_adapter(PerHeadGqaQkNormRopeAdapter())
register_qk_norm_rope_adapter(FactoredQkNormRopeAdapter())
register_qk_norm_rope_adapter(PackedStreamQkNormRopeAdapter())
register_qkv_rope_adapter(PackedQkvRopeAdapter())
register_attention_adapter(GemmaAttentionAdapter())
register_attention_adapter(FactoredTwoWayAttentionAdapter())
register_attention_adapter(DiffusersRotaryAttentionAdapter())
register_attention_adapter(DiffusersAttentionAdapter())
# the fused-layer form is tried first; it refuses cleanly (missing
# package entries, out-of-profile layers) and the ladder falls
# through to the callable-slot form
register_gated_delta_adapter(TransformersGatedDeltaFusedAdapter())
register_gated_delta_adapter(TransformersGatedDeltaAdapter())

__all__ = [
    "DiffusersAttentionAdapter",
    "DiffusersRotaryAttentionAdapter",
    "GemmaAttentionAdapter",
    "TransformersGatedDeltaAdapter",
    "FactoredTwoWayAttentionAdapter",
    "FactoredQkNormRopeAdapter",
    "PackedStreamQkNormRopeAdapter",
    "PerHeadGqaQkNormRopeAdapter",
    "QwenPerHeadQkNormRopeAdapter",
    "PackedQkvRopeAdapter",
]
