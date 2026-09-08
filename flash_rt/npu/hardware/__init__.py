"""Ascend NPU attention backend protocol implementations.

Attention on 910/A2 is served through torch_npu ops (``F.scaled_dot_product_attention``
via aclnn, or the ``npu_prompt_flash_attention``-family kernels where a
part supports them). The backend object follows the repo's
``hardware/backend.py`` ``AttentionBackend`` protocol. Part-specific
behaviour lives in files named with the part family
(``attn_backend_910b.py`` …); common behaviour stays here.
"""
