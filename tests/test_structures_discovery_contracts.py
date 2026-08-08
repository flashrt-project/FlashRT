import torch
from torch import nn

from flash_rt.structures.autobuild import auto_swaps, _layer_of, _seam_key
from flash_rt.structures.discover import discover, seam_weights
from flash_rt.structures.swap import attach


class _GatedMlp(nn.Module):
    def __init__(self, *, bias: bool):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=bias)
        self.up_proj = nn.Linear(8, 16, bias=bias)
        self.down_proj = nn.Linear(16, 8, bias=bias)
        self.act_fn = torch.nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x))
                              * self.up_proj(x))


class _Host(nn.Module):
    def __init__(self, *, bias: bool):
        super().__init__()
        self.mlp = _GatedMlp(bias=bias)


class _DualPathAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(512, 512, bias=False)
        self.to_k = nn.Linear(512, 128, bias=False)
        self.to_v = nn.Linear(512, 128, bias=False)
        self.to_out = nn.Linear(512, 512, bias=False)
        self.add_q_proj = nn.Linear(512, 512, bias=False)
        self.add_k_proj = nn.Linear(512, 128, bias=False)
        self.add_v_proj = nn.Linear(512, 128, bias=False)
        self.to_add_out = nn.Linear(512, 512, bias=False)


class _ConditionalNorm(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x, temb=None):
        scale, shift = self.linear(temb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class _DenseGelu(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        first = nn.Module()
        first.proj = nn.Linear(dim, 4 * dim)
        self.net = nn.ModuleList([first, nn.GELU(), nn.Linear(4 * dim, dim)])


class _DiffusionBlock(nn.Module):
    def __init__(self, *, positional=False, cross=False):
        super().__init__()
        self.norm1 = _ConditionalNorm()
        self.norm3 = nn.LayerNorm(512, elementwise_affine=False)
        self.attn1 = nn.Module()
        self.attn1.to_q = nn.Linear(512, 512)
        kv_dim = 768 if cross else 512
        self.attn1.to_k = nn.Linear(kv_dim, 512)
        self.attn1.to_v = nn.Linear(kv_dim, 512)
        self.attn1.to_out = nn.ModuleList([nn.Linear(512, 512)])
        self.ff = _DenseGelu()
        self.pos_embed = nn.Identity() if positional else None


class _ProjectionNamedAttention(nn.Module):
    """Attention slots used by hosts that expose q/k/v/o directly."""

    def __init__(self, dim=512):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x, context=None):
        source = x if context is None else context
        q, k, v = self.q(x), self.k(source), self.v(source)
        if context is not None:
            k = k.mean(dim=1, keepdim=True)
            v = v.mean(dim=1, keepdim=True)
        return self.o(q + k + v)


class _SequentialVideoBlock(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.norm2 = nn.LayerNorm(dim)
        self.self_attn = _ProjectionNamedAttention(dim)
        self.cross_attn = _ProjectionNamedAttention(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x, context):
        x = self.self_attn(x)
        x = self.cross_attn(x, context)
        return self.ffn(self.norm2(x))


def test_decoder_ffn_discovery_accepts_its_bias_free_weight_contract():
    seams = discover(_Host(bias=False), structures=("decoder_ffn",))

    assert [seam.path for seam in seams] == ["mlp"]


def test_decoder_ffn_discovery_refuses_unrepresented_bias_weights():
    seams = discover(_Host(bias=True), structures=("decoder_ffn",))

    assert seams == []


def test_dual_path_attention_discovers_both_independent_qkv_groups():
    seams = discover(
        nn.ModuleDict({"attention": _DualPathAttention()}),
        structures=("qkv_pack",),
    )

    assert [seam.pack_attrs for seam in seams] == [
        ("to_q", "to_k", "to_v"),
        ("add_q_proj", "add_k_proj", "add_v_proj"),
    ]
    assert [_seam_key(seam) for seam in seams] == [
        "attention.to_q",
        "attention.add_q_proj",
    ]


def test_dual_path_attention_discovers_profitable_projections_on_both_paths():
    seams = discover(
        nn.ModuleDict({"attention": _DualPathAttention()}),
        structures=("linear_proj",),
    )

    assert [seam.proj_attr for seam in seams] == [
        "to_q",
        "to_out",
        "add_q_proj",
        "to_add_out",
    ]


def test_modnorm_qkv_chain_discovers_by_direct_dataflow_slots():
    host = nn.ModuleDict({"block": _DiffusionBlock()})

    seams = discover(host, structures=("modnorm_qkv_chain",))

    assert [seam.path for seam in seams] == ["block"]
    assert seams[0].dims == {"D": 512, "C": 512}
    assert seams[0].variant["fanout"] == "qkv"


def test_modnorm_qkv_chain_refuses_an_intervening_positional_module():
    host = nn.ModuleDict({"block": _DiffusionBlock(positional=True)})

    assert discover(host, structures=("modnorm_qkv_chain",)) == []


def test_nested_diffusers_feedforward_is_a_vision_ffn_slice():
    host = nn.ModuleDict({"block": _DiffusionBlock()})

    seams = discover(host, structures=("vision_ffn",))

    assert [seam.path for seam in seams] == ["block.ff"]
    assert seams[0].fc_attrs == ("net.0.proj", "net.2")
    assert seams[0].norm_attr == "norm3"
    assert seams[0].variant["norm_affine"] == "identity"
    weights = seam_weights(host, seams[0])
    assert weights["w_norm"] is None
    assert weights["b_norm"] is None


def test_projection_named_attention_and_sequential_ffn_are_structural_slots():
    host = nn.ModuleDict({"block": _SequentialVideoBlock()})

    packs = discover(host, structures=("qkv_pack",))
    projections = discover(host, structures=("linear_proj",))
    ffns = discover(host, structures=("vision_ffn",))

    assert [seam.path for seam in packs] == [
        "block.self_attn", "block.cross_attn"]
    assert all(seam.pack_attrs == ("q", "k", "v") for seam in packs)
    assert [seam.proj_attr for seam in projections] == [
        "q", "k", "v", "o", "q", "k", "v", "o"]
    assert [seam.path for seam in ffns] == ["block.ffn"]
    assert ffns[0].fc_attrs == ("0", "2")
    assert ffns[0].norm_attr == "norm2"


def test_qkv_pack_qualification_uses_observed_dataflow_not_equal_dimensions():
    torch.manual_seed(4)
    host = nn.ModuleDict({"attention": _ProjectionNamedAttention()}).eval()
    x = torch.randn(1, 3, 512)
    context = torch.randn(1, 5, 512)

    self_plan = auto_swaps(
        host, lambda: host.attention(x),
        structures=("qkv_pack", "linear_proj"), scheme="none")
    cross_plan = auto_swaps(
        host, lambda: host.attention(x, context),
        structures=("qkv_pack", "linear_proj"), scheme="none")

    self_refusals = self_plan.notes.get("refused", [])
    cross_refusals = cross_plan.notes.get("refused", [])
    assert not any("sibling projections did not consume" in reason
                   for _, reason in self_refusals)
    assert any("sibling projections did not consume" in reason
               for _, reason in cross_refusals)


def test_bf16_structural_pack_preserves_self_attention_and_refuses_cross():
    torch.manual_seed(7)
    host = nn.ModuleDict({"attention": _ProjectionNamedAttention()}).eval()
    x = torch.randn(1, 3, 512)
    context = torch.randn(1, 5, 512)
    expected = host.attention(x)

    plan = auto_swaps(
        host, lambda: host.attention(x), structures=("qkv_pack",),
        scheme="bf16_structural")
    handle = attach(host, plan.swaps, on_guard_fail="raise")
    got = host.attention(x)
    handle.raise_on_fallback()
    handle.detach()

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    assert len(plan.swaps) == 3

    cross = auto_swaps(
        host, lambda: host.attention(x, context), structures=("qkv_pack",),
        scheme="bf16_structural")
    assert not cross.swaps
    assert any("same tensor in fixed order" in reason
               for _, reason in cross.notes.get("refused", []))


def test_vision_ffn_refuses_rms_like_one_sided_affine_norm():
    host = nn.ModuleDict({"block": _DiffusionBlock()})
    host.block.norm3.weight = nn.Parameter(torch.ones(512))
    host.block.norm3.bias = None
    refused = []

    seams = discover(host, structures=("vision_ffn",), refused=refused)

    assert seams == []
    assert refused and "one-sided affine" in refused[0][1]


def test_cross_attention_chain_owns_only_the_query_wire():
    host = nn.ModuleDict({"block": _DiffusionBlock(cross=True)})

    chain = discover(host, structures=("modnorm_qkv_chain",))
    packs = discover(host, structures=("qkv_pack",))
    projections = discover(host, structures=("linear_proj",))

    assert chain[0].variant["fanout"] == "q_only"
    assert packs == []
    assert [seam.path for seam in projections] == [
        "block.attn1.to_q",
        "block.attn1.to_k",
        "block.attn1.to_v",
        "block.attn1.to_out.0",
    ]


def test_transformer_block_layer_key_works_at_root_and_nested_paths():
    assert _layer_of("transformer_blocks.1.attn1.to_q") == (
        "transformer_blocks.1")
    assert _layer_of("head.model.transformer_blocks.1.norm1") == (
        "head.model.transformer_blocks.1")
