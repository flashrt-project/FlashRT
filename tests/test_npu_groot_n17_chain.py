"""GR00T N1.7 Ascend action-chain contracts that hold without an Ascend device.

The parts of this backend that a reviewer would otherwise have to take on
trust: that the BF16 weight spec really is the shared checkpoint description
with the FP8 steps removed rather than a second, drifting copy of it; that the
constants the pipeline hardcodes are the ones the shipped checkpoint declares;
that the cross-attention layers alternate the way the reference alternates
them; and that the two setup-time rewrites which make the captured graph small
-- folding the AdaLN modulators into a LayerNorm's affine parameters, and
inverting the reference's keep-mask into the operator's drop-mask -- are exact
rather than approximately right.

Everything here runs on CPU with torch only.
"""
import pytest

torch = pytest.importorskip("torch")

from flash_rt.npu.models.groot_n17 import pipeline as pl
from flash_rt.npu.models.groot_n17 import weights as wt


# ── the derived weight spec ───────────────────────────────────────────

def test_chain_spec_is_the_shared_description_without_the_fp8_steps():
    from flash_rt.executors.torch_weights import Quant, ToBf16, ToFp16
    from flash_rt.frontends.torch._groot_n17_thor_spec import build_spec

    shared = build_spec()
    chain = wt.action_chain_spec()

    shared_blocks = {b.name: b for b in shared.blocks}
    assert [b.name for b in chain.blocks] == ["vl_self_attn", "dit"]
    for block in chain.blocks:
        origin = shared_blocks[block.name]
        assert block.num_layers == origin.num_layers
        assert [i.name for i in block.items] == [i.name for i in origin.items]
        for item in block.items:
            kinds = [type(t) for t in item.transforms]
            assert Quant not in kinds, f"{item.name} still quantizes"
            assert ToFp16 not in kinds, f"{item.name} still casts to FP16"
            assert item.scale_into is None
        assert any(ToBf16 in [type(t) for t in i.transforms] for i in block.items)


def test_chain_spec_leaves_the_backbone_blocks_unloaded():
    """The ViT and the LLM are 3 GB this stage never reads."""
    names = {b.name for b in wt.action_chain_spec().blocks}
    assert "qwen3vl_vit" not in names and "qwen3vl_llm" not in names


def test_chain_spec_keeps_every_action_head_singleton():
    chain = wt.action_chain_spec()
    names = {item.name for item in chain.singletons}
    for required in ("vlln_w", "vlln_b", "ah_pos_embed", "ts_lin1_w", "ts_lin2_w",
                     "proj_out_1_w", "proj_out_2_w", "st_enc_l1_W", "st_enc_l2_W",
                     "ac_enc_W1_W", "ac_enc_W2_W", "ac_enc_W3_W",
                     "ac_dec_l1_W", "ac_dec_l2_W"):
        assert required in names
    assert not any(n.startswith(("dsm", "merger", "patch_embed", "llm_", "embed_tokens"))
                   for n in names)


def test_missing_checkpoint_directory_is_named():
    with pytest.raises(FileNotFoundError, match="safetensors"):
        wt.shard_paths("a-directory-that-does-not-exist")


# ── the geometry the pipeline hardcodes ───────────────────────────────

def test_geometry_matches_the_published_configuration():
    assert pl.DIT_DIM == pl.DIT_HEADS * pl.DIT_HEAD_DIM == 1536
    assert pl.VLSA_DIM == pl.VLSA_HEADS * pl.VLSA_HEAD_DIM == 2048
    assert (pl.DIT_LAYERS, pl.VLSA_LAYERS) == (32, 4)
    assert (pl.DIT_FF_DIM, pl.VLSA_FF_DIM) == (4 * pl.DIT_DIM, 4 * pl.VLSA_DIM)
    assert (pl.NUM_STEPS, pl.ACTION_HORIZON, pl.ACTION_DIM) == (4, 40, 132)
    assert pl.DIT_OUT_DIM == 1024 and pl.BACKBONE_DIM == 2048


# ── the alternation the reference performs ────────────────────────────

def _roles():
    return [(index % 2 == 0,
             index % (2 * pl.ATTEND_TEXT_EVERY_N_BLOCKS) == 0)
            for index in range(pl.DIT_LAYERS)]


def test_even_layers_cross_attend_and_odd_layers_self_attend():
    roles = _roles()
    assert [cross for cross, _ in roles[:4]] == [True, False, True, False]
    assert sum(cross for cross, _ in roles) == pl.DIT_LAYERS // 2


def test_cross_layers_alternate_between_text_and_image_keys():
    cross = [(index, text) for index, (is_cross, text) in enumerate(_roles()) if is_cross]
    assert [text for _, text in cross[:4]] == [True, False, True, False]
    assert sum(text for _, text in cross) == len(cross) // 2


# ── the two setup-time rewrites, checked as identities ────────────────

def test_folding_the_modulators_into_the_norm_affine_is_exact():
    """``norm(x) * (1 + scale) + shift`` over an affine-free LayerNorm."""
    torch.manual_seed(0)
    x = torch.randn(1, 41, pl.DIT_DIM, dtype=torch.float32)
    scale = torch.randn(pl.DIT_DIM) * 0.1
    shift = torch.randn(pl.DIT_DIM) * 0.1
    reference = (torch.nn.functional.layer_norm(x, (pl.DIT_DIM,), eps=pl.EPS)
                 * (1 + scale) + shift)
    folded = torch.nn.functional.layer_norm(x, (pl.DIT_DIM,), 1 + scale, shift, pl.EPS)
    assert torch.allclose(reference, folded, atol=1e-6)


def test_key_masks_invert_the_reference_and_materialise_the_query_axis():
    image = torch.tensor([[True, True, False, False, False]])
    attend = torch.tensor([[True, True, True, True, False]])
    text_mask, image_mask = pl.attention_masks(image.reshape(-1), attend.reshape(-1), 3)
    # The operator masks out where the entry is True: the inverse of the
    # reference's "attend here" mask, with the query axis written out.
    assert text_mask.shape == image_mask.shape == (3, 5)
    assert text_mask[0].tolist() == [True, True, False, False, True]
    assert image_mask[0].tolist() == [False, False, True, True, True]
    assert torch.equal(text_mask[0], text_mask[2])
    assert text_mask.is_contiguous() and image_mask.is_contiguous()


def test_every_key_is_masked_out_in_exactly_one_of_the_two_masks():
    """Together the two masks partition the attended tokens, as the
    reference's ``image & attn`` / ``~image & attn`` pair does."""
    torch.manual_seed(1)
    image = torch.rand(461) > 0.5
    attend = torch.rand(461) > 0.05
    text_mask, image_mask = pl.attention_masks(image, attend, 41)
    kept = (~text_mask[0]).int() + (~image_mask[0]).int()
    assert torch.equal(kept.bool(), attend)
    assert kept.max().item() == 1


# ── the step-constant tables ──────────────────────────────────────────

def test_timestep_tables_use_the_discretised_reference_timesteps():
    proj = pl._timestep_projection(4)
    tau = pl._tau_encoding(4, pl.DIT_DIM)
    assert proj.shape == (4, 256) and tau.shape == (4, pl.DIT_DIM)
    # Step 0 discretises to timestep 0: cosine half all ones, sine half zero.
    assert torch.allclose(proj[0, :128], torch.ones(128))
    assert torch.allclose(proj[0, 128:], torch.zeros(128))
    assert torch.allclose(tau[0, :pl.DIT_DIM // 2], torch.zeros(pl.DIT_DIM // 2))
    assert not torch.allclose(proj[1], proj[2])


def test_timestep_projection_matches_the_reference_embedding():
    diffusers = pytest.importorskip("diffusers")
    from diffusers.models.embeddings import Timesteps

    reference = Timesteps(num_channels=256, flip_sin_to_cos=True,
                          downscale_freq_shift=1)
    steps = torch.tensor([int(s / 4 * pl.TIMESTEP_BUCKETS) for s in range(4)])
    assert torch.allclose(reference(steps), pl._timestep_projection(4), atol=1e-6)
