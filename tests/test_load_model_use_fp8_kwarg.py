import ast
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock, patch

import pytest


def test_predict_forwards_state_to_prompt_and_observation():
    from flash_rt.api import VLAModel

    image0 = object()
    image1 = object()
    state = object()
    actions = object()

    class StateFrontend:
        prompt_state = None
        seen_obs = None

        def set_prompt(self, prompt, state=None):
            type(self).prompt_state = state

        def infer(self, obs):
            type(self).seen_obs = obs
            return {"actions": actions}

    model = VLAModel(StateFrontend(), framework="torch")
    result = model.predict(
        images=[image0, image1],
        prompt="pick up the red block",
        state=state,
    )

    assert result is actions
    assert StateFrontend.prompt_state is state
    assert StateFrontend.seen_obs["state"] is state
    assert StateFrontend.seen_obs["image"] is image0
    assert StateFrontend.seen_obs["wrist_image"] is image1


def test_predict_refreshes_prompt_when_prompt_state_changes():
    from flash_rt.api import VLAModel

    image = object()
    state0 = [0.0, 1.0]
    state1 = [1.0, 2.0]

    class TokenStateFrontend:
        prompt_states = []

        def set_prompt(self, prompt, state=None):
            type(self).prompt_states.append(list(state))

        def infer(self, obs):
            return {"actions": None}

    TokenStateFrontend.prompt_states = []
    model = VLAModel(TokenStateFrontend(), framework="torch")
    model.predict(images=[image], prompt="pick", state=state0)
    model.predict(images=[image], state=state0)
    model.predict(images=[image], state=state1)

    assert TokenStateFrontend.prompt_states == [state0, state1]


def test_predict_refreshes_prompt_when_prompt_state_is_removed():
    from flash_rt.api import VLAModel

    image = object()
    state0 = [0.0, 1.0]

    class TokenStateFrontend:
        prompt_states = []

        def set_prompt(self, prompt, state=None):
            type(self).prompt_states.append(
                None if state is None else list(state))

        def infer(self, obs):
            return {"actions": None}

    TokenStateFrontend.prompt_states = []
    model = VLAModel(TokenStateFrontend(), framework="torch")
    model.predict(images=[image], prompt="pick", state=state0)
    model.predict(images=[image], state=None)

    assert TokenStateFrontend.prompt_states == [state0, None]


def test_manual_set_prompt_tracks_prompt_state():
    from flash_rt.api import VLAModel

    image = object()
    state0 = [0.0, 1.0]

    class TokenStateFrontend:
        prompt_states = []

        def set_prompt(self, prompt, state=None):
            type(self).prompt_states.append(
                None if state is None else list(state))

        def infer(self, obs):
            return {"actions": None}

    TokenStateFrontend.prompt_states = []
    model = VLAModel(TokenStateFrontend(), framework="torch")
    model.set_prompt("pick", state=state0)
    model.predict(images=[image], state=None)

    assert TokenStateFrontend.prompt_states == [state0, None]


def test_predict_preserves_state_from_observation_dict():
    from flash_rt.api import VLAModel

    image = object()
    dict_state = object()
    kwarg_state = object()

    class ObservationFrontend:
        seen_obs = None

        def set_prompt(self, prompt):
            return None

        def infer(self, obs):
            type(self).seen_obs = obs
            return {"actions": None}

    model = VLAModel(ObservationFrontend(), framework="torch")
    model.predict(
        images={"image": image, "state": dict_state},
        prompt="pick up the red block",
        state=kwarg_state,
    )

    assert ObservationFrontend.seen_obs["state"] is dict_state
    assert ObservationFrontend.seen_obs["image"] is image


def test_load_model_only_passes_use_fp8_when_frontend_accepts_it():
    from flash_rt.api import load_model

    class NoUseFp8Frontend:
        def __init__(self, checkpoint, num_views=2):
            self.checkpoint = checkpoint
            self.num_views = num_views

        def infer(self, obs):
            return {"actions": None}

    with patch("flash_rt.hardware.detect_arch", return_value="rtx_sm120"), \
            patch("flash_rt.hardware.resolve_pipeline_class",
                  return_value=NoUseFp8Frontend):
        model = load_model(
            "/tmp/nonexistent", config="pi05", framework="torch",
            use_fp8=False)

    assert isinstance(model._pipe, NoUseFp8Frontend)


def test_load_model_propagates_use_fp8_when_frontend_accepts_it():
    from flash_rt.api import load_model

    class UseFp8Frontend:
        seen_use_fp8 = None

        def __init__(self, checkpoint, num_views=2, use_fp8=True):
            type(self).seen_use_fp8 = use_fp8

        def infer(self, obs):
            return {"actions": None}

    with patch("flash_rt.hardware.detect_arch", return_value="rtx_sm120"), \
            patch("flash_rt.hardware.resolve_pipeline_class",
                  return_value=UseFp8Frontend):
        model = load_model(
            "/tmp/nonexistent", config="pi05", framework="torch",
            use_fp8=False)

    assert isinstance(model._pipe, UseFp8Frontend)
    assert UseFp8Frontend.seen_use_fp8 is False


def test_load_model_propagates_hardware_when_frontend_accepts_it():
    from flash_rt.api import load_model

    class HardwareFrontend:
        seen_hardware = None

        def __init__(self, checkpoint, num_views=2, hardware=None):
            type(self).seen_hardware = hardware

        def infer(self, obs):
            return {"actions": None}

    with patch("flash_rt.hardware.resolve_pipeline_class",
              return_value=HardwareFrontend):
        model = load_model(
            "/tmp/nonexistent", config="pi05", framework="torch",
            hardware="rtx_sm89")

    assert isinstance(model._pipe, HardwareFrontend)
    assert HardwareFrontend.seen_hardware == "rtx_sm89"


def test_load_model_propagates_pi05_orin_tuning_kwargs_when_supported():
    from flash_rt.api import load_model

    class OrinTuningFrontend:
        seen = None

        def __init__(self, checkpoint, num_views=2, num_steps=10,
                     vision_pool_factor=1, vision_num_layers=27,
                     cache_frames=1):
            type(self).seen = {
                "num_steps": num_steps,
                "vision_pool_factor": vision_pool_factor,
                "vision_num_layers": vision_num_layers,
                "cache_frames": cache_frames,
            }

        def infer(self, obs):
            return {"actions": None}

    with patch("flash_rt.hardware.resolve_pipeline_class",
              return_value=OrinTuningFrontend):
        model = load_model(
            "/tmp/nonexistent", config="pi05", framework="torch",
            hardware="rtx_sm87", num_steps=5, vision_pool_factor=2,
            vision_num_layers=18, cache_frames=2)

    assert isinstance(model._pipe, OrinTuningFrontend)
    assert OrinTuningFrontend.seen == {
        "num_steps": 5,
        "vision_pool_factor": 2,
        "vision_num_layers": 18,
        "cache_frames": 2,
    }


def test_load_model_routes_pi05_jax_thor_fp4_and_preset_kwargs(monkeypatch):
    from flash_rt.api import load_model

    class ResolvedFrontend:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "load_model() should rewrite Pi0.5 JAX Thor FP4 requests "
                "to Pi05JaxFrontendThorFP4")

    class Pi05JaxFrontendThorFP4:
        seen = None

        def __init__(self, checkpoint, *, num_views=2, autotune=3,
                     weight_cache=True, use_fp8=True,
                     use_fp4_encoder_ffn=False, fp4_layers=(),
                     use_awq=False, awq_alpha=0.5,
                     use_p1_split_gu=False):
            type(self).seen = {
                "checkpoint": checkpoint,
                "num_views": num_views,
                "autotune": autotune,
                "weight_cache": weight_cache,
                "use_fp8": use_fp8,
                "use_fp4_encoder_ffn": use_fp4_encoder_ffn,
                "fp4_layers": fp4_layers,
                "use_awq": use_awq,
                "awq_alpha": awq_alpha,
                "use_p1_split_gu": use_p1_split_gu,
            }

        def infer(self, obs):
            return {"actions": None}

    fp4_ext = types.ModuleType("flash_rt.flash_rt_fp4")
    fp4_ext.has_nvfp4 = lambda: True
    jax_fp4_mod = types.ModuleType("flash_rt.frontends.jax.pi05_thor_fp4")
    jax_fp4_mod.Pi05JaxFrontendThorFP4 = Pi05JaxFrontendThorFP4
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fp4", fp4_ext)
    monkeypatch.setitem(
        sys.modules, "flash_rt.frontends.jax.pi05_thor_fp4", jax_fp4_mod)

    with patch("flash_rt.hardware.resolve_pipeline_class",
               return_value=ResolvedFrontend):
        model = load_model(
            "unused-orbax-checkpoint",
            config="pi05",
            framework="jax",
            hardware="thor",
            num_views=3,
            autotune=0,
            use_fp4=True,
        )

    assert isinstance(model._pipe, Pi05JaxFrontendThorFP4)
    assert Pi05JaxFrontendThorFP4.seen == {
        "checkpoint": "unused-orbax-checkpoint",
        "num_views": 3,
        "autotune": 0,
        "weight_cache": True,
        "use_fp8": True,
        "use_fp4_encoder_ffn": True,
        "fp4_layers": tuple(range(17)),
        "use_awq": True,
        "awq_alpha": 0.5,
        "use_p1_split_gu": True,
    }


def test_load_model_routes_pi05_torch_thor_decoder_fp4(monkeypatch):
    from flash_rt.api import load_model

    class ResolvedFrontend:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "decoder FP4 request must route to Pi05TorchFrontendThorFP4")

    class Pi05TorchFrontendThorFP4:
        seen = None

        def __init__(self, checkpoint, *, num_views=2, autotune=3,
                     use_fp8=True, use_fp4_encoder_ffn=False,
                     use_fp4_decoder=False, fp4_layers=(), use_awq=False,
                     awq_alpha=0.5, use_p1_split_gu=False,
                     encoder_p1_combiner="direct", encoder_down_variant=7,
                     decoder_gate_up_variant=10,
                     state_prompt_mode="exact"):
            type(self).seen = {
                "checkpoint": checkpoint,
                "num_views": num_views,
                "autotune": autotune,
                "use_fp8": use_fp8,
                "use_fp4_encoder_ffn": use_fp4_encoder_ffn,
                "use_fp4_decoder": use_fp4_decoder,
                "fp4_layers": fp4_layers,
                "use_awq": use_awq,
                "awq_alpha": awq_alpha,
                "use_p1_split_gu": use_p1_split_gu,
                "encoder_p1_combiner": encoder_p1_combiner,
                "encoder_down_variant": encoder_down_variant,
                "decoder_gate_up_variant": decoder_gate_up_variant,
                "state_prompt_mode": state_prompt_mode,
            }

        def infer(self, obs):
            return {"actions": None}

    fp4_ext = types.ModuleType("flash_rt.flash_rt_fp4")
    fp4_ext.has_nvfp4 = lambda: True
    fp4_mod = types.ModuleType("flash_rt.frontends.torch.pi05_thor_fp4")
    fp4_mod.Pi05TorchFrontendThorFP4 = Pi05TorchFrontendThorFP4
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fp4", fp4_ext)
    monkeypatch.setitem(
        sys.modules, "flash_rt.frontends.torch.pi05_thor_fp4", fp4_mod)

    with patch("flash_rt.hardware.resolve_pipeline_class",
               return_value=ResolvedFrontend):
        model = load_model(
            "unused-checkpoint",
            config="pi05",
            framework="torch",
            hardware="thor",
            num_views=2,
            autotune=0,
            use_fp4=True,
            use_fp4_decoder=True,
            fp4_layers=(),
            use_awq=False,
            use_p1_split_gu=False,
        )

    assert isinstance(model._pipe, Pi05TorchFrontendThorFP4)
    assert Pi05TorchFrontendThorFP4.seen == {
        "checkpoint": "unused-checkpoint",
        "num_views": 2,
        "autotune": 0,
        "use_fp8": True,
        "use_fp4_encoder_ffn": True,
        "use_fp4_decoder": True,
        "fp4_layers": (),
        "use_awq": False,
        "awq_alpha": 0.8,
        "use_p1_split_gu": False,
        # use_fp4_decoder=True selects the measured Thor NVFP4 tier, whose
        # encoder combiner is the fused GeGLU epilogue. use_fp4=True alone
        # still resolves to "lut_native".
        "encoder_p1_combiner": "epilogue_hw",
        "encoder_down_variant": 7,
        "decoder_gate_up_variant": 10,
        "state_prompt_mode": "exact",
    }


@pytest.mark.parametrize(
    "framework, hardware, module_name, class_name",
    [
        (
            "jax",
            "rtx_sm89",
            "flash_rt.frontends.jax.pi05_thor_fp4",
            "Pi05JaxFrontendThorFP4",
        ),
        (
            "torch",
            "rtx_sm120",
            "flash_rt.frontends.torch.pi05_thor_fp4",
            "Pi05TorchFrontendThorFP4",
        ),
    ],
)
def test_load_model_does_not_route_pi05_non_thor_fp4_to_thor_frontend(
        monkeypatch, framework, hardware, module_name, class_name):
    from flash_rt.api import load_model

    class ResolvedFrontend:
        seen = None

        def __init__(self, checkpoint, *, num_views=2, use_fp8=True):
            type(self).seen = {
                "checkpoint": checkpoint,
                "num_views": num_views,
                "use_fp8": use_fp8,
            }

        def infer(self, obs):
            return {"actions": None}

    class UnexpectedThorFP4Frontend:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "non-Thor Pi0.5 use_fp4=True must keep the resolved "
                "hardware frontend instead of rewriting to a Thor FP4 class")

    fp4_ext = types.ModuleType("flash_rt.flash_rt_fp4")
    fp4_ext.has_nvfp4 = lambda: True
    thor_fp4_mod = types.ModuleType(module_name)
    setattr(thor_fp4_mod, class_name, UnexpectedThorFP4Frontend)
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fp4", fp4_ext)
    monkeypatch.setitem(sys.modules, module_name, thor_fp4_mod)

    with patch("flash_rt.hardware.resolve_pipeline_class",
               return_value=ResolvedFrontend):
        model = load_model(
            "unused-checkpoint",
            config="pi05",
            framework=framework,
            hardware=hardware,
            num_views=3,
            use_fp4=True,
        )

    assert isinstance(model._pipe, ResolvedFrontend)
    assert ResolvedFrontend.seen == {
        "checkpoint": "unused-checkpoint",
        "num_views": 3,
        "use_fp8": True,
    }


def test_sm87_rejects_unvalidated_pi0_and_jax_backends():
    from flash_rt.hardware import resolve_pipeline_class

    for config, framework in [
        ("pi05", "jax"),
        ("pi0", "torch"),
        ("pi0", "jax"),
    ]:
        with pytest.raises(RuntimeError, match="Jetson Orin SM87"):
            resolve_pipeline_class(config, framework, "rtx_sm87")


def test_groot_n17_rtx_sm120_is_registered():
    from flash_rt.hardware import resolve_pipeline_class

    cls = resolve_pipeline_class("groot_n17", "torch", "rtx_sm120")
    assert cls.__name__ == "GrootN17TorchFrontendRtx"


def test_groot_n17_rtx_sm89_is_registered():
    from flash_rt.hardware import resolve_pipeline_class

    cls = resolve_pipeline_class("groot_n17", "torch", "rtx_sm89")
    assert cls.__name__ == "GrootN17TorchFrontendRtxSm89"


def test_groot_n17_sm120_uses_sm120_safe_dit_fp8_only_on_sm120_frontend():
    from flash_rt.frontends.torch.groot_n17_rtx_fp8 import (
        GrootN17TorchFrontendRtxFP8,
    )
    from flash_rt.frontends.torch.groot_n17_rtx_sm89 import (
        GrootN17TorchFrontendRtxSm89,
    )
    from flash_rt.frontends.torch.groot_n17_thor_fp8 import (
        GrootN17TorchFrontendThorFP8,
    )

    assert GrootN17TorchFrontendRtxFP8._DIT_FP8_IMPL == "sm120_safe"
    assert GrootN17TorchFrontendRtxSm89._DIT_FP8_IMPL == "thor_epilogue"
    assert GrootN17TorchFrontendThorFP8._DIT_FP8_IMPL == "thor_epilogue"


def test_wan22_ti2v_5b_rtx_sm120_is_registered():
    from flash_rt.hardware import resolve_pipeline_class

    cls = resolve_pipeline_class("wan22_ti2v_5b", "torch", "rtx_sm120")
    assert cls.__name__ == "Wan22TorchFrontendRtx"


def test_wan22_ti2v_5b_sm89_is_not_registered_without_validation():
    from flash_rt.hardware import resolve_pipeline_class

    with pytest.raises(RuntimeError, match="rtx_sm120"):
        resolve_pipeline_class("wan22_ti2v_5b", "torch", "rtx_sm89")


def test_load_model_accepts_wan22_ti2v_5b_config():
    from flash_rt.api import load_model

    class Wan22Frontend:
        seen = None

        def __init__(self, checkpoint, num_views=1, autotune=3):
            type(self).seen = {
                "checkpoint": checkpoint,
                "num_views": num_views,
                "autotune": autotune,
            }

        def set_prompt(self, *args, **kwargs):
            return None

        def infer(self, *args, **kwargs):
            return None

    with patch("flash_rt.hardware.resolve_pipeline_class",
              return_value=Wan22Frontend):
        model = load_model(
            "unused-checkpoint",
            config="wan22_ti2v_5b",
            framework="torch",
            hardware="rtx_sm120",
            num_views=1,
            autotune=0,
        )

    assert isinstance(model._pipe, Wan22Frontend)
    assert Wan22Frontend.seen == {
        "checkpoint": "unused-checkpoint",
        "num_views": 1,
        "autotune": 0,
    }


def test_load_model_accepts_cosmos3_edge_thor_config():
    from flash_rt.api import load_model

    class Cosmos3EdgeFrontend:
        seen = None

        def __init__(self, checkpoint, num_views=1, hardware=None):
            type(self).seen = {
                "checkpoint": checkpoint,
                "num_views": num_views,
                "hardware": hardware,
            }

        def set_prompt(self, *args, **kwargs):
            return None

        def infer(self, *args, **kwargs):
            return None

    with patch("flash_rt.hardware.resolve_pipeline_class",
              return_value=Cosmos3EdgeFrontend):
        model = load_model(
            "unused-checkpoint",
            config="cosmos3_edge",
            framework="torch",
            hardware="thor",
            num_views=1,
        )

    assert isinstance(model._pipe, Cosmos3EdgeFrontend)
    assert Cosmos3EdgeFrontend.seen == {
        "checkpoint": "unused-checkpoint",
        "num_views": 1,
        "hardware": "thor",
    }


def test_cosmos3_edge_infer_exposes_torch_reference_backend_args():
    import inspect

    from flash_rt.frontends.torch.cosmos3_edge_thor import (
        Cosmos3EdgeTorchFrontendThor,
    )

    sig = inspect.signature(Cosmos3EdgeTorchFrontendThor.infer)
    assert "reference_dump" in sig.parameters
    assert "boundary_dump" in sig.parameters
    assert "live_dump_out" in sig.parameters
    assert "live_flashrt_handoff" in sig.parameters
    assert "live_boundary_out" in sig.parameters
    assert "live_boundary_in" in sig.parameters
    assert "live_boundary_prepare_in" in sig.parameters
    assert "live_boundary_prepare_live" in sig.parameters
    assert "live_handoff_trace_out" in sig.parameters
    assert "upstream_trace_out" in sig.parameters
    assert "vae_encode_dump_out" in sig.parameters
    assert "vae_latent_in" in sig.parameters
    assert "vae_encode_dump_input" in sig.parameters
    assert "vae_encode_profile_out" in sig.parameters
    assert "vae_native_rms_silu" in sig.parameters
    assert "vae_t1_conv2d" in sig.parameters
    assert "vae_native_avgdown3d" in sig.parameters
    assert "vae_channels_last3d_conv320" in sig.parameters
    assert "vae_compile_encode" in sig.parameters
    assert "vae_compile_trace_out" in sig.parameters
    assert "prepare_dump_out" in sig.parameters
    assert "prepare_replay_in" in sig.parameters
    assert "prepare_inventory_out" in sig.parameters
    assert "prepare_slim_no_raw_state_vision" in sig.parameters
    assert "prepare_slim_derive_condition_reference" in sig.parameters
    assert "prepare_slim_derive_initial_noise" in sig.parameters
    assert "live_prelayer_bootstrap" in sig.parameters
    assert "live_warm_request" in sig.parameters
    assert "cache_warmup_vae" in sig.parameters
    assert "cache_warmup_prepare" in sig.parameters
    assert "warmup" in sig.parameters

    pipe = Cosmos3EdgeTorchFrontendThor("unused-checkpoint", hardware="thor")
    with pytest.raises(ValueError, match="official.*official_action_only.*replay.*torch_ref.*flashrt"):
        pipe.infer(output_dir="/tmp/unused", backend="bad")
    with pytest.raises(ValueError, match="warmup must be non-negative"):
        pipe.infer(output_dir="/tmp/unused", backend="official_action_only", warmup=-1)
    with pytest.raises(ValueError, match="backend='flashrt' requires reference_dump"):
        pipe.infer(output_dir="/tmp/unused", backend="flashrt")
    with pytest.raises(ValueError, match="live_dump_out.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", live_dump_out="/tmp/live.safetensors")
    with pytest.raises(ValueError, match="upstream_trace_out.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", upstream_trace_out="/tmp/upstream.json")
    with pytest.raises(ValueError, match="VAE/prepare boundary.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", vae_encode_dump_out="/tmp/vae.safetensors")
    with pytest.raises(ValueError, match="VAE/prepare boundary/profile.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", vae_encode_profile_out="/tmp/vae_profile.json")
    with pytest.raises(ValueError, match="VAE/prepare boundary/profile.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", vae_native_rms_silu=True)
    with pytest.raises(ValueError, match="VAE/prepare boundary/profile.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", vae_t1_conv2d=True)
    with pytest.raises(ValueError, match="VAE/prepare boundary/profile.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", vae_native_avgdown3d=True)
    with pytest.raises(ValueError, match="VAE/prepare boundary/profile.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", vae_channels_last3d_conv320=True)
    with pytest.raises(ValueError, match="VAE/prepare boundary/profile.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", vae_compile_encode=True)
    with pytest.raises(ValueError, match="VAE/prepare boundary/profile.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", prepare_dump_out="/tmp/prepare.pt")
    with pytest.raises(ValueError, match="VAE/prepare boundary/profile/inventory.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", prepare_inventory_out="/tmp/prepare.json")
    with pytest.raises(ValueError, match="VAE/prepare boundary/profile/inventory.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", prepare_slim_no_raw_state_vision=True)
    with pytest.raises(ValueError, match="VAE/prepare boundary/profile/inventory.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", prepare_slim_derive_initial_noise=True)
    with pytest.raises(ValueError, match="vae_compile_encode cannot be combined"):
        pipe.infer(
            output_dir="/tmp/unused",
            backend="official_action_only",
            vae_compile_encode=True,
            vae_native_rms_silu=True,
        )
    with pytest.raises(ValueError, match="vae_compile_encode cannot be combined"):
        pipe.infer(
            output_dir="/tmp/unused",
            backend="official_action_only",
            vae_compile_encode=True,
            vae_native_avgdown3d=True,
        )
    with pytest.raises(ValueError, match="cache_warmup_vae cannot be combined"):
        pipe.infer(
            output_dir="/tmp/unused",
            backend="official_action_only",
            cache_warmup_vae=True,
            vae_latent_in="/tmp/vae.safetensors",
        )
    with pytest.raises(ValueError, match="live FlashRT handoff.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", live_flashrt_handoff=True)
    with pytest.raises(ValueError, match="live FlashRT handoff.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", live_boundary_in="/tmp/boundary.safetensors")
    with pytest.raises(ValueError, match="live FlashRT handoff.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", live_boundary_prepare_live=True)
    with pytest.raises(ValueError, match="live_warm_request.*official_action_only"):
        pipe.infer(output_dir="/tmp/unused", backend="official", live_warm_request=True)
    with pytest.raises(ValueError, match="live_dump_out cannot be combined"):
        pipe.infer(
            output_dir="/tmp/unused",
            backend="official_action_only",
            live_dump_out="/tmp/live.safetensors",
            live_flashrt_handoff=True,
        )
    with pytest.raises(ValueError, match="live_boundary_prepare_live cannot be combined"):
        pipe.infer(
            output_dir="/tmp/unused",
            backend="official_action_only",
            live_boundary_in="/tmp/boundary.safetensors",
            live_boundary_prepare_live=True,
        )


def test_cosmos3_edge_live_warm_request_builds_official_warmup_cmd(tmp_path, monkeypatch):
    from flash_rt.frontends.torch import cosmos3_edge_thor
    from flash_rt.frontends.torch.cosmos3_edge_thor import (
        Cosmos3EdgeTorchFrontendThor,
    )

    input_json = tmp_path / "input.json"
    input_json.write_text('{"samples":[]}\n', encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cosmos3_edge_thor.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    pipe = Cosmos3EdgeTorchFrontendThor("unused-checkpoint", hardware="thor")
    pipe.set_prompt(input_json=str(input_json))
    result = pipe.infer(
        output_dir=str(tmp_path / "out"),
        backend="official_action_only",
        cosmos_root=str(tmp_path),
        config_file=str(config_file),
        live_warm_request=True,
        live_handoff_trace_out="trace.json",
        upstream_trace_out="upstream.json",
        cache_warmup_prepare=True,
        prepare_slim_no_raw_state_vision=True,
        prepare_slim_derive_initial_noise=True,
    )

    cmd = seen["cmd"]
    assert cmd[2] == "flash_rt.models.cosmos3_edge.action_only_official"
    assert "--flashrt-live-prelayer-bootstrap" in cmd
    assert cmd[cmd.index("--warmup") + 1] == "1"
    assert cmd[cmd.index("--flashrt-live-handoff-trace-out") + 1] == str(tmp_path / "trace.json")
    assert cmd[cmd.index("--flashrt-upstream-trace-out") + 1] == str(tmp_path / "upstream.json")
    assert "--flashrt-cache-warmup-vae" in cmd
    assert "--flashrt-cache-warmup-prepare" in cmd
    assert "--flashrt-prepare-slim-no-raw-state-vision" in cmd
    assert "--flashrt-prepare-slim-derive-initial-noise" in cmd
    assert result["live_warm_request"] is True
    assert result["live_prelayer_bootstrap"] is True
    assert result["cache_warmup_vae"] is True
    assert result["cache_warmup_prepare"] is True
    assert result["prepare_slim_no_raw_state_vision"] is True
    assert result["prepare_slim_derive_initial_noise"] is True
    assert result["warmup"] == 1
    assert result["live_handoff_trace_out"] == str(tmp_path / "trace.json")
    assert result["upstream_trace_out"] == str(tmp_path / "upstream.json")


def test_cosmos3_edge_live_boundary_in_builds_official_cmd(tmp_path, monkeypatch):
    from flash_rt.frontends.torch import cosmos3_edge_thor
    from flash_rt.frontends.torch.cosmos3_edge_thor import (
        Cosmos3EdgeTorchFrontendThor,
    )

    input_json = tmp_path / "input.json"
    input_json.write_text('{"samples":[]}\n', encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cosmos3_edge_thor.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    pipe = Cosmos3EdgeTorchFrontendThor("unused-checkpoint", hardware="thor")
    pipe.set_prompt(input_json=str(input_json))
    result = pipe.infer(
        output_dir=str(tmp_path / "out"),
        backend="official_action_only",
        cosmos_root=str(tmp_path),
        config_file=str(config_file),
        live_boundary_in="prelayer_boundary.safetensors",
        live_handoff_trace_out="boundary_in_trace.json",
    )

    cmd = seen["cmd"]
    assert cmd[2] == "flash_rt.models.cosmos3_edge.action_only_official"
    assert cmd[cmd.index("--flashrt-live-boundary-in") + 1] == str(tmp_path / "prelayer_boundary.safetensors")
    assert cmd[cmd.index("--flashrt-live-handoff-trace-out") + 1] == str(tmp_path / "boundary_in_trace.json")
    assert "--flashrt-live-prelayer-bootstrap" not in cmd
    assert result["live_boundary_in"] == str(tmp_path / "prelayer_boundary.safetensors")
    assert result["live_handoff_trace_out"] == str(tmp_path / "boundary_in_trace.json")


def test_cosmos3_edge_live_boundary_prepare_in_builds_official_cmd(tmp_path, monkeypatch):
    from flash_rt.frontends.torch import cosmos3_edge_thor
    from flash_rt.frontends.torch.cosmos3_edge_thor import (
        Cosmos3EdgeTorchFrontendThor,
    )

    input_json = tmp_path / "input.json"
    input_json.write_text('{"samples":[]}\n', encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cosmos3_edge_thor.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    pipe = Cosmos3EdgeTorchFrontendThor("unused-checkpoint", hardware="thor")
    pipe.set_prompt(input_json=str(input_json))
    result = pipe.infer(
        output_dir=str(tmp_path / "out"),
        backend="official_action_only",
        cosmos_root=str(tmp_path),
        config_file=str(config_file),
        live_boundary_prepare_in="slim_prepare.pt",
        live_boundary_out="derived_boundary.safetensors",
        live_handoff_trace_out="prepare_boundary_trace.json",
    )

    cmd = seen["cmd"]
    assert cmd[2] == "flash_rt.models.cosmos3_edge.action_only_official"
    assert cmd[cmd.index("--flashrt-live-boundary-prepare-in") + 1] == str(tmp_path / "slim_prepare.pt")
    assert cmd[cmd.index("--flashrt-live-boundary-out") + 1] == str(tmp_path / "derived_boundary.safetensors")
    assert cmd[cmd.index("--flashrt-live-handoff-trace-out") + 1] == str(tmp_path / "prepare_boundary_trace.json")
    assert "--flashrt-live-boundary-in" not in cmd
    assert result["live_boundary_prepare_in"] == str(tmp_path / "slim_prepare.pt")
    assert result["live_boundary_out"] == str(tmp_path / "derived_boundary.safetensors")
    assert result["live_handoff_trace_out"] == str(tmp_path / "prepare_boundary_trace.json")


def test_cosmos3_edge_live_boundary_prepare_live_builds_official_cmd(tmp_path, monkeypatch):
    from flash_rt.frontends.torch import cosmos3_edge_thor
    from flash_rt.frontends.torch.cosmos3_edge_thor import (
        Cosmos3EdgeTorchFrontendThor,
    )

    input_json = tmp_path / "input.json"
    input_json.write_text('{"samples":[]}\n', encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cosmos3_edge_thor.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    pipe = Cosmos3EdgeTorchFrontendThor("unused-checkpoint", hardware="thor")
    pipe.set_prompt(input_json=str(input_json))
    result = pipe.infer(
        output_dir=str(tmp_path / "out"),
        backend="official_action_only",
        cosmos_root=str(tmp_path),
        config_file=str(config_file),
        live_boundary_prepare_live=True,
        live_boundary_out="derived_boundary.safetensors",
        live_handoff_trace_out="prepare_live_trace.json",
        upstream_trace_out="prepare_live_upstream.json",
        prepare_slim_no_raw_state_vision=True,
        prepare_slim_derive_condition_reference=True,
        prepare_slim_derive_initial_noise=True,
    )

    cmd = seen["cmd"]
    assert cmd[2] == "flash_rt.models.cosmos3_edge.action_only_official"
    assert "--flashrt-live-boundary-prepare-live" in cmd
    assert cmd[cmd.index("--flashrt-live-boundary-out") + 1] == str(tmp_path / "derived_boundary.safetensors")
    assert cmd[cmd.index("--flashrt-live-handoff-trace-out") + 1] == str(tmp_path / "prepare_live_trace.json")
    assert cmd[cmd.index("--flashrt-upstream-trace-out") + 1] == str(tmp_path / "prepare_live_upstream.json")
    assert "--flashrt-live-boundary-in" not in cmd
    assert "--flashrt-live-boundary-prepare-in" not in cmd
    assert "--flashrt-prepare-slim-no-raw-state-vision" in cmd
    assert "--flashrt-prepare-slim-derive-condition-reference" in cmd
    assert "--flashrt-prepare-slim-derive-initial-noise" in cmd
    assert result["live_boundary_prepare_live"] is True
    assert result["live_boundary_out"] == str(tmp_path / "derived_boundary.safetensors")
    assert result["live_handoff_trace_out"] == str(tmp_path / "prepare_live_trace.json")
    assert result["upstream_trace_out"] == str(tmp_path / "prepare_live_upstream.json")
    assert result["prepare_slim_no_raw_state_vision"] is True
    assert result["prepare_slim_derive_condition_reference"] is True
    assert result["prepare_slim_derive_initial_noise"] is True


def test_cosmos3_edge_live_boundary_prepare_live_allows_warm_prepare_cache_cmd(tmp_path, monkeypatch):
    from flash_rt.frontends.torch import cosmos3_edge_thor
    from flash_rt.frontends.torch.cosmos3_edge_thor import (
        Cosmos3EdgeTorchFrontendThor,
    )

    input_json = tmp_path / "input.json"
    input_json.write_text('{"samples":[]}\n', encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cosmos3_edge_thor.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    pipe = Cosmos3EdgeTorchFrontendThor("unused-checkpoint", hardware="thor")
    pipe.set_prompt(input_json=str(input_json))
    result = pipe.infer(
        output_dir=str(tmp_path / "out"),
        backend="official_action_only",
        cosmos_root=str(tmp_path),
        config_file=str(config_file),
        live_boundary_prepare_live=True,
        cache_warmup_vae=True,
        cache_warmup_prepare=True,
        warmup=1,
        prepare_slim_no_raw_state_vision=True,
        prepare_slim_derive_condition_reference=True,
        prepare_slim_derive_initial_noise=True,
    )

    cmd = seen["cmd"]
    assert "--flashrt-live-boundary-prepare-live" in cmd
    assert "--flashrt-cache-warmup-vae" in cmd
    assert "--flashrt-cache-warmup-prepare" in cmd
    assert cmd[cmd.index("--warmup") + 1] == "1"
    assert "--flashrt-live-boundary-in" not in cmd
    assert "--flashrt-live-boundary-prepare-in" not in cmd
    assert result["live_boundary_prepare_live"] is True
    assert result["cache_warmup_vae"] is True
    assert result["cache_warmup_prepare"] is True
    assert result["warmup"] == 1


def test_cosmos3_edge_vae_boundary_builds_official_cmd(tmp_path, monkeypatch):
    from flash_rt.frontends.torch import cosmos3_edge_thor
    from flash_rt.frontends.torch.cosmos3_edge_thor import (
        Cosmos3EdgeTorchFrontendThor,
    )

    input_json = tmp_path / "input.json"
    input_json.write_text('{"samples":[]}\n', encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cosmos3_edge_thor.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    pipe = Cosmos3EdgeTorchFrontendThor("unused-checkpoint", hardware="thor")
    pipe.set_prompt(input_json=str(input_json))
    result = pipe.infer(
        output_dir=str(tmp_path / "out"),
        backend="official_action_only",
        cosmos_root=str(tmp_path),
        config_file=str(config_file),
        live_prelayer_bootstrap=True,
        vae_encode_dump_out="vae.safetensors",
        vae_encode_dump_input=True,
        vae_encode_profile_out="vae_profile.json",
        vae_native_rms_silu=True,
        vae_t1_conv2d=True,
        vae_native_avgdown3d=True,
        vae_channels_last3d_conv320=True,
    )

    cmd = seen["cmd"]
    assert cmd[cmd.index("--flashrt-vae-encode-dump-out") + 1] == str(tmp_path / "vae.safetensors")
    assert "--flashrt-vae-encode-dump-input" in cmd
    assert cmd[cmd.index("--flashrt-vae-encode-profile-out") + 1] == str(tmp_path / "vae_profile.json")
    assert "--flashrt-vae-native-rms-silu" in cmd
    assert "--flashrt-vae-t1-conv2d" in cmd
    assert "--flashrt-vae-native-avgdown3d" in cmd
    assert "--flashrt-vae-channels-last3d-conv320" in cmd
    assert "--flashrt-cache-warmup-vae" not in cmd
    assert result["vae_encode_dump_out"] == str(tmp_path / "vae.safetensors")
    assert result["vae_encode_dump_input"] is True
    assert result["vae_encode_profile_out"] == str(tmp_path / "vae_profile.json")
    assert result["vae_native_rms_silu"] is True
    assert result["vae_t1_conv2d"] is True
    assert result["vae_native_avgdown3d"] is True
    assert result["vae_channels_last3d_conv320"] is True


def test_cosmos3_edge_vae_compile_builds_official_cmd(tmp_path, monkeypatch):
    from flash_rt.frontends.torch import cosmos3_edge_thor
    from flash_rt.frontends.torch.cosmos3_edge_thor import (
        Cosmos3EdgeTorchFrontendThor,
    )

    input_json = tmp_path / "input.json"
    input_json.write_text('{"samples":[]}\n', encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cosmos3_edge_thor.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    pipe = Cosmos3EdgeTorchFrontendThor("unused-checkpoint", hardware="thor")
    pipe.set_prompt(input_json=str(input_json))
    result = pipe.infer(
        output_dir=str(tmp_path / "out"),
        backend="official_action_only",
        cosmos_root=str(tmp_path),
        config_file=str(config_file),
        live_prelayer_bootstrap=True,
        vae_compile_encode=True,
        vae_compile_trace_out="vae_compile.json",
    )

    cmd = seen["cmd"]
    assert "--flashrt-vae-compile-encode" in cmd
    assert cmd[cmd.index("--flashrt-vae-compile-trace-out") + 1] == str(tmp_path / "vae_compile.json")
    assert result["vae_compile_encode"] is True
    assert result["vae_compile_trace_out"] == str(tmp_path / "vae_compile.json")


def test_cosmos3_edge_prepare_boundary_builds_official_cmd(tmp_path, monkeypatch):
    from flash_rt.frontends.torch import cosmos3_edge_thor
    from flash_rt.frontends.torch.cosmos3_edge_thor import (
        Cosmos3EdgeTorchFrontendThor,
    )

    input_json = tmp_path / "input.json"
    input_json.write_text('{"samples":[]}\n', encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cosmos3_edge_thor.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    pipe = Cosmos3EdgeTorchFrontendThor("unused-checkpoint", hardware="thor")
    pipe.set_prompt(input_json=str(input_json))
    result = pipe.infer(
        output_dir=str(tmp_path / "out"),
        backend="official_action_only",
        cosmos_root=str(tmp_path),
        config_file=str(config_file),
        live_prelayer_bootstrap=True,
        prepare_dump_out="prepare.pt",
        prepare_replay_in="prepare_in.pt",
        prepare_inventory_out="prepare_inventory.json",
        prepare_slim_no_raw_state_vision=True,
        prepare_slim_derive_condition_reference=True,
        prepare_slim_derive_initial_noise=True,
    )

    cmd = seen["cmd"]
    assert cmd[cmd.index("--flashrt-prepare-dump-out") + 1] == str(tmp_path / "prepare.pt")
    assert cmd[cmd.index("--flashrt-prepare-replay-in") + 1] == str(tmp_path / "prepare_in.pt")
    assert cmd[cmd.index("--flashrt-prepare-inventory-out") + 1] == str(tmp_path / "prepare_inventory.json")
    assert "--flashrt-prepare-slim-no-raw-state-vision" in cmd
    assert "--flashrt-prepare-slim-derive-condition-reference" in cmd
    assert "--flashrt-prepare-slim-derive-initial-noise" in cmd
    assert result["prepare_dump_out"] == str(tmp_path / "prepare.pt")
    assert result["prepare_replay_in"] == str(tmp_path / "prepare_in.pt")
    assert result["prepare_inventory_out"] == str(tmp_path / "prepare_inventory.json")
    assert result["prepare_slim_no_raw_state_vision"] is True
    assert result["prepare_slim_derive_condition_reference"] is True
    assert result["prepare_slim_derive_initial_noise"] is True


def test_cosmos3_edge_action_only_output_writer(tmp_path):
    from flash_rt.frontends.torch.cosmos3_edge_thor import (
        Cosmos3EdgeTorchFrontendThor,
    )

    output_path = Cosmos3EdgeTorchFrontendThor._write_action_only_output(
        [[1.0, 2.0, 3.0]],
        tmp_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["status"] == "success"
    assert payload["outputs"][0]["content"]["action"] == [[1.0, 2.0, 3.0]]
    assert payload["outputs"][0]["files"] == []


def test_cosmos3_edge_official_action_only_rewriter(tmp_path):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _rewrite_sample_outputs_action_only,
    )

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "vision.mp4").write_bytes(b"fake")
    sample_outputs = sample_dir / "sample_outputs.json"
    sample_outputs.write_text(
        json.dumps(
            {
                "status": "success",
                "outputs": [
                    {
                        "content": {"action": [[1.0, 2.0]]},
                        "files": ["vision.mp4"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _rewrite_sample_outputs_action_only(tmp_path) == 1
    payload = json.loads(sample_outputs.read_text(encoding="utf-8"))
    assert payload["outputs"][0]["files"] == []
    assert payload["outputs"][0]["content"]["action"] == [[1.0, 2.0]]
    assert not (sample_dir / "vision.mp4").exists()


def test_cosmos3_edge_official_action_only_live_dump_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_live_dump_out,
    )

    argv, path = _extract_live_dump_out(
        ["-i", "sample.json", "--flashrt-live-dump-out", "/tmp/live.safetensors", "--seed", "0"]
    )
    assert argv == ["-i", "sample.json", "--seed", "0"]
    assert path == Path("/tmp/live.safetensors")

    argv, path = _extract_live_dump_out(
        ["-i", "sample.json", "--flashrt-live-dump-out=/tmp/live2.safetensors"]
    )
    assert argv == ["-i", "sample.json"]
    assert path == Path("/tmp/live2.safetensors")

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_LIVE_DUMP_OUT", "/tmp/live3.safetensors")
    argv, path = _extract_live_dump_out(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert path == Path("/tmp/live3.safetensors")


def test_cosmos3_edge_official_action_only_live_handoff_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_live_handoff_args,
    )

    argv, enabled, boundary, boundary_in, boundary_prepare_in, prepare_live, trace, prelayer = _extract_live_handoff_args(
        [
            "-i",
            "sample.json",
            "--flashrt-live-flashrt-handoff",
            "--flashrt-live-prelayer-bootstrap",
            "--flashrt-live-boundary-out",
            "/tmp/boundary.safetensors",
            "--flashrt-live-boundary-in",
            "/tmp/prelayer_boundary.safetensors",
            "--flashrt-live-handoff-trace-out",
            "/tmp/trace.json",
            "--seed",
            "0",
        ]
    )
    assert argv == ["-i", "sample.json", "--seed", "0"]
    assert enabled is True
    assert prelayer is True
    assert boundary == Path("/tmp/boundary.safetensors")
    assert boundary_in == Path("/tmp/prelayer_boundary.safetensors")
    assert boundary_prepare_in is None
    assert prepare_live is False
    assert trace == Path("/tmp/trace.json")

    argv, enabled, boundary, boundary_in, boundary_prepare_in, prepare_live, trace, prelayer = _extract_live_handoff_args(
        [
            "-i",
            "sample.json",
            "--flashrt-live-boundary-out=/tmp/boundary2.safetensors",
            "--flashrt-live-boundary-in=/tmp/prelayer_boundary2.safetensors",
            "--flashrt-live-boundary-prepare-in=/tmp/slim_prepare.pt",
            "--flashrt-live-boundary-prepare-live",
            "--flashrt-live-handoff-trace-out=/tmp/trace2.json",
        ]
    )
    assert argv == ["-i", "sample.json"]
    assert enabled is True
    assert prelayer is False
    assert boundary == Path("/tmp/boundary2.safetensors")
    assert boundary_in == Path("/tmp/prelayer_boundary2.safetensors")
    assert boundary_prepare_in == Path("/tmp/slim_prepare.pt")
    assert prepare_live is True
    assert trace == Path("/tmp/trace2.json")

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_LIVE_FLASHRT_HANDOFF", "1")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_LIVE_PRELAYER_BOOTSTRAP", "1")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_LIVE_BOUNDARY_OUT", "/tmp/boundary3.safetensors")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_LIVE_BOUNDARY_IN", "/tmp/prelayer_boundary3.safetensors")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_LIVE_BOUNDARY_PREPARE_IN", "/tmp/slim_prepare3.pt")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_LIVE_BOUNDARY_PREPARE_LIVE", "1")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_LIVE_HANDOFF_TRACE_OUT", "/tmp/trace3.json")
    argv, enabled, boundary, boundary_in, boundary_prepare_in, prepare_live, trace, prelayer = _extract_live_handoff_args(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is True
    assert prelayer is True
    assert boundary == Path("/tmp/boundary3.safetensors")
    assert boundary_in == Path("/tmp/prelayer_boundary3.safetensors")
    assert boundary_prepare_in == Path("/tmp/slim_prepare3.pt")
    assert prepare_live is True
    assert trace == Path("/tmp/trace3.json")


def test_cosmos3_edge_live_handoff_trace_native_scheduler_summary(tmp_path):
    from flash_rt.models.cosmos3_edge.action_only_official import _LiveFlashRTHandoff

    trace = tmp_path / "trace.json"
    handoff = _LiveFlashRTHandoff(Path("/tmp/checkpoint"), None, None, None, None, trace, prelayer_bootstrap=True)
    handoff.trace["flashrt_velocity_calls"] = [{"step": 0, "s": 0.25}, {"step": 1, "s": 0.35}]
    handoff.trace["native_scheduler"]["runs"].append(
        {
            "num_steps": 30,
            "shift": 10.0,
            "total_s": 0.42,
            "scheduler_step_total_s": 0.03,
        }
    )

    handoff.save_trace()

    payload = json.loads(trace.read_text(encoding="utf-8"))
    assert payload["summary"]["native_scheduler_enabled"] is True
    assert payload["summary"]["native_scheduler_run_count"] == 1
    assert payload["summary"]["native_scheduler_step_count"] == 30
    assert payload["summary"]["native_scheduler_step_total_s"] == pytest.approx(0.03)


def test_cosmos3_edge_official_action_only_upstream_trace_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_upstream_trace_out,
    )

    argv, path = _extract_upstream_trace_out(
        ["-i", "sample.json", "--flashrt-upstream-trace-out", "/tmp/upstream.json", "--seed", "0"]
    )
    assert argv == ["-i", "sample.json", "--seed", "0"]
    assert path == Path("/tmp/upstream.json")

    argv, path = _extract_upstream_trace_out(
        ["-i", "sample.json", "--flashrt-upstream-trace-out=/tmp/upstream2.json"]
    )
    assert argv == ["-i", "sample.json"]
    assert path == Path("/tmp/upstream2.json")

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_UPSTREAM_TRACE_OUT", "/tmp/upstream3.json")
    argv, path = _extract_upstream_trace_out(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert path == Path("/tmp/upstream3.json")


def test_cosmos3_edge_official_action_only_vae_boundary_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_vae_encode_boundary_args,
    )

    argv, dump_out, latent_in, dump_input = _extract_vae_encode_boundary_args(
        [
            "-i",
            "sample.json",
            "--flashrt-vae-encode-dump-out",
            "/tmp/vae.safetensors",
            "--flashrt-vae-latent-in=/tmp/latent.safetensors",
            "--flashrt-vae-encode-dump-input",
            "--seed",
            "0",
        ]
    )
    assert argv == ["-i", "sample.json", "--seed", "0"]
    assert dump_out == Path("/tmp/vae.safetensors")
    assert latent_in == Path("/tmp/latent.safetensors")
    assert dump_input is True

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_VAE_ENCODE_DUMP_OUT", "/tmp/env_vae.safetensors")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_VAE_LATENT_IN", "/tmp/env_latent.safetensors")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_VAE_ENCODE_DUMP_INPUT", "1")
    argv, dump_out, latent_in, dump_input = _extract_vae_encode_boundary_args(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert dump_out == Path("/tmp/env_vae.safetensors")
    assert latent_in == Path("/tmp/env_latent.safetensors")
    assert dump_input is True


def test_cosmos3_edge_official_action_only_vae_profile_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_vae_encode_profile_out,
    )

    argv, path = _extract_vae_encode_profile_out(
        ["-i", "sample.json", "--flashrt-vae-encode-profile-out", "/tmp/profile.json", "--seed", "0"]
    )
    assert argv == ["-i", "sample.json", "--seed", "0"]
    assert path == Path("/tmp/profile.json")

    argv, path = _extract_vae_encode_profile_out(
        ["-i", "sample.json", "--flashrt-vae-encode-profile-out=/tmp/profile2.json"]
    )
    assert argv == ["-i", "sample.json"]
    assert path == Path("/tmp/profile2.json")

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_VAE_ENCODE_PROFILE_OUT", "/tmp/profile3.json")
    argv, path = _extract_vae_encode_profile_out(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert path == Path("/tmp/profile3.json")


def test_cosmos3_edge_official_action_only_vae_profile_native_candidate_summary(tmp_path):
    from flash_rt.models.cosmos3_edge.action_only_official import _VAEEncodeProfiler

    out = tmp_path / "vae_profile.json"
    profiler = _VAEEncodeProfiler(out)
    profiler.encode_calls.append({"encode_call": 0, "input": None, "output": None, "s": 1.0})
    profiler.modules.extend(
        [
            {"name": "encoder.block.conv", "class": "CausalConv3d"},
            {"name": "encoder.conv1", "class": "CausalConv3d"},
        ]
    )
    profiler.events.extend(
        [
            {
                "encode_call": 0,
                "name": "encoder.conv1",
                "class": "CausalConv3d",
                "input": [
                    {"shape": [1, 12, 1, 240, 416]},
                    None,
                ],
                "output": {"shape": [1, 160, 1, 240, 416]},
                "s": 0.02,
                "parameter_numel": 52000,
            },
            {
                "encode_call": 0,
                "name": "encoder.block.conv",
                "class": "CausalConv3d",
                "input": [
                    {"shape": [1, 160, 24, 240, 416]},
                    {"shape": [1, 160, 2, 240, 416]},
                ],
                "output": {"shape": [1, 160, 24, 240, 416]},
                "s": 0.11,
                "parameter_numel": 691360,
            },
        ]
    )

    profiler.save()

    payload = json.loads(out.read_text(encoding="utf-8"))
    candidates = payload["native_candidate_summary"]
    assert candidates[0]["candidate_type"] == "steady_cached_causal_conv3d"
    assert candidates[0]["name"] == "encoder.block.conv"
    assert candidates[0]["input_shape"] == [1, 160, 24, 240, 416]
    assert candidates[0]["cache_shape"] == [1, 160, 2, 240, 416]
    assert candidates[1]["candidate_type"] == "prime_t1_no_cache"

    shape_summary = payload["shape_summary"]
    steady_shape = next(item for item in shape_summary if item["name"] == "encoder.block.conv")
    assert steady_shape["count"] == 1
    assert steady_shape["total_s"] == pytest.approx(0.11)


def test_cosmos3_edge_official_action_only_vae_native_rms_silu_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_vae_native_rms_silu,
    )

    argv, enabled = _extract_vae_native_rms_silu(
        ["-i", "sample.json", "--flashrt-vae-native-rms-silu", "--seed", "0"]
    )
    assert argv == ["-i", "sample.json", "--seed", "0"]
    assert enabled is True

    argv, enabled = _extract_vae_native_rms_silu(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is False

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_VAE_NATIVE_RMS_SILU", "1")
    argv, enabled = _extract_vae_native_rms_silu(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is True


def test_cosmos3_edge_official_action_only_vae_t1_conv2d_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_vae_t1_conv2d,
    )

    argv, enabled = _extract_vae_t1_conv2d(["-i", "sample.json", "--flashrt-vae-t1-conv2d", "--seed", "0"])
    assert argv == ["-i", "sample.json", "--seed", "0"]
    assert enabled is True

    argv, enabled = _extract_vae_t1_conv2d(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is False

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_VAE_T1_CONV2D", "1")
    argv, enabled = _extract_vae_t1_conv2d(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is True


def test_cosmos3_edge_official_action_only_vae_native_avgdown3d_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_vae_native_avgdown3d,
    )

    argv, enabled = _extract_vae_native_avgdown3d(
        ["-i", "sample.json", "--flashrt-vae-native-avgdown3d", "--seed", "0"]
    )
    assert argv == ["-i", "sample.json", "--seed", "0"]
    assert enabled is True

    argv, enabled = _extract_vae_native_avgdown3d(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is False

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_VAE_NATIVE_AVGDOWN3D", "1")
    argv, enabled = _extract_vae_native_avgdown3d(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is True


def test_cosmos3_edge_official_action_only_vae_channels_last3d_conv320_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_vae_channels_last3d_conv320,
    )

    argv, enabled = _extract_vae_channels_last3d_conv320(
        ["-i", "sample.json", "--flashrt-vae-channels-last3d-conv320", "--seed", "0"]
    )
    assert argv == ["-i", "sample.json", "--seed", "0"]
    assert enabled is True

    argv, enabled = _extract_vae_channels_last3d_conv320(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is False

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_VAE_CHANNELS_LAST3D_CONV320", "1")
    argv, enabled = _extract_vae_channels_last3d_conv320(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is True


def test_cosmos3_edge_official_action_only_vae_compile_encode_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_vae_compile_encode,
    )

    argv, enabled, trace = _extract_vae_compile_encode(
        ["-i", "sample.json", "--flashrt-vae-compile-encode", "--flashrt-vae-compile-trace-out", "/tmp/aot.json"]
    )
    assert argv == ["-i", "sample.json"]
    assert enabled is True
    assert trace == Path("/tmp/aot.json")

    argv, enabled, trace = _extract_vae_compile_encode(
        ["-i", "sample.json", "--flashrt-vae-compile-trace-out=/tmp/aot2.json"]
    )
    assert argv == ["-i", "sample.json"]
    assert enabled is False
    assert trace == Path("/tmp/aot2.json")

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_VAE_COMPILE_ENCODE", "1")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_VAE_COMPILE_TRACE_OUT", "/tmp/aot3.json")
    argv, enabled, trace = _extract_vae_compile_encode(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is True
    assert trace == Path("/tmp/aot3.json")


def test_cosmos3_edge_official_action_only_prepare_boundary_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_prepare_boundary_args,
    )

    argv, dump, replay, inventory, slim, derive_ref, derive_noise = _extract_prepare_boundary_args(
        [
            "-i",
            "sample.json",
            "--flashrt-prepare-dump-out",
            "/tmp/prepare.pt",
            "--flashrt-prepare-replay-in=/tmp/prepare_in.pt",
            "--flashrt-prepare-inventory-out",
            "/tmp/prepare.json",
            "--flashrt-prepare-slim-no-raw-state-vision",
            "--flashrt-prepare-slim-derive-condition-reference",
            "--flashrt-prepare-slim-derive-initial-noise",
        ]
    )
    assert argv == ["-i", "sample.json"]
    assert dump == Path("/tmp/prepare.pt")
    assert replay == Path("/tmp/prepare_in.pt")
    assert inventory == Path("/tmp/prepare.json")
    assert slim is True
    assert derive_ref is True
    assert derive_noise is True

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_PREPARE_DUMP_OUT", "/tmp/env_prepare.pt")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_PREPARE_REPLAY_IN", "/tmp/env_prepare_in.pt")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_PREPARE_INVENTORY_OUT", "/tmp/env_prepare.json")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_PREPARE_SLIM_NO_RAW_STATE_VISION", "1")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_PREPARE_SLIM_DERIVE_CONDITION_REFERENCE", "1")
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_PREPARE_SLIM_DERIVE_INITIAL_NOISE", "1")
    argv, dump, replay, inventory, slim, derive_ref, derive_noise = _extract_prepare_boundary_args(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert dump == Path("/tmp/env_prepare.pt")
    assert replay == Path("/tmp/env_prepare_in.pt")
    assert inventory == Path("/tmp/env_prepare.json")
    assert slim is True
    assert derive_ref is True
    assert derive_noise is True


def test_cosmos3_edge_prepare_payload_inventory_tensors():
    from dataclasses import dataclass

    import torch

    from flash_rt.models.cosmos3_edge.action_only_official import (
        _prepare_payload_inventory,
        _slim_prepare_payload_no_raw_state_vision,
    )

    @dataclass
    class ToyGenData:
        raw_state_vision: list
        x0_tokens_vision: list

    gen_data = ToyGenData(
        raw_state_vision=[torch.zeros((1, 3, 2), dtype=torch.float32)],
        x0_tokens_vision=[torch.zeros((1, 1, 2), dtype=torch.float32)],
    )
    payload = (
        [],
        gen_data,
        [],
        [],
        [torch.zeros((8,), dtype=torch.float32)],
        [torch.zeros((8,), dtype=torch.float32)],
        [torch.zeros((8,), dtype=torch.bool)],
    )
    inventory = _prepare_payload_inventory(payload, signature=("sig",), source="unit")

    assert inventory["schema"] == "flashrt_cosmos3_edge_prepare_inventory_v1"
    assert inventory["tensor_count"] == 5
    assert inventory["tensor_bytes"] == 1 * 3 * 2 * 4 + 1 * 1 * 2 * 4 + 8 * 4 + 8 * 4 + 8
    assert inventory["field_names"] == [
        "sequence_plans",
        "gen_data_clean",
        "cond_text_tokens",
        "uncond_text_tokens",
        "initial_noise",
        "condition_reference",
        "condition_mask",
    ]
    by_top = {item["field"]: item for item in inventory["tensor_bytes_by_top_level"]}
    assert by_top["gen_data_clean"]["tensor_count"] == 2
    assert by_top["condition_mask"]["bytes"] == 8

    slim = _slim_prepare_payload_no_raw_state_vision(payload)
    assert slim[1].raw_state_vision is None
    assert payload[1].raw_state_vision is not None
    slim_inventory = _prepare_payload_inventory(slim, signature=("sig",), source="unit")
    assert slim_inventory["tensor_count"] == 4
    paths = {item["path"] for item in slim_inventory["largest_tensors"]}
    assert "gen_data_clean.raw_state_vision[0]" not in paths


def test_cosmos3_edge_official_action_only_warmup_vae_cache_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_warmup_vae_cache,
    )

    argv, enabled = _extract_warmup_vae_cache(
        ["-i", "sample.json", "--flashrt-cache-warmup-vae", "--seed", "0"]
    )
    assert argv == ["-i", "sample.json", "--seed", "0"]
    assert enabled is True

    argv, enabled = _extract_warmup_vae_cache(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is False

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_CACHE_WARMUP_VAE", "1")
    argv, enabled = _extract_warmup_vae_cache(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is True


def test_cosmos3_edge_official_action_only_warmup_prepare_cache_arg_parsing(monkeypatch):
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _extract_warmup_prepare_cache,
    )

    argv, enabled = _extract_warmup_prepare_cache(
        ["-i", "sample.json", "--flashrt-cache-warmup-prepare", "--seed", "0"]
    )
    assert argv == ["-i", "sample.json", "--seed", "0"]
    assert enabled is True

    argv, enabled = _extract_warmup_prepare_cache(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is False

    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_CACHE_WARMUP_PREPARE", "1")
    argv, enabled = _extract_warmup_prepare_cache(["-i", "sample.json"])
    assert argv == ["-i", "sample.json"]
    assert enabled is True


def test_cosmos3_edge_upstream_trace_marks_expected_handoff_exception(tmp_path):
    from flash_rt.models.cosmos3_edge.action_only_official import _UpstreamTrace

    class _FlashRTVelocityReady(RuntimeError):
        pass

    trace = _UpstreamTrace(tmp_path / "trace.json")
    with pytest.raises(_FlashRTVelocityReady):
        with trace.record("Cosmos3VFMNetwork.forward", expected_exceptions=("_FlashRTVelocityReady",)):
            raise _FlashRTVelocityReady("handoff")

    trace.save()
    payload = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert payload["events"][0]["ok"] is True
    assert payload["events"][0]["metadata"]["expected_exception"] == "_FlashRTVelocityReady"
    assert payload["summary"][0]["ok_count"] == 1


def test_cosmos3_edge_warmup_vae_cache_signature_includes_content():
    torch = pytest.importorskip("torch")
    from flash_rt.models.cosmos3_edge.action_only_official import _WarmupVAECache

    a = torch.zeros(2, 3, dtype=torch.float32)
    b = a.clone()
    c = a.clone()
    c[-1, -1] = 1.0

    assert _WarmupVAECache._signature(a) == _WarmupVAECache._signature(b)
    assert _WarmupVAECache._signature(a) != _WarmupVAECache._signature(c)


def test_cosmos3_edge_warmup_prepare_cache_signature_includes_content():
    torch = pytest.importorskip("torch")
    from flash_rt.models.cosmos3_edge.action_only_official import _WarmupPrepareCache

    batch_a = {"video": [torch.zeros(2, 3, dtype=torch.float32)], "caption": ["pick"]}
    batch_b = {"video": [batch_a["video"][0].clone()], "caption": ["pick"]}
    batch_c = {"video": [batch_a["video"][0].clone()], "caption": ["pick"]}
    batch_c["video"][0][-1, -1] = 1.0

    assert _WarmupPrepareCache._signature(batch_a, [0], False) == _WarmupPrepareCache._signature(batch_b, [0], False)
    assert _WarmupPrepareCache._signature(batch_a, [0], False) != _WarmupPrepareCache._signature(batch_c, [0], False)
    assert _WarmupPrepareCache._signature(batch_a, [0], False) != _WarmupPrepareCache._signature(batch_a, [1], False)


def test_wan22_infer_exposes_teacache_parameters():
    import inspect
    from flash_rt.frontends.torch.wan22_rtx import Wan22TorchFrontendRtx

    sig = inspect.signature(Wan22TorchFrontendRtx.infer)
    for name in (
        "teacache",
        "teacache_threshold",
        "teacache_start_step",
        "teacache_end_step",
        "teacache_cache_device",
    ):
        assert name in sig.parameters


def test_load_model_accepts_groot_n17_config():
    from flash_rt.api import load_model

    class GrootN17RtxFp8Frontend:
        seen = []

        def __init__(self, checkpoint, num_views=2, embodiment_tag=None):
            type(self).seen.append({
                "checkpoint": checkpoint,
                "num_views": num_views,
                "embodiment_tag": embodiment_tag,
            })

        def set_prompt(self, *args, **kwargs):
            return None

        def infer(self, *args, **kwargs):
            return None

    GrootN17RtxFp8Frontend.seen = []
    with patch("flash_rt.hardware.resolve_pipeline_class",
               return_value=GrootN17RtxFp8Frontend) as resolve:
        model_default = load_model(
            "unused-checkpoint-default",
            config="groot_n17",
            framework="torch",
            hardware="rtx_sm89",
            num_views=2,
            embodiment_tag="oxe_droid_relative_eef_relative_joint",
        )
        model_use_fp8 = load_model(
            "unused-checkpoint-explicit-fp8",
            config="groot_n17",
            framework="torch",
            hardware="rtx_sm89",
            num_views=2,
            embodiment_tag="oxe_droid_relative_eef_relative_joint",
            use_fp8=True,
        )

    assert isinstance(model_default._pipe, GrootN17RtxFp8Frontend)
    assert isinstance(model_use_fp8._pipe, GrootN17RtxFp8Frontend)
    assert [call.args for call in resolve.call_args_list] == [
        ("groot_n17", "torch", "rtx_sm89"),
        ("groot_n17", "torch", "rtx_sm89"),
    ]
    assert GrootN17RtxFp8Frontend.seen == [
        {
            "checkpoint": "unused-checkpoint-default",
            "num_views": 2,
            "embodiment_tag": "oxe_droid_relative_eef_relative_joint",
        },
        {
            "checkpoint": "unused-checkpoint-explicit-fp8",
            "num_views": 2,
            "embodiment_tag": "oxe_droid_relative_eef_relative_joint",
        },
    ]


def test_load_model_routes_groot_n17_rtx_sm120_default_to_fp8_frontend():
    from flash_rt.api import load_model

    class ResolvedFrontend:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "load_model() should rewrite default groot_n17 RTX SM120 "
                "requests to the FP8 production frontend")

    class GrootN17RtxFp8Frontend:
        seen = None

        def __init__(self, checkpoint, num_views=2, embodiment_tag=None):
            type(self).seen = {
                "checkpoint": checkpoint,
                "num_views": num_views,
                "embodiment_tag": embodiment_tag,
            }

        def set_prompt(self, *args, **kwargs):
            return None

        def infer(self, *args, **kwargs):
            return None

    with patch("flash_rt.hardware.resolve_pipeline_class",
              return_value=ResolvedFrontend), \
            patch("flash_rt.frontends.torch.groot_n17_rtx_fp8."
                  "GrootN17TorchFrontendRtxFP8",
                  GrootN17RtxFp8Frontend):
        model = load_model(
            "unused-checkpoint-sm120-fp8",
            config="groot_n17",
            framework="torch",
            hardware="rtx_sm120",
            num_views=2,
            embodiment_tag="oxe_droid_relative_eef_relative_joint",
        )

    assert isinstance(model._pipe, GrootN17RtxFp8Frontend)
    assert GrootN17RtxFp8Frontend.seen == {
        "checkpoint": "unused-checkpoint-sm120-fp8",
        "num_views": 2,
        "embodiment_tag": "oxe_droid_relative_eef_relative_joint",
    }


def test_load_model_routes_groot_n17_rtx_fp16_reference_path():
    from flash_rt.api import load_model

    class ResolvedFrontend:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "load_model() should rewrite groot_n17 RTX SM89 FP16 "
                "requests to the FP16 reference frontend")

    class GrootN17RtxFp16Frontend:
        seen = None

        def __init__(self, checkpoint, num_views=2, embodiment_tag=None):
            type(self).seen = {
                "checkpoint": checkpoint,
                "num_views": num_views,
                "embodiment_tag": embodiment_tag,
            }

        def set_prompt(self, *args, **kwargs):
            return None

        def infer(self, *args, **kwargs):
            return None

    with patch("flash_rt.hardware.resolve_pipeline_class",
              return_value=ResolvedFrontend), \
            patch("flash_rt.frontends.torch.groot_n17_rtx_sm89_fp16."
                  "GrootN17TorchFrontendRtxSm89FP16",
                  GrootN17RtxFp16Frontend):
        model = load_model(
            "unused-checkpoint-fp16",
            config="groot_n17",
            framework="torch",
            hardware="rtx_sm89",
            num_views=2,
            embodiment_tag="oxe_droid_relative_eef_relative_joint",
            use_fp16=True,
            use_fp8=False,
        )

    assert isinstance(model._pipe, GrootN17RtxFp16Frontend)
    assert GrootN17RtxFp16Frontend.seen == {
        "checkpoint": "unused-checkpoint-fp16",
        "num_views": 2,
        "embodiment_tag": "oxe_droid_relative_eef_relative_joint",
    }


def test_groot_n17_rtx_fp8_layout_selection():
    from flash_rt.frontends.torch.groot_n17_rtx_sm89 import (
        GrootN17TorchFrontendRtxSm89,
    )

    assert GrootN17TorchFrontendRtxSm89.fp8_layout == "nk"


def test_groot_n17_rtx_sm89_runtime_weights_are_materialized_in_nk_layout():
    from flash_rt.frontends.torch.groot_n17_rtx_sm89 import (
        _GrootN17FP8BackboneMixin,
    )

    class FakeMatrix:
        def __init__(self, rows, cols, label):
            self.shape = (rows, cols)
            self.label = label

        def __getitem__(self, key):
            row_sel, col_sel = key
            if row_sel != slice(None):
                raise AssertionError("test fake only supports full-row slices")
            start = 0 if col_sel.start is None else col_sel.start
            stop = self.shape[1] if col_sel.stop is None else col_sel.stop
            return FakeMatrix(self.shape[0], stop - start,
                              f"{self.label}[{start}:{stop}]")

        def t(self):
            return FakeMatrix(self.shape[1], self.shape[0],
                              f"{self.label}.t")

        def contiguous(self):
            return FakeMatrix(self.shape[0], self.shape[1],
                              f"{self.label}.contiguous")

    frontend = object.__new__(_GrootN17FP8BackboneMixin)
    frontend._vit_qkv_w = [FakeMatrix(1024, 3072, f"vit_qkv_{i}")
                           for i in range(24)]
    frontend._vit_o_w = [FakeMatrix(1024, 1024, f"vit_o_{i}")
                         for i in range(24)]
    frontend._vit_fc1_w = [FakeMatrix(1024, 4096, f"vit_fc1_{i}")
                           for i in range(24)]
    frontend._vit_fc2_w = [FakeMatrix(4096, 1024, f"vit_fc2_{i}")
                           for i in range(24)]

    for j in range(3):
        setattr(frontend, f"_dsm{j}_fc1_w", FakeMatrix(4096, 4096, f"dsm{j}_fc1"))
        setattr(frontend, f"_dsm{j}_fc2_w", FakeMatrix(4096, 2048, f"dsm{j}_fc2"))

    frontend._llm_qkv_w = [FakeMatrix(2048, 4096, f"llm_qkv_{i}")
                           for i in range(16)]
    frontend._llm_o_w = [FakeMatrix(2048, 2048, f"llm_o_{i}")
                         for i in range(16)]
    frontend._llm_gate_w = [FakeMatrix(2048, 6144, f"llm_gate_{i}")
                            for i in range(16)]
    frontend._llm_up_w = [FakeMatrix(2048, 6144, f"llm_up_{i}")
                          for i in range(16)]
    frontend._llm_down_w = [FakeMatrix(6144, 2048, f"llm_down_{i}")
                            for i in range(16)]

    frontend._vlsa_q_w = [FakeMatrix(2048, 2048, f"vlsa_q_{i}")
                          for i in range(4)]
    frontend._vlsa_k_w = [FakeMatrix(2048, 2048, f"vlsa_k_{i}")
                          for i in range(4)]
    frontend._vlsa_v_w = [FakeMatrix(2048, 2048, f"vlsa_v_{i}")
                          for i in range(4)]
    frontend._vlsa_o_w = [FakeMatrix(2048, 2048, f"vlsa_o_{i}")
                          for i in range(4)]
    frontend._vlsa_fc1_w = [FakeMatrix(2048, 8192, f"vlsa_fc1_{i}")
                            for i in range(4)]
    frontend._vlsa_fc2_w = [FakeMatrix(8192, 2048, f"vlsa_fc2_{i}")
                            for i in range(4)]

    frontend._prepare_fp8_runtime_weights()

    runtime = frontend._rtx_fp8_runtime
    assert runtime["vit_q"][0].shape == (1024, 1024)
    assert runtime["vit_fc1"][0].shape == (4096, 1024)
    assert runtime["dsm_fc2"][0].shape == (2048, 4096)
    assert runtime["llm_q"][0].shape == (2048, 2048)
    assert runtime["llm_gate"][0].shape == (6144, 2048)
    assert runtime["vlsa_fc2"][0].shape == (2048, 8192)
    assert runtime["vit_q"][0].label == "vit_qkv_0[0:1024].t.contiguous"
    assert runtime["llm_down"][0].label == "llm_down_0.t.contiguous"


def test_groot_n17_rtx_sm89_fp8_helper_dispatches_nt_then_cast():
    from flash_rt.models.groot_n17 import pipeline_rtx_sm89

    calls = []

    class Gemm:
        def fp8_nt_dev(self, *args):
            calls.append(("fp8_nt_dev", args))

    class Fvk:
        def cast_bf16_to_fp16(self, *args):
            calls.append(("cast_bf16_to_fp16", args))

    pipeline_rtx_sm89._fp8_matmul_fp16(
        Gemm(),
        Fvk(),
        act_fp8_ptr=11,
        weight_ptr=12,
        out_fp16_ptr=13,
        bf16_tmp_ptr=14,
        M=15,
        N=16,
        K=17,
        act_scale_ptr=18,
        weight_scale_ptr=19,
        stream=20,
    )

    assert calls == [
        ("fp8_nt_dev", (11, 12, 14, 15, 16, 17, 18, 19, 20)),
        ("cast_bf16_to_fp16", (14, 13, 15 * 16, 20)),
    ]


def test_groot_n17_rtx_sm89_rejects_use_fp8_false_without_fp16():
    from flash_rt.api import load_model

    with pytest.raises(ValueError, match="defaults to FP8"):
        load_model(
            "unused-checkpoint",
            config="groot_n17",
            framework="torch",
            hardware="rtx_sm89",
            use_fp8=False,
        )


def test_groot_n17_rtx_sm120_rejects_use_fp8_false_without_fp16():
    from flash_rt.api import load_model

    with pytest.raises(ValueError, match="defaults to FP8"):
        load_model(
            "unused-checkpoint",
            config="groot_n17",
            framework="torch",
            hardware="rtx_sm120",
            use_fp8=False,
        )


def test_load_model_redirects_qwen3_vl_to_direct_frontend():
    from flash_rt.api import load_model

    with pytest.raises(NotImplementedError, match="chat-style VLM"):
        load_model(
            "unused-checkpoint",
            config="qwen3_vl",
            framework="torch",
            hardware="rtx_sm89",
        )


def test_frontend_fp8_layout_selection():
    from flash_rt.frontends._fp8_layout import select_fp8_layout

    assert select_fp8_layout("rtx_sm89", None) == "nk"
    assert select_fp8_layout("rtx_sm120", None) == "kn"
    assert select_fp8_layout("rtx_sm120", "nk") == "nk"


def test_vla_frontend_constructors_accept_use_fp8():
    frontend_classes = {
        "flash_rt/frontends/torch/pi05_rtx.py": "Pi05TorchFrontendRtx",
        "flash_rt/frontends/jax/pi05_rtx.py": "Pi05JaxFrontendRtx",
        "flash_rt/frontends/torch/pi05_thor.py": "Pi05TorchFrontendThor",
        "flash_rt/frontends/jax/pi05_thor.py": "Pi05JaxFrontendThor",
        "flash_rt/frontends/torch/pi05_thor_fp4.py": "Pi05TorchFrontendThorFP4",
        "flash_rt/frontends/jax/pi05_thor_fp4.py": "Pi05JaxFrontendThorFP4",
        "flash_rt/frontends/torch/pi0_rtx.py": "Pi0TorchFrontendRtx",
        "flash_rt/frontends/jax/pi0_rtx.py": "Pi0JaxFrontendRtx",
        "flash_rt/frontends/torch/pi0_thor.py": "Pi0TorchFrontendThor",
        "flash_rt/frontends/jax/pi0_thor.py": "Pi0JaxFrontendThor",
        "flash_rt/frontends/torch/pi0fast.py": "Pi0FastTorchFrontend",
        "flash_rt/frontends/jax/pi0fast.py": "Pi0FastJaxFrontend",
        "flash_rt/frontends/torch/groot_rtx.py": "GrootTorchFrontendRtx",
        "flash_rt/frontends/torch/groot_thor.py": "GrootTorchFrontendThor",
    }

    repo_root = Path(__file__).resolve().parents[1]
    for rel_path, class_name in frontend_classes.items():
        tree = ast.parse((repo_root / rel_path).read_text())
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name)
        init = next(
            node for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        args = [arg.arg for arg in init.args.args]
        args += [arg.arg for arg in init.args.kwonlyargs]
        assert "use_fp8" in args, f"{class_name} must accept use_fp8"


def test_pi05_jax_rtx_frontend_mirrors_runtime_knobs():
    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse(
        (repo_root / "flash_rt/frontends/jax/pi05_rtx.py").read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Pi05JaxFrontendRtx")
    init = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    args = [arg.arg for arg in init.args.args]
    assigned = set()
    for node in ast.walk(init):
        targets = list(getattr(node, "targets", []))
        if isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        for target in targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                assigned.add(target.attr)

    for arg in (
        "num_steps",
        "vision_pool_factor",
        "vision_num_layers",
        "cache_frames",
    ):
        assert arg in args
    for attr in (
        "_num_steps",
        "_vision_pool_factor",
        "_vision_num_layers",
        "_cache_frames",
        "_frame_count",
        "_int8_weights",
        "_int8_weight_scales",
    ):
        assert attr in assigned


def _make_dit_fp8_fixtures(fp8_layout):
    """Build minimal Mock/dict fixtures for ``dit_forward`` FP8 dispatch tests."""
    from flash_rt.models.groot_n17 import pipeline_thor

    class DummyAttn:
        def get_slot_ptrs(self, site, layer_idx):
            return {"Q": 31, "K": 32, "V": 33, "O": 34}

        def run(self, site, layer_idx, q_seq, *, kv_seq=None, stream=0, state_nk=None):
            return None

    gemm = Mock()
    fvk = Mock()
    attn = DummyAttn()
    dims = {"Sa": 2, "D": 4, "FF": 8, "Skv_text": 1, "Skv_image": 1}
    bufs = {
        "h": 1, "xn": 2, "o_proj_out": 3, "ff_proj_out": 4,
        "qkv_xn_fp8": 5, "qkv_buf": 6, "xn_fp8": 7, "ff_fp8": 8,
    }
    weights = {
        "scale_msa": [11] * 32, "shift_msa": [12] * 32,
        "q_w": [13] * 32, "q_b": [14] * 32,
        "k_w": [15] * 32, "k_b": [16] * 32,
        "v_w": [17] * 32, "v_b": [18] * 32,
        "o_w": [19] * 32, "o_b": [20] * 32,
        "ff_proj_w": [21] * 32, "ff_proj_b": [22] * 32,
        "ff_down_w": [23] * 32, "ff_down_b": [24] * 32,
        "qkv_w_fp8": [25] * 16, "qkv_b": [26] * 16,
        "act_qkv_scale": [27] * 16, "w_qkv_scale": [28] * 16,
        "qkv_fp8_layout": fp8_layout,
        "ff_proj_w_fp8": [29] * 32, "ff_down_w_fp8": [30] * 32,
        "act_fc1_scale": [41] * 32, "act_fc2_scale": [42] * 32,
        "w_fc1_scale": [43] * 32, "w_fc2_scale": [44] * 32,
        "ff_fp8_layout": fp8_layout,
    }
    return pipeline_thor, gemm, fvk, bufs, weights, dims, attn


def test_groot_n17_dit_fp8_kn_layout_dispatches_nn_dev():
    pipeline_thor, gemm, fvk, bufs, weights, dims, attn = _make_dit_fp8_fixtures("kn")

    pipeline_thor.dit_forward(
        gemm=gemm, fvk=fvk, bufs=bufs, weights=weights,
        dims=dims, attn=attn, layers_subset=[1],
    )

    assert gemm.fp8_nn_dev.call_count == 3
    gemm.fp8_run_dev.assert_not_called()
    gemm.fp8_nt_dev.assert_not_called()


def test_groot_n17_dit_fp8_nk_layout_dispatches_nt_dev():
    pipeline_thor, gemm, fvk, bufs, weights, dims, attn = _make_dit_fp8_fixtures("nk")

    pipeline_thor.dit_forward(
        gemm=gemm, fvk=fvk, bufs=bufs, weights=weights,
        dims=dims, attn=attn, layers_subset=[1],
    )

    assert gemm.fp8_nt_dev.call_count == 3
    gemm.fp8_nn_dev.assert_not_called()
    gemm.fp8_run_dev.assert_not_called()
