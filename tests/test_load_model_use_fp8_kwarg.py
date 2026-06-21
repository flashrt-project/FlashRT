import ast
from pathlib import Path
import sys
import types
from unittest.mock import patch

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
        "fp4_layers": tuple(range(18)),
        "use_awq": True,
        "awq_alpha": 0.5,
        "use_p1_split_gu": True,
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


def test_pi05_rtx_fp8_layout_selection():
    from flash_rt.frontends.torch.pi05_rtx import _select_fp8_layout

    assert _select_fp8_layout("rtx_sm89", None) == "nk"
    assert _select_fp8_layout("rtx_sm120", None) == "kn"
    assert _select_fp8_layout("rtx_sm120", "nk") == "nk"


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
