def test_detect_arch_returns_rocm_on_hip_runtime():
    from flash_rt.hardware import detect_arch

    assert detect_arch() == "rocm"


def test_resolve_pi05_torch_rocm_frontend_class():
    from flash_rt.hardware import resolve_pipeline_class
    from flash_rt.frontends.torch.pi05_rocm import Pi05TorchFrontendRocm

    cls = resolve_pipeline_class("pi05", "torch", "rocm")
    assert cls is Pi05TorchFrontendRocm


def test_pi05_rocm_weight_helper_imports():
    from flash_rt.frontends.torch.pi05_rocm_weights import (
        build_rocm_vision_weights_from_openpi_model,
        weight_ptr,
    )

    assert callable(build_rocm_vision_weights_from_openpi_model)
    assert callable(weight_ptr)
