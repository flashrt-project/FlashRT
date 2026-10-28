"""Checkpoint-free calibration identity and CTA profile regressions."""
import json
from pathlib import Path
import pytest
from flash_rt.models.lawam.cache_identity import file_sha256, processor_sha256
from flash_rt.models.lawam.pipeline_rtx import select_dit_num_ctas, validate_dit_num_ctas

@pytest.mark.parametrize("sms,expected", [(32,32),(36,36),(64,64),(68,64),(170,64)])
def test_cta_profile(sms, expected):
    assert select_dit_num_ctas(sms) == expected

@pytest.mark.parametrize("ctas", [0,20,24,28,31,33,65,170,True])
def test_invalid_cta_profile(ctas):
    with pytest.raises(ValueError, match="32..64"):
        validate_dit_num_ctas(ctas)

def test_weight_identity_covers_bytes_after_header(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"a" * (128 * 1024))
    before = file_sha256(weights)
    with weights.open("r+b") as handle:
        handle.seek(100 * 1024)
        handle.write(b"b")
    assert weights.stat().st_size == 128 * 1024
    assert file_sha256(weights) != before

class Processor:
    def __init__(self):
        self.vocab = {"a":1}
        self.image_config = {"size":256}
        self.template = "{{ messages }}"
    def save_pretrained(self, directory):
        root = Path(directory)
        (root / "tokenizer.json").write_text(json.dumps(self.vocab))
        (root / "preprocessor_config.json").write_text(json.dumps(self.image_config))
        (root / "chat_template.jinja").write_text(self.template)

@pytest.mark.parametrize("component", ["vocab","image_config","template"])
def test_processor_identity_tracks_contents(component):
    processor = Processor()
    before = processor_sha256(processor)
    assert processor_sha256(processor) == before
    setattr(processor, component, {"changed":1} if component != "template" else "changed")
    assert processor_sha256(processor) != before


def test_calibration_key_uses_weight_and_processor_contents(tmp_path, monkeypatch):
    from flash_rt.frontends.torch import lawam_rtx
    monkeypatch.setattr(lawam_rtx.torch.cuda, "get_device_capability", lambda _: (12, 0))
    frontend = lawam_rtx.LawamTorchFrontendRtx.__new__(lawam_rtx.LawamTorchFrontendRtx)
    frontend.checkpoint_path = str(tmp_path)
    frontend.calibration_dir = tmp_path
    frontend.precision, frontend.fp8_sites, frontend.device = {"vit":"fp8"}, {}, "cuda:0"
    frontend._processor = Processor()
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"a" * 131072)
    before = frontend._calibration_path()
    assert frontend._calibration_path() == before
    with weights.open("r+b") as handle:
        handle.seek(100000)
        handle.write(b"b")
    after_weights = frontend._calibration_path()
    assert after_weights != before
    frontend._processor.vocab["new"] = 2
    assert frontend._calibration_path() != after_weights
    assert before.name.endswith("_lawam_v2.json")


@pytest.mark.parametrize("sms", [24,28,31])
def test_device_below_attention_cta_minimum_is_rejected(sms):
    with pytest.raises(ValueError, match="32..64"):
        select_dit_num_ctas(sms)


@pytest.mark.parametrize("ctas", [24,28])
def test_packing_rejects_incomplete_attention_grid(ctas):
    from flash_rt.models.lawam.pipeline_rtx import pack_dit_persistent
    with pytest.raises(ValueError, match="32..64"):
        pack_dit_persistent(None, {}, ctas, "cuda")
