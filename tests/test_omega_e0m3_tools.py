import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


converter = _load_tool("convert_omega_pack_e0m3")
checker = _load_tool("check_omega_e0m3_layer")


def _record(n: int = 64, k: int = 64) -> dict:
    return {
        "format": "dit_svdquant_v1",
        "weight_res_q": torch.randn(n, k, dtype=torch.float16),
        "lowrank_A": torch.empty(n, 0, dtype=torch.float16),
        "lowrank_B": torch.empty(k, 0, dtype=torch.float16),
        "act_scale_table": torch.ones(2, k),
        "duquant_rotation_blocks": torch.eye(64).half().view(1, 64, 64),
        "duquant_rotation_perm": torch.arange(k, dtype=torch.int64),
        "duquant_rotation_out_blocks": torch.eye(64).half().view(1, 64, 64),
        "weight_bits": 4,
        "a_bits": 4,
        "rank": 0,
        "in_features": k,
        "out_features": n,
    }


def test_converter_accepts_complete_rank_zero_record():
    assert converter.validate_record("layer", _record()) == (64, 64)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r: r.update(format="other"), "unsupported format"),
        (lambda r: r.update(rank=1), "only rank=0"),
        (lambda r: r.update(in_features=128), "metadata shape mismatch"),
        (lambda r: r.update(act_scale_table=torch.ones(2, 32)),
         "act_scale_table"),
        (lambda r: r.update(duquant_rotation_blocks=torch.eye(64).half()),
         "input rotation shape"),
        (lambda r: r.update(duquant_rotation_perm=torch.zeros(64, dtype=torch.int64)),
         "not a permutation"),
    ],
)
def test_converter_rejects_incomplete_or_inconsistent_records(mutation, message):
    record = copy.deepcopy(_record())
    mutation(record)
    with pytest.raises(ValueError, match=message):
        converter.validate_record("layer", record)


def test_full_pack_count_defaults_to_252_but_fixture_and_subset_are_exact():
    assert converter.expected_record_count(
        {}, layer_regex="", selected_count=251, explicit=None) == 252
    assert converter.expected_record_count(
        {"recipe": "synthetic fixture (test)"}, layer_regex="", selected_count=4,
        explicit=None) == 4
    assert converter.expected_record_count(
        {}, layer_regex="layer.0", selected_count=1, explicit=None) == 1


def test_validate_only_runs_without_cuda_and_writes_no_artifact(tmp_path):
    pack_path = tmp_path / "fixture_pack.pt"
    out_path = tmp_path / "must_not_exist.pt"
    torch.save({
        "__meta__": {"recipe": "synthetic fixture (test)"},
        "layer": _record(),
    }, pack_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "convert_omega_pack_e0m3.py"),
            "--pack", str(pack_path),
            "--out", str(out_path),
            "--validate-only",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "validated 1/1 records" in result.stdout
    assert not out_path.exists()


def test_validate_only_fails_before_cuda_for_bad_records(tmp_path):
    pack_path = tmp_path / "bad_pack.pt"
    out_path = tmp_path / "must_not_exist.pt"
    record = _record()
    record["rank"] = 1
    torch.save({
        "__meta__": {"recipe": "synthetic fixture (test)"},
        "layer": record,
    }, pack_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "convert_omega_pack_e0m3.py"),
            "--pack", str(pack_path),
            "--out", str(out_path),
            "--validate-only",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "only rank=0 is supported" in result.stderr
    assert not out_path.exists()


def test_converter_refuses_to_overwrite_source_pack(tmp_path):
    pack_path = tmp_path / "fixture_pack.pt"
    torch.save({
        "__meta__": {"recipe": "synthetic fixture (test)"},
        "layer": _record(),
    }, pack_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "convert_omega_pack_e0m3.py"),
            "--pack", str(pack_path),
            "--out", str(pack_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "must not overwrite" in result.stderr
    loaded = torch.load(pack_path, map_location="cpu", weights_only=True)
    assert "layer" in loaded


def test_artifact_schema_requires_complete_direct_packed_sfb_coverage():
    class FakeFp4:
        @staticmethod
        def sfa_size_bytes(n, k, is_sfb):
            assert is_sfb
            return 32

    artifact = {
        "format": "omega_e0m3_v1",
        "schema_version": 1,
        "fold": "none",
        "selected_record_count": 1,
        "selected_layers": ["layer"],
        "weights": {
            "layer": {
                "packed": torch.zeros(64, 32, dtype=torch.uint8),
                "sfb": torch.zeros(32, dtype=torch.uint8),
                "N": 64,
                "K": 64,
            }
        },
        "aux": {"layer": {"fold": "none"}},
    }
    entry, aux = checker.validate_artifact(
        artifact, "layer", 64, 64, FakeFp4())
    assert entry["packed"].shape == (64, 32)
    assert aux["fold"] == "none"

    broken = copy.deepcopy(artifact)
    broken["aux"] = {}
    with pytest.raises(ValueError, match="do not cover"):
        checker.validate_artifact(broken, "layer", 64, 64, FakeFp4())


def test_docs_and_gitignore_only_describe_milestone_one_surface():
    docs = (ROOT / "docs" / "omega_pack_e0m3.md").read_text()
    gitignore = (ROOT / ".gitignore").read_text().splitlines()

    for nonexistent in (
        "omega_e0m3_linear.py",
        "check_omega_e0m3_consumer.py",
        "serve_omega_e0m3.py",
        "omega_e0m3_graph.py",
        "check_omega_e0m3_graph_smoke.py",
        "start_e0m3_server.sh",
    ):
        assert nonexistent not in docs
    assert "*.pt" not in gitignore
    assert "/artifacts/omega_e0m3/*.pt" in gitignore
