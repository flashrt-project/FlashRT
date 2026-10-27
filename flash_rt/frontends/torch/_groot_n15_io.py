"""FlashRT -- GR00T N1.5 observation / action IO through the upstream transforms.

The policy's pre- and post-processing -- video crop/resize, state
normalization and rotation conversion, the Eagle2.5 chat template, tokenizer
and image processor, action unnormalization -- is the upstream Isaac-GR00T
data-config transform, used here unchanged so the model sees exactly the
tensors the reference policy would. Only the transforms are used; the
upstream model is never built.

``gr00t`` is imported lazily when an adapter is constructed. Its source root
can be given explicitly or through ``$GROOT_N15_ROOT``; it must provide the
named data config (``DATA_CONFIG_MAP``).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

ENV_ROOT = "GROOT_N15_ROOT"
ENV_DATA_CONFIG = "GROOT_N15_DATA_CONFIG"


def _import_gr00t(groot_root: str | None) -> Any:
    root = groot_root or os.environ.get(ENV_ROOT)
    if root:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(os.path.join(root, "gr00t")):
            raise FileNotFoundError(f"{root} has no gr00t package (Isaac-GR00T source root)")
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        import gr00t
    except ImportError as exc:
        raise RuntimeError(
            "GR00T N1.5 IO needs the upstream gr00t package (Isaac-GR00T): pass "
            f"groot_root or set ${ENV_ROOT} to its source root, or install it: {exc}"
        ) from exc
    return gr00t


class GrootN15IO:
    """One data config + embodiment: observation -> model inputs, actions -> dict."""

    def __init__(self, checkpoint_dir: str, data_config: str | None,
                 embodiment_tag: str, groot_root: str | None = None) -> None:
        data_config = data_config or os.environ.get(ENV_DATA_CONFIG)
        if not data_config:
            raise ValueError(
                "GR00T N1.5 needs a data config name (Isaac-GR00T DATA_CONFIG_MAP key): "
                f"pass data_config or set ${ENV_DATA_CONFIG}")
        gr00t = _import_gr00t(groot_root)
        from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
        from gr00t.data.schema import DatasetMetadata
        from gr00t.experiment.data_config import DATA_CONFIG_MAP

        if data_config not in DATA_CONFIG_MAP:
            raise ValueError(f"unknown data_config {data_config!r}; available: "
                             f"{sorted(DATA_CONFIG_MAP)}")
        cfg = DATA_CONFIG_MAP[data_config]
        self.modality = cfg.modality_config()
        self._transform = cfg.transform()
        self._transform.eval()
        tag = EmbodimentTag(embodiment_tag)
        meta_path = os.path.join(checkpoint_dir, "experiment_cfg", "metadata.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"no experiment_cfg/metadata.json in {checkpoint_dir}")
        with open(meta_path) as f:
            metas = json.load(f)
        if tag.value not in metas:
            raise ValueError(f"checkpoint metadata has no embodiment {tag.value!r}; "
                             f"available: {sorted(metas)}")
        self._transform.set_metadata(DatasetMetadata.model_validate(metas[tag.value]))
        self.embodiment_tag = tag.value
        self.embodiment_id = int(EMBODIMENT_TAG_MAPPING[tag.value])
        self.video_keys = tuple(self.modality["video"].modality_keys)
        self.state_keys = tuple(self.modality["state"].modality_keys)
        self.action_keys = tuple(self.modality["action"].modality_keys)
        langs = tuple(self.modality["language"].modality_keys) if "language" in self.modality else ()
        if len(langs) != 1:
            raise ValueError(f"data config {data_config!r} has language keys {langs}; "
                             "exactly one is supported")
        self.language_key = langs[0]
        if len(self.modality["video"].delta_indices) != 1 or \
                len(self.modality["state"].delta_indices) != 1:
            raise ValueError("only single-frame video and state observations are supported")
        self.action_horizon = len(self.modality["action"].delta_indices)
        self.eagle_config_path = os.path.join(
            os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model",
            "config.json")

    def eagle_config(self) -> dict:
        with open(self.eagle_config_path) as f:
            return json.load(f)

    def image_token_id(self) -> int:
        return int(self.eagle_config()["image_token_index"])

    def prepare(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Unbatched observation (``video.*`` ``[1, H, W, 3]``, ``state.*``
        ``[1, D]``, the language key ``[prompt]``) -> normalized model inputs,
        as ``Gr00tPolicy.get_action`` builds them."""
        batched = {}
        for k, v in obs.items():
            v = v if isinstance(v, np.ndarray) else np.array(v)
            batched[k] = np.expand_dims(v, axis=0)
        return self._transform(batched)

    def unnormalize(self, actions: Any) -> dict[str, np.ndarray]:
        """Normalized actions ``[1, H, action_dim]`` (torch, CPU) -> the
        per-key action dict, unbatched."""
        out = self._transform.unapply({"action": actions})
        return {k: np.squeeze(v, axis=0) if isinstance(v, np.ndarray) else v
                for k, v in out.items()}
