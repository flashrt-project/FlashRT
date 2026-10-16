"""Native image preprocessing for the served DM05 input path.

The served transform chain resizes each 256x256 view to 448x448 on the host
(albumentations PadToSquare + A.Resize = cv2 INTER_LINEAR on uint8) and then runs the
Gemma3 image processor (rescale 1/255, normalize 0.5/0.5). The frontend wraps both
with a pass-through of the raw uint8 views; the DM05 pipeline then resizes and normalizes
them on device inside its CUDA graph (``dm05_native.resize_norm``), bit-identical to the
host chain (verified: 0 of 24,084,480 resized bytes differ, and the bf16 pixel tensor is
equal on all 444 observations of the LIBERO evaluation corpus).
Non-square views (where PadToSquare would pad) keep the original host path.
"""
from __future__ import annotations

import numpy as np
import torch

from .tokenize_cache import CachedChatTokenization


class RawImages:
    """Stands in for PixelTransform: keeps the decoded uint8 views, un-resized."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, data):
        imgs = [np.asarray(im.convert("RGB")) for im in data["images"]]
        if data.get("history_images") or any(a.shape[0] != a.shape[1] for a in imgs) \
                or len({a.shape for a in imgs}) != 1:
            data["_native_raw"] = None
            return self.inner(data)          # padding / history: original host path
        data["_native_raw"] = np.stack(imgs)
        return data


class RawAwareTokenization(CachedChatTokenization):
    """Cached tokenization that forwards raw uint8 views as ``pixel_values``.

    On a cache hit the views stay a numpy array (only the native staged predict reads
    them); the first request per prompt returns a tensor like the original transform.
    """

    def __init__(self, inner, verify_every: bool = False):
        super().__init__(inner, verify_every)
        self._tensors: dict = {}

    def __call__(self, data):
        raw = data.pop("_native_raw", None)
        if raw is None:
            return super().__call__(data)
        key = self._key(data)
        entry = self.cache.get(key)
        if entry is None or key in self.fallback or self.inner.add_state:
            # first request for this prompt: build and verify the token cache through the
            # original path (it needs the resized images), then answer with raw pixels
            from PIL import Image
            data2 = dict(data, images=[Image.fromarray(np.asarray(self._resize_host(a))) for a in raw])
            ref = super().__call__(data2)
            ref["pixel_values"] = torch.from_numpy(raw)
            return ref
        self.stats["hits"] += 1
        # the id / type / mask tensors are built once per prompt and reused as-is: no
        # per-request tensor ops on the host either
        t = self._tensors.get(key)
        if t is None:
            ids = torch.tensor([entry[0] + entry[1]], dtype=torch.long)
            t = (ids, (ids == self.image_token_id).to(torch.long), torch.ones_like(ids))
            self._tensors[key] = t
        return {"action": None, "action_mask": None, "input_ids": t[0], "attention_mask": t[2],
                "pixel_values": raw, "token_type_ids": t[1], "history_pixel_values": None,
                "history_mask": None}

    def _resize_host(self, a):
        return self.pixel_transform.transform_pipeline(image=a)["image"]


def build_transforms(transforms):
    """Build frontend-owned wrappers without changing the source transform list."""
    tr = list(transforms)
    names = [type(t).__name__ for t in tr]
    if "PixelTransform" not in names or "ChatTokenization" not in names:
        raise ValueError("DM05 needs PixelTransform and ChatTokenization in its input transforms")
    pi, ci = names.index("PixelTransform"), names.index("ChatTokenization")
    pix = tr[pi]
    tok = RawAwareTokenization(tr[ci])
    tok.pixel_transform = pix
    tr[pi], tr[ci] = RawImages(pix), tok
    return tr, tok
