"""FlashRT -- GROOT N1.7 torch frontend for RTX.

This adapter keeps the N1.7 public contract from the Thor frontend while
switching the DiT attention backend to the RTX vendored FA2 implementation.
Backbone prompt processing and calibration remain shared with the N1.7
frontend, while DiT attention uses RTX-owned FA2 slots.
"""

from __future__ import annotations

import torch

from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor


class GrootN17TorchFrontendRtx(GrootN17TorchFrontendThor):
    """N1.7 RTX frontend.

    The RTX path avoids Thor's strided FMHA side-load and uses the RTX N1.7
    attention backend for DiT self/cross attention.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        num_views: int = 2,
        embodiment_tag: str = "oxe_droid_relative_eef_relative_joint",
        device: str = "cuda:0",
        load_strided_fmha: bool = False,
    ):
        super().__init__(
            checkpoint_path,
            num_views=num_views,
            embodiment_tag=embodiment_tag,
            device=device,
            load_strided_fmha=load_strided_fmha,
        )

    def infer(
        self,
        state_normalized: torch.Tensor,
        *,
        initial_noise=None,
        num_inference_timesteps: int = 4,
        action_horizon: int = 40,
        num_timestep_buckets: int = 1000,
        use_dit_graph: bool = True,
    ) -> torch.Tensor:
        return super().infer(
            state_normalized,
            initial_noise=initial_noise,
            num_inference_timesteps=num_inference_timesteps,
            action_horizon=action_horizon,
            num_timestep_buckets=num_timestep_buckets,
            use_dit_graph=use_dit_graph,
        )

    def _warmup_infer(self) -> None:
        warm_state = torch.zeros(1, 1, 132, dtype=torch.float32)
        torch.manual_seed(0)
        warm_noise = torch.randn(
            1, 40, 132, dtype=torch.bfloat16, device=self.device)
        _ = self.infer(warm_state, initial_noise=warm_noise, use_dit_graph=False)

    def _build_dit_attn(self, Sa: int) -> None:
        from flash_rt.hardware.rtx.attn_backend_groot_n17 import (
            RtxFlashAttnBackendGrootN17,
        )

        Skv_text = int(self._dit_cross_K[0].shape[0])
        Skv_image = int(self._dit_cross_K[1].shape[0])
        attn = RtxFlashAttnBackendGrootN17(
            num_vit_groups=int(getattr(self, "_num_vit_views", 4)),
            llm_seq_max=int(self.Se),
            vl_self_attn_seq_max=int(self.Se),
            sa=int(Sa),
            s_kv_text=Skv_text,
            s_kv_image=Skv_image,
            device=self.device,
        )

        # Cross K/V are precomputed by the shared N1.7 frontend as exact-size
        # tensors.  RTX backend owns padded-to-max slots so pipeline pointers
        # stay stable; copy the active prefix for each cross block.
        for j, (k_src, v_src) in enumerate(zip(self._dit_cross_K, self._dit_cross_V)):
            attn.dit_cross_K[j].view(attn.dit_cross_K[j].shape[0], -1)[
                : k_src.shape[0]
            ].copy_(k_src)
            attn.dit_cross_V[j].view(attn.dit_cross_V[j].shape[0], -1)[
                : v_src.shape[0]
            ].copy_(v_src)
        self._dit_attn = attn

    def _precompute_dit_cross_kv(self) -> None:
        super()._precompute_dit_cross_kv()
        # Prompt changes produce new exact-size K/V tensors.  Drop any RTX
        # attention slots/graphs so the next infer copies the fresh K/V.
        for name in ("_dit_attn", "_dit_graphs"):
            if hasattr(self, name):
                delattr(self, name)

    def _run_dit(self, bufs: dict, shift_list, scale_list, Sa: int) -> None:
        from flash_rt.models.groot_n17 import pipeline_rtx

        if not hasattr(self, "_dit_attn"):
            self._build_dit_attn(Sa)

        weights = {
            "scale_msa": [t.data_ptr() for t in scale_list],
            "shift_msa": [t.data_ptr() for t in shift_list],
            "q_w": [w.data_ptr() for w in self._dit_q_w],
            "q_b": [b.data_ptr() for b in self._dit_q_b],
            "k_w": [w.data_ptr() for w in self._dit_k_w],
            "k_b": [b.data_ptr() for b in self._dit_k_b],
            "v_w": [w.data_ptr() for w in self._dit_v_w],
            "v_b": [b.data_ptr() for b in self._dit_v_b],
            "o_w": [w.data_ptr() for w in self._dit_o_w],
            "o_b": [b.data_ptr() for b in self._dit_o_b],
            "ff_proj_w": [w.data_ptr() for w in self._dit_ff_proj_w],
            "ff_proj_b": [b.data_ptr() for b in self._dit_ff_proj_b],
            "ff_down_w": [w.data_ptr() for w in self._dit_ff_down_w],
            "ff_down_b": [b.data_ptr() for b in self._dit_ff_down_b],
        }
        Skv_text = int(self._dit_cross_K[0].shape[0])
        Skv_image = int(self._dit_cross_K[1].shape[0])
        dims = {
            "Sa": int(Sa),
            "D": 1536,
            "FF": 6144,
            "Skv_text": Skv_text,
            "Skv_image": Skv_image,
        }
        bufs_ptrs = {
            "h": bufs["dit_h"].data_ptr(),
            "xn": bufs["dit_xn"].data_ptr(),
            "o_proj_out": bufs["dit_o_proj_out"].data_ptr(),
            "ff_proj_out": bufs["dit_ff_proj_out"].data_ptr(),
        }
        if not hasattr(self, "_gemm"):
            import flash_rt.flash_rt_kernels as _fvk
            self._fvk = _fvk
            self._gemm = _fvk.GemmRunner()

        pipeline_rtx.dit_forward(
            gemm=self._gemm,
            fvk=self._fvk,
            bufs=bufs_ptrs,
            weights=weights,
            dims=dims,
            attn=self._dit_attn,
        )
