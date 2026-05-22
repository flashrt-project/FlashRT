    m.def("lingbot_ada_rms_residual_fp8_bf16", &lingbot_ada_rms_residual_fp8_bf16,
          py::arg("residual"), py::arg("x"),
          py::arg("rms_weight"), py::arg("gamma"), py::arg("beta"),
          py::arg("out_fp8"), py::arg("act_scale"),
          py::arg("B"), py::arg("S"), py::arg("D"),
          py::arg("eps") = 1e-6f, py::arg("stream") = 0,
          "Per-sample γ/β AdaRMSNorm + residual + FP8 static quant fused.");

    m.def("lingbot_ada_rms_fp8_bf16", &lingbot_ada_rms_fp8_bf16,
          py::arg("x"), py::arg("rms_weight"),
          py::arg("gamma"), py::arg("beta"),
          py::arg("out_fp8"), py::arg("act_scale"),
          py::arg("B"), py::arg("S"), py::arg("D"),
          py::arg("eps") = 1e-6f, py::arg("stream") = 0,
          "Per-sample γ/β AdaRMSNorm + FP8 static quant fused (no residual).");

    m.def("lingbot_rope_inplace_bf16", &lingbot_rope_inplace_bf16,
          py::arg("x"), py::arg("cos_table"), py::arg("sin_table"),
          py::arg("B"), py::arg("S"), py::arg("NH"), py::arg("HD"),
          py::arg("stream") = 0,
          "In-place split-half RoPE for bf16 Q or K tensors. cos/sin "
          "are pre-built fp32 tables [B, S, HD/2].");

    m.def("lingbot_rope_to_out_bf16", &lingbot_rope_to_out_bf16,
          py::arg("x"), py::arg("out"),
          py::arg("cos_table"), py::arg("sin_table"),
          py::arg("B"), py::arg("S"), py::arg("NH"), py::arg("HD"),
          py::arg("stream") = 0,
          "Out-of-place split-half RoPE (bf16): reads x, writes out. "
          "Graph-safe (no in-place hazard). cos/sin fp32 [B, S, HD/2].");

    m.def("lingbot_qkv_bias_rope_bf16", &lingbot_qkv_bias_rope_bf16,
          py::arg("q_raw"), py::arg("k_raw"), py::arg("v_raw"),
          py::arg("q_bias"), py::arg("k_bias"), py::arg("v_bias"),
          py::arg("cos_table"), py::arg("sin_table"),
          py::arg("q_out"), py::arg("k_out"), py::arg("v_out"),
          py::arg("M"), py::arg("NHQ"), py::arg("NHKV"), py::arg("HD"),
          py::arg("stream") = 0,
          "Fused QKV bias-add + RoPE (q/k roped, v bias-only). One launch "
          "replaces 3 add_bias + 2 rope. Raw bf16 GEMM outputs in, "
          "[M, NH*HD] bf16 out; cos/sin fp32 [M, HD/2]. Pass 0 for a null bias.");

    m.def("lingbot_qkv_bias_rope_fp16out", &lingbot_qkv_bias_rope_fp16out,
          py::arg("q_raw"), py::arg("k_raw"), py::arg("v_raw"),
          py::arg("q_bias"), py::arg("k_bias"), py::arg("v_bias"),
          py::arg("cos_table"), py::arg("sin_table"),
          py::arg("q_out"), py::arg("k_out"), py::arg("v_out"),
          py::arg("M"), py::arg("NHQ"), py::arg("NHKV"), py::arg("HD"),
          py::arg("stream") = 0,
          "G39: same as qkv_bias_rope_bf16 but writes fp16 outputs (feeds the "
          "fp16 attention island — fmha consumes fp16 with no cast).");

    m.def("lingbot_ada_rms_fp8_mpad_bf16", &lingbot_ada_rms_fp8_mpad_bf16,
          py::arg("x"), py::arg("rms_weight"),
          py::arg("gamma"), py::arg("beta"),
          py::arg("out_fp8"), py::arg("act_scale"),
          py::arg("B"), py::arg("S"), py::arg("PADM"), py::arg("D"),
          py::arg("eps") = 1e-6f, py::arg("stream") = 0,
          "M-padded ada_rms_fp8: writes [B,PADM,D] FP8 (rows [S,PADM) zeroed) "
          "so q/k/v GEMMs read M=PADM directly (no pad copy).");

    m.def("lingbot_ada_rms_residual_fp8_mpad_bf16", &lingbot_ada_rms_residual_fp8_mpad_bf16,
          py::arg("residual"), py::arg("x"),
          py::arg("rms_weight"), py::arg("gamma"), py::arg("beta"),
          py::arg("out_fp8"), py::arg("act_scale"),
          py::arg("B"), py::arg("S"), py::arg("PADM"), py::arg("D"),
          py::arg("eps") = 1e-6f, py::arg("stream") = 0,
          "M-padded ada_rms_residual_fp8: writes [B,PADM,D] FP8 (rows [S,PADM) "
          "zeroed) so gate/up GEMMs read M=PADM directly (no pad copy).");

    m.def("lingbot_ada_rms_residual_fp16_mpad", &lingbot_ada_rms_residual_fp16_mpad,
          py::arg("residual"), py::arg("x"),
          py::arg("rms_weight"), py::arg("gamma"), py::arg("beta"),
          py::arg("out_fp16"),
          py::arg("B"), py::arg("S"), py::arg("PADM"), py::arg("D"),
          py::arg("eps") = 1e-6f, py::arg("stream") = 0,
          "G53: M-padded ada_rms_residual with fp16 output (no FP8 quant) — feeds "
          "the denoise FP4 gate_up quant directly, skipping the fp8->fp16 dequant.");

    m.def("lingbot_silu_mul_fp8_mpad_bf16", &lingbot_silu_mul_fp8_mpad_bf16,
          py::arg("gate"), py::arg("up"), py::arg("out_fp8"), py::arg("act_scale"),
          py::arg("S"), py::arg("PADM"), py::arg("I"), py::arg("stream") = 0,
          "Fused SwiGLU tail: silu(gate)*up + FP8 static-quant, writing a "
          "pre-M-padded [PADM, I] FP8 buffer (rows [S,PADM) zeroed) so the "
          "down GEMM reads M=PADM directly with no pad copy. bf16 gate/up in.");

    m.def("lingbot_silu_mul_merged_fp8_mpad_bf16", &lingbot_silu_mul_merged_fp8_mpad_bf16,
          py::arg("gu"), py::arg("out_fp8"), py::arg("act_scale"),
          py::arg("S"), py::arg("PADM"), py::arg("I"), py::arg("stream") = 0,
          "Merged-input SwiGLU tail: gate/up interleaved per row in one [S,2*I] "
          "buffer (single merged gate_up GEMM output); silu(gate)*up + FP8 "
          "static-quant -> [PADM, I]. Pairs with a merged gate_up GEMM to cut a "
          "GEMM launch + the silu/mul/quant glue.");

    m.def("lingbot_silu_mul_merged_fp8_mpad_fp16in", &lingbot_silu_mul_merged_fp8_mpad_fp16in,
          py::arg("gu"), py::arg("out_fp8"), py::arg("act_scale"),
          py::arg("S"), py::arg("PADM"), py::arg("I"), py::arg("stream") = 0,
          "G54: fp16-input silu_mul_merged — reads gate_up from the FP4 GEMM "
          "fp16 output directly (no bf16 cast).");

    m.def("lingbot_qkv_bias_rope_merged_bf16", &lingbot_qkv_bias_rope_merged_bf16,
          py::arg("qkv"), py::arg("q_bias"), py::arg("k_bias"), py::arg("v_bias"),
          py::arg("cos_table"), py::arg("sin_table"),
          py::arg("q_out"), py::arg("k_out"), py::arg("v_out"),
          py::arg("M"), py::arg("NHQ"), py::arg("NHKV"), py::arg("HD"),
          py::arg("stream") = 0,
          "G49 merged-input qkv bias+RoPE: reads q/k/v from column offsets of "
          "one merged [M, ROWQKV] GEMM output (no split); writes 3 separate "
          "bf16 outputs. Pairs with a merged qkv GEMM to cut 2 GEMM launches.");

    m.def("lingbot_qkv_bias_rope_merged_fp16out", &lingbot_qkv_bias_rope_merged_fp16out,
          py::arg("qkv"), py::arg("q_bias"), py::arg("k_bias"), py::arg("v_bias"),
          py::arg("cos_table"), py::arg("sin_table"),
          py::arg("q_out"), py::arg("k_out"), py::arg("v_out"),
          py::arg("M"), py::arg("NHQ"), py::arg("NHKV"), py::arg("HD"),
          py::arg("stream") = 0,
          "G49 merged-input qkv bias+RoPE, fp16 outputs (fp16 attention island).");

    m.def("lingbot_vit_qkv_bias_rope_bf16", &lingbot_vit_qkv_bias_rope_bf16,
          py::arg("qkv"), py::arg("qkv_bias"),
          py::arg("cos_table"), py::arg("sin_table"),
          py::arg("q_out"), py::arg("k_out"), py::arg("v_out"),
          py::arg("M"), py::arg("NH"), py::arg("HD"),
          py::arg("stream") = 0,
          "P1 ViT: fused QKV bias-add + 2-D M-RoPE from the interleaved "
          "[M, 3*NH*HD] ViT qkv GEMM output. q/k get bias+split-half RoPE, v "
          "bias-only; writes 3×[M, NH*HD] bf16. cos/sin fp32 [M, HD] (first "
          "HD/2 cols used). One launch replaces add_bias + the eager fp32 "
          "RoPE storm. Pass 0 for a null qkv_bias.");
