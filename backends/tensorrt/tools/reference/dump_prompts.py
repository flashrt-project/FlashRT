"""prompt-dynamic references for the pi0.5 policy engine.

With one calibrated FlashRT pipeline (calibrated on the default prompt), for
each prompt: set the prompt, run the library SigLIP graph, encoder and decoder
from pinned noise on observation 0, and record the raw actions. Also writes the
tables a prompt-dynamic engine needs: the token embedding table and the RoPE
table, and checks that the pipeline's prompt embeddings and encoder/decoder
RoPE rows are exactly what the engine derives from them:

  prompt embeddings = fp16(fp32(embedding[tokens]) * sqrt(D))
  encoder rope      = table[0:Se],   decoder rope = table[Se:Se + 10]

FlashRT keeps Se even by repeating the last prompt embedding; the engine gets
the same by repeating the last token (engine_tokens).

FlashRT's set_prompt re-derives the static FP8 activation scales for a new
prefix length (cache or warmup data); an engine keeps one calibration for all
prompts. The reference therefore restores the real-data calibration from the
first prompt after every set_prompt, so both sides run the same model.
"""
import argparse
import os
import sys

import sentencepiece
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi05_pipeline as m1  # noqa: E402
from flash_rt.utils.paligemma_tokenizer import resolve_paligemma_tokenizer_path  # noqa: E402

LIBERO_PROMPTS = (
    "pick up the red block and place it in the tray",
    "put the bowl on the plate",
    "open the top drawer of the cabinet and put the bowl inside",
    "turn on the stove",
)


def main():
    p = argparse.ArgumentParser()
    m1.add_policy_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    enc_calls, dec_calls = [], []
    enc_orig = m1.fe_mod.encoder_forward_with_fp4_subset
    dec_orig = m1.fe_mod.decoder_forward_fp4

    def enc_rec(*a, **k):
        if k.get("stream", 0) == 0:
            enc_calls.append((a, k))
        return enc_orig(*a, **k)

    def dec_rec(*a, **k):
        if k.get("stream", 0) == 0:
            dec_calls.append((a, k))
        return dec_orig(*a, **k)

    m1.fe_mod.encoder_forward_with_fp4_subset = enc_rec
    m1.fe_mod.decoder_forward_fp4 = dec_rec
    pipe, obs = m1.build_pipeline(args)
    sp = sentencepiece.SentencePieceProcessor(model_file=resolve_paligemma_tokenizer_path())

    D = pipe.embedding_weight.shape[1]
    table = torch.cat([pipe._kc_t[:, :, None], pipe._ks_t[:, :, None]], dim=2).reshape(pipe._kc_t.shape[0], 256)
    S_sig = pipe.sig_S
    g = torch.Generator().manual_seed(args.seed)
    noise_in = torch.randn(10, 32, generator=g).half().cuda()
    calib = (pipe._enc_calib_scales.clone(), list(pipe._enc_alpha_host), pipe._ae_calib_scales.clone())
    out = {"embedding": pipe.embedding_weight.contiguous(), "rope_table": table.contiguous(),
           "noise_in": noise_in}

    for i, text in enumerate(LIBERO_PROMPTS):
        tokens = [sp.bos_id()] + sp.encode(text) + sp.encode("\n")
        if i == 0:
            assert tokens == m1.PROMPT_TOKENS
        # A prompt of an already captured length is updated in place without
        # recapture, so the last recorded arguments (same buffers) still apply.
        pipe.set_prompt(tokens)
        pipe._enc_calib_scales.copy_(calib[0])
        pipe._enc_alpha_host[:] = calib[1]
        pipe._ae_calib_scales.copy_(calib[2])
        pipe.infer(obs[0])
        torch.cuda.synchronize()
        (ea, ek), (da, dk) = enc_calls[-1], dec_calls[-1]
        assert ea[4]["alpha_host"] is pipe._enc_alpha_host and ea[4]["alpha_host"] == calib[1]
        Se = int(ea[5]["Se"])
        engine_tokens = tokens + ([tokens[-1]] if (S_sig + len(tokens)) % 2 else [])
        assert S_sig + len(engine_tokens) == Se

        emb = (pipe.embedding_weight[torch.tensor(engine_tokens, device="cuda")].float() * D ** 0.5).half()
        checks = {
            "prompt embeddings": torch.equal(pipe._lang_emb[:Se - S_sig], emb),
            "encoder rope": torch.equal(pipe._enc_rope[:Se], table[:Se]),
            "decoder rope": torch.equal(pipe._dec_rope, table[Se:Se + 10]),
        }

        m1.upload_views(pipe, obs[0], args.num_views)
        pipe._siglip_u8_graph.replay()
        torch.cuda.synchronize()
        checks["encoder input prompt rows"] = torch.equal(pipe._enc_x[S_sig:Se], emb)
        pipe._Kc.zero_(); pipe._Vc.zero_()
        enc_orig(*ea, **ek)
        pipe._g_noise.view(10, 32).copy_(noise_in)
        dec_orig(*da, **dk)
        torch.cuda.synchronize()
        out[f"p{i}.tokens"] = torch.tensor(engine_tokens, dtype=torch.int32)
        out[f"p{i}.actions"] = pipe._g_noise.view(10, 32).clone()
        print(f"prompt {i} {text!r}: {len(tokens)} tokens, Se={Se}, checks {checks}")
        if not all(checks.values()):
            sys.exit(1)

    out = {n: t.detach().contiguous().cpu() for n, t in out.items()}
    save_file(out, args.out)
    print("wrote", args.out, "prompts", len(LIBERO_PROMPTS))


if __name__ == "__main__":
    main()
