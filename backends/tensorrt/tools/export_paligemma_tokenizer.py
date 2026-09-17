#!/usr/bin/env python3
"""Write a Hugging Face tokenizer.json for the PaliGemma SentencePiece model
(openpi's big_vision/paligemma_tokenizer.model), for C++ runtimes that load
tokenizer.json, and check it tokenizes like SentencePiece.

usage: export_paligemma_tokenizer.py <paligemma_tokenizer.model> <out_dir>
"""
import sys

import sentencepiece
from transformers import GemmaTokenizerFast

model_path, out_dir = sys.argv[1:3]
fast = GemmaTokenizerFast(vocab_file=model_path, add_bos_token=False, add_eos_token=False)
fast.save_pretrained(out_dir)
sp = sentencepiece.SentencePieceProcessor(model_file=model_path)
texts = ["pick up the red block and place it in the tray", "put the bowl on the plate",
         "open the top drawer of the cabinet and put the bowl inside", "turn on the stove",
         "put both the alphabet soup and the tomato sauce in the basket", "\n",
         "Task: close the microwave, State: 12 255 0;\nAction: "]
ok = True
for t in texts:
    a, b = sp.encode(t), fast.encode(t, add_special_tokens=False)
    ok &= a == b
    if a != b:
        print("MISMATCH", repr(t), a, b)
print("bos", sp.bos_id(), fast.bos_token_id, "| all texts match:", ok)
