#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full inference script for Meta-Llama-3-8B with LoRA adapters,
optimized for reduced hallucinations, mixed-precision, and maximal complete-sentence outputs.
"""
import os
import re
import torch
from torch import autocast
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig,
)
from peft import PeftModel

# -------------------------------------------------------------------------------
# FORCE GPU-0 ONLY
# -------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------
BASE_MODEL_NAME    = "meta-llama/Meta-Llama-3-8B"
ADAPTER_DIR        = "./llama3-qa-finetuned"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS     = 128
TEMPERATURE        = 0.7
TOP_K              = 50
TOP_P              = 0.9
REPETITION_PENALTY = 1.1

# -------------------------------------------------------------------------------
# QUANTIZATION CONFIG (Bits & Bytes)
# -------------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# -------------------------------------------------------------------------------
# CUSTOM STOPPING CRITERIA: stop on first newline
# -------------------------------------------------------------------------------
class StopOnNewline(StoppingCriteria):
    def __init__(self, stop_token_id: int):
        super().__init__()
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.stop_token_id

# -------------------------------------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------------------------------------
# LOAD MODEL + LoRA WITH QUANTIZATION CONFIG
# -------------------------------------------------------------------------------
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR, torch_dtype=torch.float16)
model.eval().to(DEVICE)

# Determine newline token ID and build stopping criteria
newline_token_id = tokenizer.convert_tokens_to_ids("\n")
stopping_criteria = StoppingCriteriaList([StopOnNewline(newline_token_id)])

# -------------------------------------------------------------------------------
# ANSWER FUNCTION
# -------------------------------------------------------------------------------
def answer(question: str) -> str:
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    seq_len = inputs["input_ids"].size(-1)

    with autocast(device_type="cuda", dtype=torch.float16):
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    # Decode and strip whitespace
    new_tokens = generated[0, seq_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Extract all complete sentences (ending in . ? or !) and join them
    sentences = re.findall(r".+?[\.\?!](?:\s|$)", text)
    if sentences:
        response = " ".join(sentences).strip()
    else:
        # Fallback: return full text
        response = text

    # Clear GPU cache if used
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return response

# -------------------------------------------------------------------------------
# INTERACTIVE LOOP
# -------------------------------------------------------------------------------
def main():
    print(f"Loaded LoRA adapter from '{ADAPTER_DIR}' on {DEVICE}. (type 'exit' to quit)\n")
    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in {"exit", "quit"}:
            break

        reply = answer(question)
        print(f"A> {reply}\n")

if __name__ == "__main__":
    main()
