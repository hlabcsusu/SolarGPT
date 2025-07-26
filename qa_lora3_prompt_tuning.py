#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA app for Meta-LLaMA-3-8B with LoRA adapters and K-12 storytelling prompt tuning.
"""

import os
import re
import torch
from torch import autocast
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel

# -----------------------------------------------------------------------------
# GPU & Memory Settings
# -----------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_MODEL_NAME     = "meta-llama/Meta-Llama-3-8B"
ADAPTER_DIR         = "./llama3-qa-finetuned"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS      = 128
TEMPERATURE         = 0.7
TOP_K               = 50
TOP_P               = 0.9
REPETITION_PENALTY  = 1.1

# -----------------------------------------------------------------------------
# Quantization Configuration (BitsAndBytes 8-bit)
# -----------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# -----------------------------------------------------------------------------
# Custom Stopping Criteria: Stop at newline
# -----------------------------------------------------------------------------
class StopOnNewline(StoppingCriteria):
    def __init__(self, stop_token_id: int):
        super().__init__()
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.stop_token_id

# -----------------------------------------------------------------------------
# Load Tokenizer and Base Model
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

newline_token_id = tokenizer.convert_tokens_to_ids("\n")
stopping_criteria = StoppingCriteriaList([StopOnNewline(newline_token_id)])

# -----------------------------------------------------------------------------
# Prompt Engineering: System Instruction + Few-Shot Examples
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful science guide for curious kids and teens.
- Answer in 2-3 fun and easy-to-understand sentences.
- Use real facts, simple words, and a storytelling style.
- Make learning exciting but don't make things up.
"""

FEW_SHOT = """Example 1:
Q: How do sunspot groups evolve to form active regions that are capable of producing major solar flares?
A: Sunspots are like magnetic storms on the Sun's surface that grow and twist over time. When these storms become really tangled, they can snap and release huge bursts of energy called solar flares. It's kind of like stretching a rubber band until it breaks!

Example 2:
Q: What is the significance of magnetic helicity in predicting the eruptive potential of a solar active region?
A: Magnetic helicity tells us how twisted the Sun's magnetic fields are. When there's a lot of twist, it means energy is being stored—like winding up a toy. If the twist gets too strong, it can suddenly burst out as a solar flare or explosion!

Example 3:
Q: Why is it important to study the time delay between the onset of a solar flare and the associated coronal mass ejection?
A: Watching when a flare and a solar blast (CME) happen tells scientists which one caused the other. If the flare comes first, it might be the trigger; if the blast starts first, it could be the main event. It's like figuring out who pushed the first domino!

Example 4:
Q: How can differential emission measure (DEM) analysis enhance our understanding of flare heating and cooling processes?
A: DEM is like using special glasses to see how hot different parts of the Sun are during a flare. It helps scientists track how fast things heat up and cool down. That way, we learn what's really happening during those big solar fireworks!

Example 5:
Q: What role do quasi-separatrix layers (QSLs) play in three-dimensional models of magnetic reconnection during solar flares?
A: QSLs are invisible zones where the Sun's magnetic lines suddenly change direction. These spots often spark solar flares by helping magnetic fields snap and reconnect. Imagine them as twisty paths where solar energy gets ready to explode!
"""

# -----------------------------------------------------------------------------
# Answer Generation Function
# -----------------------------------------------------------------------------
def answer(question: str) -> str:
    prompt = SYSTEM_PROMPT + "\n" + FEW_SHOT + f"\nNow answer:\nQ: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
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

    new_tokens = generated[0, seq_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    sentences = re.findall(r".+?[\.!?](?:\s|$)", text)
    response = " ".join(sentences).strip() if sentences else text

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return response

# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------
def main():
    print(f"\n🛰️  LLaMA-3 QA App is ready on {DEVICE} with LoRA from '{ADAPTER_DIR}'!\n(Type 'exit' or Ctrl+C to quit)\n")
    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in {"exit", "quit"}:
            break

        print(f"A> {answer(question)}\n")

if __name__ == "__main__":
    main()
