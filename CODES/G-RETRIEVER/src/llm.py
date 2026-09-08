import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llm(model_id: str, device: str = "cuda:1"):
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    return model, tokenizer


@torch.no_grad()
def generate_letters_batch(prompts, llm, tokenizer, batch_size: int = 16, max_new_tokens: int = 5):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raws, preds = [], []
    for i in tqdm(range(0, len(prompts), batch_size), desc="generate"):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt",
            padding=True, truncation=True, max_length=8192,
        ).to(llm.device)

        out = llm.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        for seq in gen:
            raw = tokenizer.decode(seq, skip_special_tokens=True).strip()
            m = re.search(r"\b([ABCD])\b", raw.upper())
            raws.append(raw)
            preds.append(m.group(1) if m else None)
    return raws, preds
