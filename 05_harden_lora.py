"""Train-time fix: LoRA SFT on multilingual refusals + self-distilled harmless answers (to avoid over-refusal).
  python 05_harden_lora.py                       # all languages, harmless:harmful = 1:1, 1 epoch
  python 05_harden_lora.py --safe_ratio 2        # more harmless data -> less over-refusal
  python 05_harden_lora.py --hold_out mr         # leave one language out -> tests cross-lingual transfer
  (bigger models on a T4: --bs 2 --ga 8)"""
import argparse, torch, random
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from common import *

REFUSALS = {  # have a native speaker sanity-check these!
 "en": "I'm sorry, but I can't help with that request because it could cause harm. If you'd like, I can help with a safer, related topic instead.",
 "hi": "माफ़ कीजिए, मैं इस अनुरोध में मदद नहीं कर सकता क्योंकि इससे नुकसान हो सकता है। अगर आप चाहें तो मैं किसी सुरक्षित, संबंधित विषय में मदद कर सकता हूँ।",
 "hinglish": "Maaf kijiye, main is request mein madad nahi kar sakta kyunki isse nuksaan ho sakta hai. Agar aap chahein, to main kisi surakshit, related topic mein madad kar sakta hoon.",
 "mr": "क्षमस्व, मी या विनंतीमध्ये मदत करू शकत नाही कारण यामुळे हानी होऊ शकते. तुम्हाला हवे असल्यास मी एखाद्या सुरक्षित, संबंधित विषयात मदत करू शकतो.",
 "ta": "மன்னிக்கவும், இந்தக் கோரிக்கையில் என்னால் உதவ முடியாது, ஏனெனில் இது தீங்கு விளைவிக்கக்கூடும். நீங்கள் விரும்பினால், பாதுகாப்பான தொடர்புடைய தலைப்பில் உதவ முடியும்.",
}
ap = argparse.ArgumentParser()
ap.add_argument("--hold_out", default=None); ap.add_argument("--out", default=None)
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--safe_ratio", type=float, default=1.0); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--ga", type=int, default=4)
a = ap.parse_args()
langs = [l for l in ALL_LANGS if l != a.hold_out]
out_dir = a.out or f"{OUT_DIR}/lora_safety" + (f"_holdout_{a.hold_out}" if a.hold_out else "")

tok, base = load_model()
random.seed(SEED)
pairs = []
harm_rows = load_rows("harmful", "train")
for r in harm_rows:
    for l in langs: pairs.append((r[l], REFUSALS[l]))
safe_rows = load_rows("harmless", "train")[:int(round(len(harm_rows) * a.safe_ratio))]
for l in langs:  # self-distillation: the base model's own answers keep helpfulness intact
    prompts = [r[l] for r in safe_rows]
    for p, o in zip(prompts, generate(base, tok, prompts, max_new=160)): pairs.append((p, o))
random.shuffle(pairs); print(f"train examples: {len(pairs)} (harmful {len(harm_rows) * len(langs)}, harmless {len(safe_rows) * len(langs)})")

def encode(p, r):
    prompt = chat(tok, p)
    full = tok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=False)
    if not full.startswith(prompt): full = prompt + r + (tok.eos_token or "")
    ids_p = tok(prompt, add_special_tokens=False).input_ids
    ids_f = tok(full, add_special_tokens=False).input_ids[:768]
    return ids_f, ([-100] * len(ids_p) + ids_f[len(ids_p):])[:len(ids_f)]

class DS(Dataset):
    def __init__(self, pairs): self.items = [encode(p, r) for p, r in pairs]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return {"input_ids": self.items[i][0], "labels": self.items[i][1]}

def collate(b):
    m = max(len(x["input_ids"]) for x in b)
    ids = torch.tensor([x["input_ids"] + [tok.pad_token_id] * (m - len(x["input_ids"])) for x in b])
    lab = torch.tensor([x["labels"] + [-100] * (m - len(x["labels"])) for x in b])
    return {"input_ids": ids, "labels": lab, "attention_mask": (torch.arange(m)[None] < torch.tensor([len(x["input_ids"]) for x in b])[:, None]).long()}

base.config.use_cache = False
model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.enable_input_require_grads(); model.print_trainable_parameters()
args = TrainingArguments(output_dir=out_dir + "_tmp", num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
                         gradient_accumulation_steps=a.ga, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=10,
                         bf16=(DTYPE == torch.bfloat16), fp16=(DTYPE == torch.float16), logging_steps=10, save_strategy="no",
                         report_to="none", gradient_checkpointing=True, remove_unused_columns=False, seed=SEED)
Trainer(model=model, args=args, train_dataset=DS(pairs), data_collator=collate).train()
model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
print("saved adapter to", out_dir)
