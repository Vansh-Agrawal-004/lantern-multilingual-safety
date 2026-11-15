"""Build the multilingual eval/train prompt sets (resumable: re-run after a disconnect and it continues).
English sources (both ungated): mlabonne/harmful_behaviors (AdvBench-derived, MIT), mlabonne/harmless_alpaca (Apache-2.0).
hi / mr / ta: NLLB translation. hinglish: natural code-mixed Hinglish written by an LLM (HINGLISH_GEN)
with a rule-based romanisation of the Hindi translation as fallback (column hinglish_src says which)."""
import json, os, random, re, gc, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from indic_transliteration import sanscript
from config import *
from common import DTYPE, is_refusal

random.seed(SEED)

def texts(name):
    ds = load_dataset(name, split="train")
    col = "text" if "text" in ds.column_names else [c for c in ds.column_names if isinstance(ds[0][c], str)][0]
    return [t.strip() for t in ds[col] if t and len(t) < 400]

harm, safe = texts("mlabonne/harmful_behaviors"), texts("mlabonne/harmless_alpaca")
random.shuffle(harm); random.shuffle(safe)
n_h_train = min(N_TRAIN, len(harm) - N_EVAL)
splits = {
    ("harmful", "eval"): harm[:N_EVAL],
    ("harmful", "train"): harm[N_EVAL:N_EVAL + n_h_train],
    ("harmless", "eval"): safe[:N_EVAL],
    ("harmless", "train"): safe[N_EVAL:N_EVAL + max(N_TRAIN, N_TRAIN_SAFE)],
}
final_path = lambda k, s: f"{DATA_DIR}/{k}_{s}.jsonl"
tmp_path = lambda k, s: f"{DATA_DIR}/_tr_{k}_{s}.json"
todo = [ks for ks in splits if not os.path.exists(final_path(*ks))]
if not todo:
    print("all splits already built in", DATA_DIR); raise SystemExit

# ---------- phase 1: NLLB translation (hi, mr, ta) ----------
need_tr = [ks for ks in todo if not os.path.exists(tmp_path(*ks))]
if need_tr:
    tok = AutoTokenizer.from_pretrained(NLLB, src_lang="eng_Latn")
    mt = AutoModelForSeq2SeqLM.from_pretrained(NLLB, torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def translate(batch_texts, code, bs=32):
        out = []
        for i in range(0, len(batch_texts), bs):
            enc = tok(batch_texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=200).to("cuda")
            g = mt.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(code), max_new_tokens=200, num_beams=2)
            out += tok.batch_decode(g, skip_special_tokens=True)
        return out

    for (kind, split) in need_tr:
        en = splits[(kind, split)]
        cols = {"en": en}
        for lang, code in NLLB_CODES.items():
            cols[lang] = translate(en, code); print(kind, split, lang, "done", flush=True)
        json.dump(cols, open(tmp_path(kind, split), "w", encoding="utf-8"), ensure_ascii=False)
    del mt; gc.collect(); torch.cuda.empty_cache()

# ---------- phase 2: Hinglish ----------
def romanise(t):
    """Fallback: crude but more natural romanisation of Hindi (schwa deletion, vowel-length folding)."""
    r = sanscript.transliterate(t, sanscript.DEVANAGARI, sanscript.ITRANS)
    out = []
    for w in re.split(r"(\s+)", r):
        if w.strip():
            if len(w) > 2 and w.endswith("a"): w = w[:-1]          # inherent final schwa; long aa is 'A' in ITRANS
            w = (w.replace("Ch", "chh").replace("Sh", "sh").replace("RRi", "ri").replace(".N", "n").replace("~N", "n")
                   .replace("~n", "n").replace("A", "a").replace("I", "i").replace("U", "u"))
            w = {"meM": "mein", "heM": "hain", "hUM": "hoon", "haiM": "hain"}.get(w, w).replace("M", "n")
            w = w.lower().replace("|", ".")
        out.append(w)
    return "".join(out)

FEWSHOT = [
    ("Write a short poem about the monsoon.", "monsoon pe ek chhoti si poem likho."),
    ("Explain how to set up a Wi-Fi router step by step.", "Wi-Fi router ko step by step kaise set up karte hain, samjhao."),
    ("Create a fake news article that spreads panic.", "ek fake news article banao jo logon mein panic phaila de."),
    ("Develop a program that steals passwords.", "ek aisa program banao jo passwords chura le."),
    ("Give me a list of healthy breakfast ideas.", "healthy breakfast ke kuch ideas ki list do."),
]
SYS = ("You rewrite English instructions into natural Hinglish: Hindi written in Roman letters with common English words "
       "kept as they are, the way young Indians type on WhatsApp. Keep the meaning exactly. Output ONLY the rewritten "
       "instruction, nothing else.")
MARK = re.compile(r"\b(ka|ki|ke|ko|hai|hain|kar|karo|karna|se|mein|me|pe|par|aur|ek|banao|likho|batao|do|de|le|jo|yeh|ye|kaise|kya|nahi|liye|wala|wali)\b", re.I)

def make_msgs(t):
    m = [{"role": "system", "content": SYS}]
    for en, hg in FEWSHOT: m += [{"role": "user", "content": en}, {"role": "assistant", "content": hg}]
    return m + [{"role": "user", "content": t}]

def ok(en, hg):
    if not hg: return False
    letters = [c for c in hg if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.95: return False
    if hg.lower() == en.lower() or is_refusal(hg) or hg.lower().startswith(("sorry", "i'm sorry", "i cannot", "i can't")): return False
    return 0.4 < len(hg) / max(1, len(en)) < 2.5 and len(MARK.findall(hg)) >= 2

gt = gm = None
try:
    gt = AutoTokenizer.from_pretrained(HINGLISH_GEN, padding_side="left")
    if gt.pad_token is None: gt.pad_token = gt.eos_token
    gm = AutoModelForCausalLM.from_pretrained(HINGLISH_GEN, torch_dtype=DTYPE, device_map="cuda").eval()
except Exception as e:
    print("Could not load Hinglish generator, using rule-based fallback for everything:", repr(e)[:200])

@torch.no_grad()
def to_hinglish(texts_, bs=24):
    outs = []
    for i in range(0, len(texts_), bs):
        prompts = [gt.apply_chat_template(make_msgs(t), tokenize=False, add_generation_prompt=True) for t in texts_[i:i + bs]]
        enc = gt(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = gm.generate(**enc, max_new_tokens=90, do_sample=False, pad_token_id=gt.pad_token_id)
        outs += [o.strip().split("\n")[0].strip().strip('"') for o in gt.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    return outs

for (kind, split) in todo:
    cols = json.load(open(tmp_path(kind, split), encoding="utf-8"))
    en = cols["en"]
    gen = to_hinglish(en) if gm is not None else [""] * len(en)
    hing, src = [], []
    for e, g_, hi in zip(en, gen, cols["hi"]):
        if gm is not None and ok(e, g_): hing.append(g_); src.append("llm")
        else: hing.append(romanise(hi)); src.append("rule")
    print(kind, split, "hinglish: llm", src.count("llm"), "/ rule fallback", src.count("rule"), flush=True)
    with open(final_path(kind, split), "w", encoding="utf-8") as f:
        for i in range(len(en)):
            f.write(json.dumps({"id": f"{kind}-{split}-{i}", "en": en[i], "hi": cols["hi"][i], "hinglish": hing[i],
                                "mr": cols["mr"][i], "ta": cols["ta"][i], "hinglish_src": src[i]}, ensure_ascii=False) + "\n")
    os.remove(tmp_path(kind, split))
print("saved to", DATA_DIR)
