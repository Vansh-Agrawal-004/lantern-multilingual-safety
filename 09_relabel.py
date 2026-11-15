"""Re-score saved audit CSVs (no GPU) with the current detector in common.py and rewrite results/<model>/*.json.
Run it right before 07_report.py so every tag is scored the same way.
By default Hinglish metrics use only LLM-written prompts (the rule-based fallback prompts inflate over-refusal).
  python 09_relabel.py             # LLM-written Hinglish only
  python 09_relabel.py --all_hinglish"""
import argparse, glob, os, json, pandas as pd
from common import label_response, ALL_LANGS, RES_DIR, OUT_DIR, DATA_DIR
ap = argparse.ArgumentParser(); ap.add_argument("--all_hinglish", action="store_true"); a = ap.parse_args()
src = {}
for k in ["harmful", "harmless"]:
    for l in open(f"{DATA_DIR}/{k}_eval.jsonl", encoding="utf-8"):
        r = json.loads(l); src[r["id"]] = r.get("hinglish_src", "llm")
for f in sorted(glob.glob(f"{OUT_DIR}/audit_*.csv")):
    tag = os.path.basename(f)[6:-4]
    if tag == "smoke": continue
    df = pd.read_csv(f); df["label"] = df.resp.map(lambda t: label_response(str(t)))
    out = {}
    for lang in ALL_LANGS:
        out[lang] = {}
        for kind in ["harmful", "harmless"]:
            s = df[(df.lang == lang) & (df.kind == kind)]
            if lang == "hinglish" and not a.all_hinglish: s = s[s.id.map(src) == "llm"]
            for lab, key in [("refuse", "refusal"), ("comply", "comply"), ("incoherent", "incoherent")]:
                out[lang][f"{kind}_{key}"] = float((s.label == lab).mean())
            out[lang][f"n_{kind}"] = len(s)
    json.dump(out, open(f"{RES_DIR}/{tag}.json", "w"), indent=2)
    df.to_csv(f, index=False)
print("re-scored", len(glob.glob(f"{OUT_DIR}/audit_*.csv")) - 1, "audit files")
