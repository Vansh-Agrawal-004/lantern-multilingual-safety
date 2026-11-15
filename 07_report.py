"""Aggregate results/<model>/*.json into markdown tables + a figure with 95% Wilson intervals."""
import json, glob, os, math
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

def wilson(p, n, z=1.96):
    if not n: return 0.0, 0.0
    den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)

rows = []
for f in sorted(glob.glob(f"{RES_DIR}/*.json")):
    tag = os.path.basename(f)[:-5]
    if tag == "locate" or tag.startswith("judge_"): continue
    for lang, v in json.load(open(f)).items(): rows.append(dict(tag=tag, lang=lang, **v))
df = pd.DataFrame(rows)
def piv(col):
    t = df.pivot(index="tag", columns="lang", values=col)[ALL_LANGS].round(3); t["mean"] = t.mean(axis=1).round(3); return t
harm, safe = piv("harmful_refusal"), piv("harmless_refusal")
gap = (harm[ALL_LANGS] - safe[ALL_LANGS]).round(3); gap["mean"] = gap.mean(axis=1).round(3)
parts = [("Refusal rate on harmful prompts (higher = safer)", harm),
         ("Refusal rate on harmless prompts (lower = less over-refusal)", safe),
         ("Discrimination = harmful refusal - harmless refusal (higher = refuses the right things)", gap)]
if "harmful_incoherent" in df: parts += [("Incoherent / 'cannot understand' responses to harmful prompts", piv("harmful_incoherent")),
                                         ("Incoherent responses to harmless prompts (language-ability proxy)", piv("harmless_incoherent"))]
jt = {}
for f in sorted(glob.glob(f"{RES_DIR}/judge_*.json")):
    jt[os.path.basename(f)[6:-5]] = {l: v["HARMFUL"] for l, v in json.load(open(f)).items()}
if jt:
    j = pd.DataFrame(jt).T.reindex(columns=ALL_LANGS).round(3); j["mean"] = j.mean(axis=1).round(3)
    parts.append(("LLM-judge: share of harmful prompts that got REAL harmful help (lower = safer)", j))
md = "".join(f"## {t}\n\n{tb.to_markdown()}\n\n" for t, tb in parts)
open(f"{RES_DIR}/report.md", "w").write(md); print(md)

steer = [t for t in gap.index if t.startswith("steer_") and "_cond" in t and "_uncond" not in t]
top_steer = list(gap.loc[steer, "mean"].sort_values(ascending=False).index[:2])
sel = [t for t in ["baseline"] if t in gap.index] + [t for t in gap.index if t.startswith("hardened") and "holdout" not in t] + top_steer
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
w = 0.8 / max(1, len(sel))
for k, (a_, col, title) in enumerate(zip(ax, ["harmful_refusal", "harmless_refusal"], ["Harmful refusal (higher = safer)", "Over-refusal (lower = better)"])):
    for i, t in enumerate(sel):
        ps, errs = [], [[], []]
        for l in ALL_LANGS:
            r = df[(df.tag == t) & (df.lang == l)].iloc[0]; p = r[col]; n = r.get("n_" + col.split("_")[0], 100)
            lo, hi = wilson(p, n); ps.append(p); errs[0].append(p - lo); errs[1].append(hi - p)
        a_.bar([x + i * w for x in range(len(ALL_LANGS))], ps, w, yerr=errs, capsize=2, label=t)
    a_.set_xticks([x + 0.4 - w / 2 for x in range(len(ALL_LANGS))]); a_.set_xticklabels(ALL_LANGS); a_.set_title(title)
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{RES_DIR}/summary.png", dpi=150)
