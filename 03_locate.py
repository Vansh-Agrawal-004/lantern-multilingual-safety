"""WHERE does refusal live? Per-layer refusal directions (difference of means at the last prompt token) for each language;
cosine to the English direction; and held-out AUROC (harmful vs harmless eval prompts) of the English / pooled / own-language
direction, plus a prompt-length-only control (is the 'harmfulness' signal just dataset style?)."""
import json, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from common import *

tok, model = load_model()
dirs, ev = {}, {}
pooled_h, pooled_s = [], []
for lang in ALL_LANGS:
    h = last_token_acts(model, tok, [r[lang] for r in load_rows("harmful", "train")])
    s = last_token_acts(model, tok, [r[lang] for r in load_rows("harmless", "train")])
    d = h.mean(0) - s.mean(0)                                   # [L, d]
    dirs[lang] = d / d.norm(dim=-1, keepdim=True)
    pooled_h.append(h.mean(0)); pooled_s.append(s.mean(0))
    ev[lang] = (last_token_acts(model, tok, [r[lang] for r in load_rows("harmful", "eval")]),
                last_token_acts(model, tok, [r[lang] for r in load_rows("harmless", "eval")]))
    print("acts done:", lang, flush=True)
d = torch.stack(pooled_h).mean(0) - torch.stack(pooled_s).mean(0)
dirs["pooled"] = d / d.norm(dim=-1, keepdim=True)

L = dirs["en"].shape[0]
cos = {l: [float(dirs[l][i] @ dirs["en"][i]) for i in range(L)] for l in ALL_LANGS}
def auroc_curve(lang, key):
    h, s = ev[lang]; out = []
    for i in range(L):
        sc = torch.cat([h[:, i] @ dirs[key][i], s[:, i] @ dirs[key][i]]).numpy()
        out.append(float(roc_auc_score(np.r_[np.ones(len(h)), np.zeros(len(s))], sc)))
    return out
auc = {l: auroc_curve(l, "en") for l in ALL_LANGS}           # English direction applied to language l
auc_pooled = {l: auroc_curve(l, "pooled") for l in ALL_LANGS}
auc_own = {l: auroc_curve(l, l) for l in ALL_LANGS}
len_auc = {}
for l in ALL_LANGS:
    hl = [len(r[l]) for r in load_rows("harmful", "eval")]; sl = [len(r[l]) for r in load_rows("harmless", "eval")]
    len_auc[l] = float(roc_auc_score(np.r_[np.ones(len(hl)), np.zeros(len(sl))], np.r_[hl, sl]))

lo, hi = int(0.3 * L), int(0.75 * L)
mean_auc = np.mean([auc[l] for l in ALL_LANGS], axis=0)
best = lo + int(np.argmax(mean_auc[lo:hi]))
top3 = sorted(range(lo, hi), key=lambda i: -mean_auc[i])[:3]
print("best layer:", best, "(top3:", top3, ") mean AUROC(en-dir):", round(float(mean_auc[best]), 3), " n_layers:", L)
print(f"{'lang':9s} {'cos(en)':>8s} {'AUC en-dir':>11s} {'AUC pooled':>11s} {'AUC own':>8s} {'AUC length-only':>16s}")
for l in ALL_LANGS:
    print(f"{l:9s} {cos[l][best]:8.3f} {auc[l][best]:11.3f} {auc_pooled[l][best]:11.3f} {auc_own[l][best]:8.3f} {len_auc[l]:16.3f}")

torch.save({k: v for k, v in dirs.items()}, f"{OUT_DIR}/directions.pt")
json.dump({"best_layer": best, "top3_layers": top3, "n_layers": L, "cosine_to_en": cos, "auroc_en_dir": auc,
           "auroc_pooled_dir": auc_pooled, "auroc_own_dir": auc_own, "auroc_length_only": len_auc},
          open(f"{RES_DIR}/locate.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for l in ALL_LANGS:
    ax[0].plot(cos[l], label=l); ax[1].plot(auc[l], label=l); ax[2].plot(auc_pooled[l], label=l)
for a_, t in zip(ax, ["cosine(lang direction, English direction)", "AUROC: English direction", "AUROC: pooled direction"]):
    a_.axvline(best, ls="--", c="gray"); a_.set_xlabel("layer"); a_.set_title(t); a_.legend()
plt.tight_layout(); plt.savefig(f"{RES_DIR}/locate.png", dpi=150)
