"""Close the gap WITHOUT retraining: conditional refusal steering.
Direction source: English-only vs pooled multilingual.  Gate: global threshold ('cond'), per-language threshold ('condlang',
needs language ID at inference), or none ('uncond').  Also a small layer sweep around the best layer."""
import json, argparse, torch
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--coefs", type=float, nargs="+", default=[2.0, 3.0, 4.5]); a = ap.parse_args()

info = json.load(open(f"{RES_DIR}/locate.json")); best = info["best_layer"]
dirs = torch.load(f"{OUT_DIR}/directions.pt")
tok, model = load_model()
NL = len(get_layers(model))
H = {l: last_token_acts(model, tok, [r[l] for r in load_rows("harmful", "train")]) for l in ALL_LANGS}   # [N, L, d]
S = {l: last_token_acts(model, tok, [r[l] for r in load_rows("harmless", "train")]) for l in ALL_LANGS}

def calib(src, layer):
    d = dirs[src][layer]
    ph = {l: H[l][:, layer] @ d for l in ALL_LANGS}; ps = {l: S[l][:, layer] @ d for l in ALL_LANGS}
    per = {l: dict(gap=float(ph[l].mean() - ps[l].mean()), thr=float((ph[l].mean() + ps[l].mean()) / 2)) for l in ALL_LANGS}
    use = ["en"] if src == "en" else ALL_LANGS          # global calibration: English-only for the English direction
    ch, cs = torch.cat([ph[l] for l in use]), torch.cat([ps[l] for l in use])
    glob = dict(gap=float(ch.mean() - cs.mean()), thr=float((ch.mean() + cs.mean()) / 2))
    return d, per, glob

meta = {"layer": best, "calibration": {}, "configs": {}}
def run(name, src, gate, coef, layer):
    d, per, glob = calib(src, layer)
    if layer == best: meta["calibration"][src] = {"per_lang": per, "global": glob}
    def mk(lang):
        if gate == "uncond": return ConditionalSteer(model, layer, d, -1e9, coef * glob["gap"])
        c = per[lang] if gate == "condlang" else glob
        return ConditionalSteer(model, layer, d, c["thr"], coef * max(c["gap"], 0.0))
    summary, records = evaluate(model, tok, ctx=mk)
    save_results(name, summary, records)
    meta["configs"][name] = {"src": src, "gate": gate, "coef": coef, "layer": layer}
    json.dump(meta, open(f"{OUT_DIR}/steer_config.json", "w"), indent=1)

for src in ["en", "pooled"]:
    for gate in ["cond", "condlang"]:
        for coef in ([3.0] if src == "en" else a.coefs):
            run(f"steer_{src}_{gate}_x{coef}", src, gate, coef, best)
run("steer_en_uncond_x1.5", "en", "uncond", 1.5, best)           # blunt baseline: steers every prompt
for layer in sorted({max(0, best - 3), min(NL - 1, best + 3)} - {best}):
    run(f"steer_pooled_condlang_x3.0_L{layer}", "pooled", "condlang", 3.0, layer)

torch.save({"en": dirs["en"][best], "pooled": dirs["pooled"][best]}, f"{OUT_DIR}/steering_vectors.pt")
print("saved steering vectors + calibration to", OUT_DIR)
