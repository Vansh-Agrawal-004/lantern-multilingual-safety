"""Layer sweep for the language-agnostic pooled direction + global gate (no language ID needed).
  python 04c_layer_sweep.py --layers 12 13 14 15 16 18"""
import json, argparse, torch
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--coefs", type=float, nargs="+", default=[4.5]); ap.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16, 18]); a = ap.parse_args()

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
    json.dump(meta, open(f"{OUT_DIR}/steer_config_sweep.json", "w"), indent=1)

for layer in a.layers:
    if layer < NL:
        for coef in a.coefs:
            run(f"steer_pooled_cond_x{coef}_L{layer}", "pooled", "cond", coef, layer)
