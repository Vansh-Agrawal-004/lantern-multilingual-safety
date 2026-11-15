"""OPTIONAL: same experiment with SafeTune's own RefusalDirectionTrainer (needs Python >= 3.12).
alpha is model-dependent (README: 0.3 on a 0.5B model, default 20 is far too strong) -> we sweep."""
from common import *
from safetune.runner import steer

tok, model = load_model()
hr, sr = load_rows("harmful", "train"), load_rows("harmless", "train")
cal = {
    "en": ([r["en"] for r in hr[:64]], [r["en"] for r in sr[:64]]),
    "multi": ([r[l] for r in hr[:13] for l in ALL_LANGS], [r[l] for r in sr[:13] for l in ALL_LANGS]),
}
for src, (harmful, harmless) in cal.items():
    for alpha in [0.1, 0.3]:
        trainer = steer.RefusalDirectionTrainer(model, tok, alpha=alpha)
        wrapped, _ = trainer.calibrate(harmful=harmful, harmless=harmless)
        summary, records = evaluate(model, tok, ctx=lambda lang=None, w=wrapped: w)   # `with wrapped:` activates hooks
        save_results(f"safetune_steer_{src}_a{alpha}", summary, records)
