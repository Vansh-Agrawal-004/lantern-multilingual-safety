"""Evaluate refusal (harmful) and over-refusal (harmless) per language, with refuse/comply/incoherent labels.
  python 02_eval.py --tag baseline
  python 02_eval.py --tag hardened --adapter outputs/<model_tag>/lora_safety"""
import argparse
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--adapter", default=None); ap.add_argument("--n", type=int, default=None)
a = ap.parse_args()
tok, model = load_model(a.adapter)
summary, records = evaluate(model, tok, n=a.n)
save_results(a.tag, summary, records)
