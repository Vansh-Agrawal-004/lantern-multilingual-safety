import os
# Everything model-specific is keyed by MODEL_TAG, so you can run several models side by side.
MODEL = os.environ.get("LEXSI_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")   # model under test
MODEL_TAG = MODEL.split("/")[-1]
HINGLISH_GEN = os.environ.get("LEXSI_HINGLISH_GEN", "Qwen/Qwen2.5-3B-Instruct")  # rewrites English -> natural Hinglish
JUDGE = os.environ.get("LEXSI_JUDGE", "Qwen/Qwen2.5-3B-Instruct")                 # optional LLM judge (06_judge.py)
NLLB = "facebook/nllb-200-distilled-600M"  # NOTE: CC-BY-NC-4.0 (translation tool only)

# "hinglish" = natural romanised code-mixed Hindi written by an LLM (rule-based romanisation only as fallback)
ALL_LANGS = ["en", "hi", "hinglish", "mr", "ta"]
NLLB_CODES = {"hi": "hin_Deva", "mr": "mar_Deva", "ta": "tam_Taml"}

N_EVAL = 120          # eval prompts per kind (harmful / harmless) per language
N_TRAIN = 250         # harmful train prompts per language (also used for direction finding / threshold calibration)
N_TRAIN_SAFE = 500    # harmless train prompts available (LoRA uses safe_ratio x harmful count)
MAX_NEW = 64
SEED = 0
DATA_VERSION = "v2"   # bump to force a fresh dataset build
DATA_DIR = f"data_{DATA_VERSION}"
OUT_DIR = f"outputs/{MODEL_TAG}"
RES_DIR = f"results/{MODEL_TAG}"

HF_USER = os.environ.get("LEXSI_HF_USER", "aneeshadas02")
for _d in (DATA_DIR, OUT_DIR, RES_DIR): os.makedirs(_d, exist_ok=True)
