#!/usr/bin/env python3

import os
import logging
from typing import Dict

import pandas as pd
from tqdm import tqdm
import torch
from langdetect import detect_langs
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

# ------------- CONFIG -------------
HF_TOKEN = os.getenv("HF_TOKEN", None)    
INPUT_FILE = r"C:\\Users\\Elouan\\Documents\\Projet Biais LLM\\Construction du Dataset\\spotify_songs_with_regions_and_gender2.xlsx"
OUTPUT_FILE = r"C:\\Users\\Elouan\\Documents\\Projet Biais LLM\\Construction du Dataset\\spotify_songs_with_regions_and_gender_translated_simple.xlsx"
MODEL_NAME = "facebook/nllb-200-3.3B" #"facebook/nllb-200-distilled-600M" 
NEW_COL = "lyrics_english"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECT_THRESHOLD = 0.75
LOG_LEVEL = logging.INFO
MAX_GEN_LENGTH = 1024
# ----------------------------------

logging.basicConfig(level=LOG_LEVEL, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

LANG_MAP: Dict[str, str] = {
    "no": "nob_Latn","id": "ind_Latn","pl": "pol_Latn","ca": "cat_Latn","ru": "rus_Cyrl",
    "es": "spa_Latn","da": "dan_Latn","tr": "tur_Latn","hi": "hin_Deva","fr": "fra_Latn",
    "it": "ita_Latn","cy": "cym_Latn","ko": "kor_Hang","vi": "vie_Latn","et": "est_Latn",
    "sv": "swe_Latn","af": "afr_Latn","fi": "fin_Latn","ro": "ron_Latn","de": "deu_Latn",
    "sk": "slk_Latn","ja": "jpn_Jpan","hr": "hrv_Latn","ar": "arb_Arab","cs": "ces_Latn",
    "so": "som_Latn","pt": "por_Latn","en": "eng_Latn","sw": "swa_Latn","sq": "sqi_Latn",
    "nl": "nld_Latn","tl": "tgl_Latn",
}
TARGET_LANG = "eng_Latn"

def detect_overall_is_english(text: str, threshold: float = DETECT_THRESHOLD) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True
    try:
        langs = detect_langs(text)
        if not langs:
            return False
        top = langs[0]
        return str(top.lang) == "en" and float(top.prob) >= threshold
    except Exception:
        return False

def load_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_excel(path) if path.lower().endswith((".xls", ".xlsx")) else pd.read_csv(path)

def save_df(df: pd.DataFrame, path: str):
    df.to_excel(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)

def main():
    df = load_df(INPUT_FILE)
    if NEW_COL not in df.columns:
        df[NEW_COL] = ""

    tokenizer_kwargs = {}
    model_kwargs = {}
    if HF_TOKEN:
        tokenizer_kwargs["use_auth_token"] = HF_TOKEN
        model_kwargs["use_auth_token"] = HF_TOKEN

    logger.info("Loading tokenizer and model: %s on %s", MODEL_NAME, DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tokenizer_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, **model_kwargs).to(DEVICE)

    supports_src_lang = hasattr(tokenizer, "src_lang")

    n = len(df)
    logger.info("Translating %d rows (1 model call per row when needed)...", n)

    for idx in tqdm(range(n), desc="rows"):
        lyrics = df.at[idx, "lyrics"] if "lyrics" in df.columns else ""
        src_raw = str(df.at[idx, "language"]).strip().lower() if "language" in df.columns else ""

        if not isinstance(lyrics, str) or not lyrics.strip():
            df.at[idx, NEW_COL] = ""
            continue

        if src_raw == "en" and detect_overall_is_english(lyrics):
            df.at[idx, NEW_COL] = lyrics
            continue

        src_flores = LANG_MAP.get(src_raw, None)

        if supports_src_lang and src_flores:
            try:
                tokenizer.src_lang = src_flores
            except Exception:
                pass

        if src_flores:
            prefix = f"Translate from {src_flores} to {TARGET_LANG}:\n\n"
        else:
            prefix = f"Translate to English:\n\n"

        prompt = prefix + lyrics

        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(DEVICE)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=MAX_GEN_LENGTH)
            translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error("Translation failed for row %d: %s", idx, e)
            translated = ""

        df.at[idx, NEW_COL] = translated

    save_df(df, OUTPUT_FILE)
    logger.info("All done. Saved to %s", OUTPUT_FILE)

if __name__ == "__main__":
    main()