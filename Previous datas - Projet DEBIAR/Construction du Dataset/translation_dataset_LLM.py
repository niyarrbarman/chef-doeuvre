#!/usr/bin/env python3
"""
NLLB-3B translation pipeline for spotify_songs_with_regions_and_gender.xlsx

- Uses column 'language' (2-letter ISO codes e.g. 'en','es','tl'...)
- If language != 'en' -> translate whole lyrics from that declared source
- If language == 'en' -> detect per-sentence non-english parts and translate only those
- Uses facebook/nllb-200-7b (FLORES codes)
- Batch GPU inference, adds column 'lyrics_english', saves to .xlsx
"""

import os
import re
import sys
import time
import logging
from typing import List, Tuple, Dict

import pandas as pd
from tqdm import tqdm

import torch
import huggingface_hub
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from langdetect import detect_langs

# --------------- CONFIG ----------------
INPUT_FILE = "C:\\Users\\Elouan\\Documents\\Projet Biais LLM\\Construction du Dataset\\spotify_songs_with_regions_and_gender2.xlsx"
OUTPUT_FILE = "C:\\Users\\Elouan\\Documents\\Projet Biais LLM\\Construction du Dataset\\spotify_songs_with_regions_and_gender_translated.xlsx"
NEW_COL = "lyrics_english"
MODEL_NAME = "facebook/nllb-200-distilled-600M"  #"facebook/nllb-200-3.3B"
MAX_BATCH_TOKENS = 3000  # dynamic batching threshold
#BATCH_SIZE = 4
MAX_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SENT_DETECT_PROB_THRESH = 0.60   # threshold to consider a sentence non-English
SENTENCE_SPLIT_RE = r'(?<=[\.\!\?\n])\s+'
LOG_LEVEL = logging.INFO


load_dotenv()
huggingface_hub.login(token=os.getenv("HF_TOKEN"))
# ---------------------------------------

logging.basicConfig(level=LOG_LEVEL, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Mapping 2-letter codes (your 32 langs) -> FLORES codes used by NLLB
LANG_MAP: Dict[str, str] = {
    "no": "nob_Latn",   # Norwegian Bokmål
    "id": "ind_Latn",
    "pl": "pol_Latn",
    "ca": "cat_Latn",
    "ru": "rus_Cyrl",
    "es": "spa_Latn",
    "da": "dan_Latn",
    "tr": "tur_Latn",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "cy": "cym_Latn",
    "ko": "kor_Hang",
    "vi": "vie_Latn",
    "et": "est_Latn",
    "sv": "swe_Latn",
    "af": "afr_Latn",
    "fi": "fin_Latn",
    "ro": "ron_Latn",
    "de": "deu_Latn",
    "sk": "slk_Latn",
    "ja": "jpn_Jpan",
    "hr": "hrv_Latn",
    "ar": "arb_Arab",
    "cs": "ces_Latn",
    "so": "som_Latn",
    "pt": "por_Latn",
    "en": "eng_Latn",
    "sw": "swa_Latn",   # swahili
    "sq": "sqi_Latn",
    "nl": "nld_Latn",
    "tl": "tgl_Latn",   # tagalog
}

TARGET_LANG = "eng_Latn"  # FLORES target for English

# --------------- helpers ----------------
def load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def save_dataframe(df: pd.DataFrame, path: str):
    df.to_excel(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")


def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(SENTENCE_SPLIT_RE, text)
    return [p.strip() for p in parts if p.strip()]


def detect_sentence_language_prob(sentence: str) -> Tuple[str, float]:
    try:
        langs = detect_langs(sentence)
        if not langs:
            return "unknown", 0.0
        top = langs[0]
        return str(top.lang), float(top.prob)
    except Exception:
        return "unknown", 0.0


# --------------- Translator ---------------
class NLLBTranslator:
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        logger.info(f"Loading model {model_name} in 8-bit on {device} ...")
        self.device = torch.device(device)

        # require bitsandbytes installed
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=True)
        # load in 8bit; device_map auto places weights on GPU/CPU
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map={"": "cpu"},   # force tout sur CPU
            low_cpu_mem_usage=True,
            use_auth_token=True
        )
            
        """model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_auth_token=True
        )"""

        self.forced_bos = None
        if hasattr(self.tokenizer, "lang_code_to_id") and TARGET_LANG in self.tokenizer.lang_code_to_id:
            self.forced_bos = self.tokenizer.lang_code_to_id[TARGET_LANG]

    def translate_batch(self, texts: List[str], max_length: int = MAX_LENGTH) -> List[str]:
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)

        gen_kwargs = {"max_length": max_length}
        if self.forced_bos is not None:
            gen_kwargs["forced_bos_token_id"] = self.forced_bos

        with torch.inference_mode():
            outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


# --------------- main pipeline ---------------
def process_file(input_path: str, output_path: str):
    df = load_dataframe(input_path)
    n = len(df)
    logger.info(f"Loaded {n} rows from {input_path}")

    if NEW_COL not in df.columns:
        df[NEW_COL] = ""

    translator = NLLBTranslator(MODEL_NAME, DEVICE)

    pieces_to_translate = []
    row_pieces_map = {}

    logger.info("Preparing segments...")
    for idx in range(n):
        lyrics = df.at[idx, "lyrics"] if "lyrics" in df.columns else ""
        src_raw = str(df.at[idx, "language"]).strip().lower() if "language" in df.columns else ""

        if not isinstance(lyrics, str) or not lyrics.strip():
            row_pieces_map[idx] = [{"text": "", "translate": False}]
            continue

        src_code = LANG_MAP.get(src_raw, None)

        if src_raw != "en" and src_code is not None:
            row_pieces_map[idx] = [{"text": lyrics, "translate": True}]
            pieces_to_translate.append((idx, 0, lyrics))
        else:
            sents = split_sentences(lyrics)
            pieces = []
            for si, s in enumerate(sents):
                code, prob = detect_sentence_language_prob(s)
                need_trans = (code != "en" and prob >= SENT_DETECT_PROB_THRESH) or (code == "unknown" and len(s) > 200)
                pieces.append({"text": s, "translate": need_trans})
                if need_trans:
                    pieces_to_translate.append((idx, si, s))
            row_pieces_map[idx] = pieces

    logger.info(f"Segments: {len(pieces_to_translate)}")

    # --------- Dynamic batching based on token count ---------
    i = 0
    pbar = tqdm(total=len(pieces_to_translate), desc="Translating")
    while i < len(pieces_to_translate):
        batch = []
        tokens_in_batch = 0

        while i < len(pieces_to_translate):
            row_idx, piece_idx, txt = pieces_to_translate[i]
            token_len = len(translator.tokenizer.encode(txt))

            if tokens_in_batch + token_len > MAX_BATCH_TOKENS and len(batch) > 0:
                break

            batch.append((row_idx, piece_idx, txt))
            tokens_in_batch += token_len
            i += 1

        texts = [b[2] for b in batch]

        try:
            translations = translator.translate_batch(texts)
        except Exception as e:
            logger.exception(f"Batch error: {e}")
            translations = [""] * len(texts)

        for (row_idx, piece_idx, _), trans in zip(batch, translations):
            row_pieces_map[row_idx][piece_idx]["translated"] = trans

        pbar.update(len(batch))
    pbar.close()

    # ------- Rebuild output -------
    for idx in range(n):
        out = []
        for p in row_pieces_map.get(idx, []):
            out.append(p.get("translated", p.get("text", "")))
        df.at[idx, NEW_COL] = " ".join(out)

    save_dataframe(df, output_path)
    logger.info("All done.")


if __name__ == "__main__":
    start_time = time.time()
    try:
        process_file(INPUT_FILE, OUTPUT_FILE)
    except Exception as exc:
        logger.exception(f"Fatal error: {exc}")
        sys.exit(1)
    elapsed = time.time() - start_time
    logger.info(f"Total elapsed: {elapsed:.1f}s")
