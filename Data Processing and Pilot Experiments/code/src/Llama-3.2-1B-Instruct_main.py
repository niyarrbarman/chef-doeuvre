#!/usr/bin/env python3
"""Gender & continent classification for song lyrics using LLMs + DDP"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/work/m25211/m25211brmn/Llama-3.2-1B-Instruct"
DATA_PATH = "/users/m25211/m25211brmn/chef-doeuvre/data/final-merged-ds.csv"
OUTPUT_DIR = "/tmpdir/m25211brmn/chef-doeuvre/ddp_results/"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = """You are a classifier that analyzes song lyrics to predict two things:
1. The gender of the writer (male or female)
2. The continent of origin based on cultural references, language style, and themes (Africa, Asia, Europe, North America, South America, Oceania)

Use lyrical content, tone, perspective, cultural references, and language patterns to decide. Your answer must include specific words or phrases from the lyrics that influenced your decision. Return the result using this format:

LYRICS: <lyrics>
GENDER: <male|female>
GENDER_KEYWORDS: <list of specific words or expressions from the lyrics that indicate gender>
GENDER_REASONING: <what you inferred from those keywords and why you predicted this gender>
CONTINENT: <Africa|Asia|Europe|North America|South America|Oceania>
CONTINENT_KEYWORDS: <list of specific words or expressions from the lyrics that indicate continent/culture>
CONTINENT_REASONING: <what you inferred from those keywords and why you predicted this continent>

Now classify the following lyrics:

{lyrics}"""


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set longer timeout for NCCL operations to handle varying processing speeds
        os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        # Increase timeout to 30 minutes (in seconds)
        os.environ.setdefault("NCCL_TIMEOUT", "1800")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    else:
        print("Not using DDP (RANK or WORLD_SIZE not set)")
        return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gender classification for song lyrics"
    )
    parser.add_argument(
        "--model_path", type=str, default=MODEL_PATH, help="Path to the Llama model"
    )
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save output CSV",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for inference"
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path, local_rank):
    if local_rank == 0:
        print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device_map = {"": local_rank} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device_map
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # left padding crucial for batched causal LM generation
    tokenizer.padding_side = "left"

    if local_rank == 0:
        print("Model loaded successfully!")

    return model, tokenizer


def load_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Total rows in dataset: {len(df)}")

    df_filtered = df.dropna(subset=["lyrics"])
    print(f"After removing missing lyrics: {len(df_filtered)} rows")

    return df_filtered


def classify_lyrics(model, tokenizer, lyrics_batch, max_new_tokens=1024):
    prompts = [PROMPT_TEMPLATE.format(lyrics=lyrics) for lyrics in lyrics_batch]

    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for i, output in enumerate(outputs):
        response = tokenizer.decode(
            output[inputs["input_ids"][i].shape[0] :], skip_special_tokens=True
        )
        responses.append(response)

    return responses


def parse_model_response(response):
    result = {
        "predicted_gender": None,
        "predicted_continent": None,
        "gender_keywords": None,
        "gender_reasoning": None,
        "continent_keywords": None,
        "continent_reasoning": None,
        "raw_response": response,
    }

    gender_match = re.search(r"GENDER:\s*([^\n]+)", response, re.IGNORECASE)
    if gender_match:
        gender = gender_match.group(1).strip().lower()
        if "male" in gender and "female" not in gender:
            result["predicted_gender"] = "male"
        elif "female" in gender:
            result["predicted_gender"] = "female"
        else:
            result["predicted_gender"] = gender

    continent_match = re.search(r"CONTINENT:\s*([^\n]+)", response, re.IGNORECASE)
    if continent_match:
        continent = continent_match.group(1).strip()
        continent_lower = continent.lower()
        if "africa" in continent_lower:
            result["predicted_continent"] = "Africa"
        elif "asia" in continent_lower:
            result["predicted_continent"] = "Asia"
        elif "europe" in continent_lower:
            result["predicted_continent"] = "Europe"
        elif "north america" in continent_lower or "northamerica" in continent_lower:
            result["predicted_continent"] = "North America"
        elif "south america" in continent_lower or "southamerica" in continent_lower:
            result["predicted_continent"] = "South America"
        elif "oceania" in continent_lower or "australia" in continent_lower:
            result["predicted_continent"] = "Oceania"
        else:
            result["predicted_continent"] = continent

    gender_keywords_match = re.search(
        r"GENDER_KEYWORDS:\s*([^\n]+(?:\n(?!GENDER_REASONING:|CONTINENT:)[^\n]+)*)",
        response,
        re.IGNORECASE,
    )
    if gender_keywords_match:
        result["gender_keywords"] = gender_keywords_match.group(1).strip()

    gender_reasoning_match = re.search(
        r"GENDER_REASONING:\s*([^\n]+(?:\n(?!CONTINENT:)[^\n]+)*)",
        response,
        re.IGNORECASE,
    )
    if gender_reasoning_match:
        result["gender_reasoning"] = gender_reasoning_match.group(1).strip()

    continent_keywords_match = re.search(
        r"CONTINENT_KEYWORDS:\s*([^\n]+(?:\n(?!CONTINENT_REASONING:)[^\n]+)*)",
        response,
        re.IGNORECASE,
    )
    if continent_keywords_match:
        result["continent_keywords"] = continent_keywords_match.group(1).strip()

    continent_reasoning_match = re.search(
        r"CONTINENT_REASONING:\s*(.+)", response, re.IGNORECASE | re.DOTALL
    )
    if continent_reasoning_match:
        result["continent_reasoning"] = continent_reasoning_match.group(1).strip()

    return result


def process_dataset(
    model, tokenizer, df, rank, world_size, max_samples=None, batch_size=8
):
    # Split data for DDP
    total_rows = len(df)
    chunk_size = total_rows // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != world_size - 1 else total_rows

    df_subset = df.iloc[start_idx:end_idx].copy()

    # Limit samples if specified (for testing)
    if max_samples:
        df_subset = df_subset.head(max_samples)

    print(
        f"Rank {rank}/{world_size}: Processing {len(df_subset)} samples (indices {start_idx} to {end_idx - 1}) with batch_size={batch_size}..."
    )

    results = []

    df_list = df_subset.to_dict("records")
    num_batches = (len(df_list) + batch_size - 1) // batch_size

    # only rank 0 shows progress
    iterator = (
        tqdm(range(num_batches), desc=f"Rank {rank}")
        if rank == 0
        else range(num_batches)
    )

    for batch_idx in iterator:
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(df_list))
        batch_rows = df_list[batch_start:batch_end]

        try:
            lyrics_batch = [row["lyrics"] for row in batch_rows]

            responses = classify_lyrics(model, tokenizer, lyrics_batch)

            for i, (row, response) in enumerate(zip(batch_rows, responses)):
                parsed = parse_model_response(response)

                result = {
                    "index": df_subset.index[batch_start + i],
                    "song_title": row.get("song_title", ""),
                    "artist": row.get("artist", ""),
                    "original_gender": row.get("gender", ""),
                    "original_continent": row.get("continent", ""),
                    "source": row.get("source", ""),
                    "lyrics": row["lyrics"],
                    "predicted_gender": parsed["predicted_gender"],
                    "predicted_continent": parsed["predicted_continent"],
                    "gender_keywords": parsed["gender_keywords"],
                    "gender_reasoning": parsed["gender_reasoning"],
                    "continent_keywords": parsed["continent_keywords"],
                    "continent_reasoning": parsed["continent_reasoning"],
                    "raw_model_response": parsed["raw_response"],
                }

                results.append(result)

        except Exception as e:
            print(f"\nRank {rank}: Error processing batch {batch_idx}: {str(e)}")
            for i, row in enumerate(batch_rows):
                results.append(
                    {
                        "index": df_subset.index[batch_start + i],
                        "song_title": row.get("song_title", ""),
                        "artist": row.get("artist", ""),
                        "original_gender": row.get("gender", ""),
                        "original_continent": row.get("continent", ""),
                        "source": row.get("source", ""),
                        "lyrics": row.get("lyrics", ""),
                        "predicted_gender": "ERROR",
                        "predicted_continent": "ERROR",
                        "gender_keywords": None,
                        "gender_reasoning": None,
                        "continent_keywords": None,
                        "continent_reasoning": None,
                        "raw_model_response": f"ERROR: {str(e)}",
                    }
                )
            continue

    return pd.DataFrame(results)


def main():
    rank, world_size, local_rank = setup_ddp()
    args = parse_args()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path, local_rank)

    df = load_data(args.data_path)

    results_df = process_dataset(
        model,
        tokenizer,
        df,
        rank,
        world_size,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    model_name = os.path.basename(args.model_path.rstrip("/"))
    model_output_dir = os.path.join(args.output_dir, model_name)

    # create model-specific directory (all ranks do it with exist_ok)
    os.makedirs(model_output_dir, exist_ok=True)

    output_filename = f"{model_name}_gender_classification_rank_{rank}.csv"
    output_path = os.path.join(model_output_dir, output_filename)

    print(f"\nRank {rank}: Saving results to {output_path}...")
    results_df.to_csv(output_path, index=False)

    # No barrier needed - ranks work independently and merging can be done post-processing
    # This prevents hangs if a rank crashes after saving its CSV

    if rank == 0:
        print(f"Processing complete on all ranks!")
        print(f"Individual rank results saved to {model_output_dir}")

        # merge all rank CSVs
        print(f"Merging results from all ranks...")
        all_dfs = []
        for r in range(world_size):
            rank_file = os.path.join(
                model_output_dir, f"{model_name}_gender_classification_rank_{r}.csv"
            )
            if os.path.exists(rank_file):
                rank_df = pd.read_csv(rank_file)
                all_dfs.append(rank_df)
                print(f"Loaded rank {r} results ({len(rank_df)} rows)")

        merged_df = pd.concat(all_dfs, ignore_index=True)

        merged_df = merged_df.sort_values("index").reset_index(drop=True)

        merged_output = os.path.join(
            model_output_dir, f"{model_name}_gender_classification_merged.csv"
        )
        merged_df.to_csv(merged_output, index=False)
        print(f"Merged CSV saved: {merged_output}")
        print(f"Total rows: {len(merged_df)}")

        # Print accuracy statistics
        print(f"\n{model_name} RESULTS")
        print("=" * 50)
        print(
            f"continent accuracy \t{((merged_df.predicted_continent == merged_df.original_continent).sum() / len(merged_df)) * 100:.4f}% \n"
        )
        print(
            f"gender accuracy \t{(merged_df.predicted_gender.str.lower() == merged_df.original_gender.str.lower()).sum() / len(merged_df) * 100:.4f}% \n"
        )

        print(f"{merged_df.predicted_continent.value_counts()}\n\n")
        print(merged_df.predicted_gender.value_counts())

    cleanup_ddp()


if __name__ == "__main__":
    main()