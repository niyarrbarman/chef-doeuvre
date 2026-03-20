import torch
import torch.distributed as dist
import json
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

# ============================================================================
# ETHNICITY/REGION PREDICTION ONLY
# ============================================================================
# torchrun --nproc_per_node=<gpus> --nnodes=<nodes> \
#          --node_rank=0 --master_addr=<ip> --master_port=29500 ethnicity_predict.py
# ============================================================================

# ========================= CONFIG =========================
CSV_PATH = "/tmpdir/m25211shrm/musiclm/final-merged-ds.csv"
OUTPUT_PATH = "/tmpdir/m25211shrm/musiclm/ethnicity_predictions.csv"
MODEL_PATH = "/tmpdir/m25211shrm/musiclm/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 200
BATCH_SIZE = 32
# ========================================================

# ========================= DISTRIBUTED SETUP =========================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    else:
        return 0, 1, 0, False

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"Distributed inference initialized (world_size={world_size})")
    return rank, world_size, local_rank, True

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

RANK, WORLD_SIZE, LOCAL_RANK, IS_DISTRIBUTED = setup_distributed()
device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

# ========================= LOAD DATA =========================
df = pd.read_csv(CSV_PATH)
if RANK == 0:
    print(f"Loaded {len(df)} rows")
df = df[500:].reset_index(drop=True)
total_rows = len(df)

if IS_DISTRIBUTED:
    dist.barrier()

# ========================= LOAD MODEL =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)
model = model.to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

if IS_DISTRIBUTED:
    dist.barrier()

# ====================== ETHNICITY PROMPT ======================
ETHNICITY_SYSTEM = """You are analyzing song lyrics to predict the artist's region/ethnicity.

Based on the lyrics, predict:
- Artist Region: Must be EXACTLY one of [North America, Europe, Asia, South America, Africa, Oceania, Unknown]

Output ONLY valid JSON:
{"artist_region": "North America", "reasoning": "Brief explanation"}

NO extra text before or after JSON."""

def build_prompt(lyrics: str):
    return [
        {"role": "system", "content": ETHNICITY_SYSTEM},
        {"role": "user", "content": f"Predict artist region from these lyrics:\n\n{lyrics[:1500]}"}
    ]

# ====================== HELPER FUNCTIONS ======================
def extract_json(text: str) -> dict:
    text = re.sub(r'```json\s*|\s*```', '', text)
    matches = re.findall(r'\{[^{}]*\}', text)
    for match in matches:
        try:
            parsed = json.loads(match)
            if "artist_region" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    return None

def normalize_region(region_str: str) -> str:
    if not region_str or not isinstance(region_str, str):
        return "Unknown"
    r = region_str.lower().strip()
    if any(x in r for x in ["north america", "usa", "us", "america", "american", "canada", "mexico"]):
        return "North America"
    elif any(x in r for x in ["europe", "uk", "british", "french", "german", "italian", "spanish"]):
        return "Europe"
    elif any(x in r for x in ["asia", "japan", "korean", "chinese", "india"]):
        return "Asia"
    elif any(x in r for x in ["south america", "latin", "brazilian", "argentina"]):
        return "South America"
    elif any(x in r for x in ["africa", "nigerian", "south africa"]):
        return "Africa"
    elif any(x in r for x in ["oceania", "australia", "new zealand"]):
        return "Oceania"
    return "Unknown"

def parse_output(text: str) -> dict:
    parsed = extract_json(text)
    if parsed:
        return {"artist_region": normalize_region(parsed.get("artist_region", "Unknown"))}
    return {"artist_region": "Unknown"}

# ====================== BATCHED INFERENCE ======================
def infer_batch(lyrics_list: list) -> list:
    valid_indices, valid_lyrics = [], []
    for i, lyrics in enumerate(lyrics_list):
        if lyrics and len(str(lyrics).strip()) >= 10:
            valid_indices.append(i)
            valid_lyrics.append(str(lyrics).strip())

    if not valid_lyrics:
        return [{"artist_region": "Unknown"}] * len(lyrics_list)

    try:
        all_inputs = [tokenizer.apply_chat_template(build_prompt(l), tokenize=False,
                                                     add_generation_prompt=True) for l in valid_lyrics]
        inputs = tokenizer(all_inputs, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=True, temperature=0.3, top_p=0.9,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)

        batch_results = []
        for i, output in enumerate(outputs):
            input_len = inputs.input_ids[i].ne(tokenizer.pad_token_id).sum().item()
            gen_text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            batch_results.append(parse_output(gen_text))

        final = [None] * len(lyrics_list)
        for i in range(len(lyrics_list)):
            if i in valid_indices:
                final[i] = batch_results[valid_indices.index(i)]
            else:
                final[i] = {"artist_region": "Unknown"}
        return final

    except Exception as e:
        print(f"Rank {RANK} error: {e}")
        return [{"artist_region": "Unknown"}] * len(lyrics_list)

# ====================== MAIN LOOP ======================
def get_rank_indices(total, rank, world_size):
    per_rank = total // world_size
    remainder = total % world_size
    start = rank * per_rank + min(rank, remainder)
    end = start + per_rank + (1 if rank < remainder else 0)
    return start, end

start_idx, end_idx = get_rank_indices(total_rows, RANK, WORLD_SIZE)
my_indices = list(range(start_idx, end_idx))

if IS_DISTRIBUTED:
    dist.barrier()

my_results = []
num_batches = (len(my_indices) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in tqdm(range(num_batches), desc=f"Rank {RANK}", disable=(RANK != 0)):
    b_start = batch_idx * BATCH_SIZE
    b_end = min(b_start + BATCH_SIZE, len(my_indices))
    batch_indices = my_indices[b_start:b_end]
    batch_lyrics = [str(df.iloc[i]["lyrics"]) if pd.notna(df.iloc[i]["lyrics"]) else ""
                    for i in batch_indices]

    results_parsed = infer_batch(batch_lyrics)
    for i, idx in enumerate(batch_indices):
        my_results.append({"idx": idx, "predicted_region": results_parsed[i]["artist_region"]})

# ====================== GATHER & SAVE (FIXED) ======================
if IS_DISTRIBUTED:
    dist.barrier()
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, my_results)
    if RANK == 0:
        all_results = sorted([r for g in gathered for r in g], key=lambda x: x["idx"])
else:
    all_results = my_results

if RANK == 0:
    for result in all_results:
        idx = result["idx"]
        df.at[idx, "predicted_region"] = result["predicted_region"]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Calculate accuracy
    if 'continent' in df.columns: # <--- FIX APPLIED HERE: Checking for 'continent' column
        df['gt_region'] = df['continent'].apply(normalize_region) # <--- ACCESSING 'continent' column
        correct = (df['predicted_region'] == df['gt_region']).sum()
        total = len(df)
        accuracy = correct / total
        print(f"\n{'='*50}")
        print(f"ETHNICITY/REGION PREDICTION ACCURACY: {accuracy:.4f} ({correct}/{total})")
        print(f"{'='*50}")
        print(f"\nPredictions:\n{df['predicted_region'].value_counts()}")
        print(f"\nGround Truth:\n{df['gt_region'].value_counts()}")
    else:
        print("\nWARNING: Ground truth column 'continent' not found. Accuracy not calculated.")

if IS_DISTRIBUTED:
    dist.barrier()
    cleanup_distributed()
