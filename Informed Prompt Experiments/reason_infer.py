import torch
import torch.distributed as dist
import json
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import socket

# ============================================================================
# REASONING-FIRST INFERENCE
# ============================================================================
# torchrun --nproc_per_node=<gpus> --nnodes=<nodes> \
#          --node_rank=0 --master_addr=<ip> --master_port=29500 reason_first.py
# ============================================================================

# ========================= CONFIG =========================
CSV_PATH = "/tmpdir/m25211shrm/musiclm/final-merged-ds.csv"
OUTPUT_PATH = "/tmpdir/m25211shrm/musiclm/deepseek_7b_inference_results_reason_first.csv"
MODEL_PATH = "/tmpdir/m25211shrm/musiclm/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 600
DEBUG_FIRST_N = 3
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
        print("Not using distributed mode - running on single GPU")
        return 0, 1, 0, False

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"Distributed inference initialized (world_size={world_size}, batch={BATCH_SIZE})")
    return rank, world_size, local_rank, True

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def print_rank0(*args, **kwargs):
    if RANK == 0:
        print(*args, **kwargs)

RANK, WORLD_SIZE, LOCAL_RANK, IS_DISTRIBUTED = setup_distributed()
device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
print(f"Rank {RANK}: Using device {device}")

# ========================= LOAD DATA =========================
df = pd.read_csv(CSV_PATH)
print_rank0(f"Loaded {len(df)} rows from {CSV_PATH}")
df = df[500:].reset_index(drop=True)
total_rows = len(df)
print_rank0(f"Processing {total_rows} rows")

if IS_DISTRIBUTED:
    dist.barrier()

# ========================= LOAD MODEL =========================
print(f"Rank {RANK}: Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)
model = model.to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"Rank {RANK}: Model loaded")
if IS_DISTRIBUTED:
    dist.barrier()

# ====================== REASONING-FIRST PROMPT ======================
REASON_FIRST_SYSTEM = """You are a forensic linguist. First, analyze the lyrics and make predictions about artist gender and region. Then rate which linguistic attributes you observed.

**Step 1: Make predictions**
- Artist Gender: Must be EXACTLY either "Male" or "Female" (no other options allowed)
- Artist Region: Must be EXACTLY one of [North America, Europe, Asia, South America, Africa, Oceania, Unknown]

**Step 2: Rate each attribute from 1 (not used/not present) to 10 (heavily used/very prominent):**
- Emotions (love, anger, sadness, joy, fear)
- Romance topics (relationships, heartbreak)
- Party/club themes (nightlife, dancing)
- Violence (aggression, conflict)
- Politics/religion themes
- Success/money themes
- Family themes
- Slang usage
- Formal language
- Profanity
- Intensifiers (very, really, so)
- Hedges (maybe, perhaps, kind of)
- First-person pronouns (I, me, my)
- Second-person pronouns (you, your)
- Third-person pronouns (he, she, they)
- Confidence markers
- Doubt/uncertainty markers
- Politeness indicators
- Aggression/toxicity
- Cultural references

**Output ONLY valid JSON in this exact format:**
{
  "artist_gender": "Male",
  "artist_region": "North America",
  "reasoning_steps": "1. First I noticed... 2. Then I observed... 3. This led me to conclude...",
  "attribute_scores": {
    "emotions": 7,
    "romance_topics": 8,
    "party_club": 3,
    "violence": 2,
    "politics_religion": 1,
    "success_money": 5,
    "family": 2,
    "slang_usage": 6,
    "formal_language": 2,
    "profanity": 4,
    "intensifiers": 5,
    "hedges": 2,
    "first_person": 9,
    "second_person": 7,
    "third_person": 3,
    "confidence": 6,
    "doubt_uncertainty": 2,
    "politeness": 1,
    "aggression_toxicity": 3,
    "cultural_references": 5
  }
}

CRITICAL: 
- All scores must be integers from 1 to 10
- artist_gender MUST be either "Male" or "Female" - nothing else is valid
- NO extra text before or after JSON"""

def build_prompt(lyrics: str):
    return [
        {"role": "system", "content": REASON_FIRST_SYSTEM},
        {"role": "user", "content": f"Analyze these lyrics:\n\n{lyrics[:1500]}"}
    ]

# ====================== HELPER FUNCTIONS ======================
ATTRIBUTE_NAMES = [
    "emotions", "romance_topics", "party_club", "violence", "politics_religion",
    "success_money", "family", "slang_usage", "formal_language", "profanity",
    "intensifiers", "hedges", "first_person", "second_person", "third_person",
    "confidence", "doubt_uncertainty", "politeness", "aggression_toxicity", "cultural_references"
]

def extract_json(text: str) -> dict:
    text = re.sub(r'```json\s*|\s*```', '', text)
    matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if "artist_gender" in parsed and "artist_region" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end+1])
            if "artist_gender" in parsed and "artist_region" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    return None

def normalize_gender(gender_str: str) -> str:
    if not gender_str or not isinstance(gender_str, str):
        return "Male"
    g = gender_str.lower().strip()
    if any(x in g for x in ["female", "woman", "girl", "she", "feminine", "femme", "lady"]):
        return "Female"
    return "Male"

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

def validate_scores(scores: dict) -> dict:
    return {k: max(1, min(10, int(v))) if isinstance(v, (int, float, str)) else 1 
            for k, v in scores.items()}

def parse_output(text: str) -> dict:
    if "assistant" in text.lower():
        text = text[text.lower().index("assistant") + len("assistant"):].strip()
    parsed = extract_json(text)
    if parsed:
        parsed["artist_gender"] = normalize_gender(parsed.get("artist_gender", "Unknown"))
        parsed["artist_region"] = normalize_region(parsed.get("artist_region", "Unknown"))
        if "attribute_scores" in parsed:
            parsed["attribute_scores"] = validate_scores(parsed["attribute_scores"])
        return parsed
    return {"artist_gender": "Unknown", "artist_region": "Unknown", "attribute_scores": {}}

# ====================== BATCHED INFERENCE ======================
def infer_batch(lyrics_list: list) -> list:
    results = []
    valid_indices, valid_lyrics = [], []
    
    for i, lyrics in enumerate(lyrics_list):
        if lyrics and len(str(lyrics).strip()) >= 10:
            valid_indices.append(i)
            valid_lyrics.append(str(lyrics).strip())
    
    if not valid_lyrics:
        return [{"artist_gender": "Unknown", "artist_region": "Unknown", 
                "attribute_scores": {}}] * len(lyrics_list)
    
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
                final[i] = {"artist_gender": "Unknown", "artist_region": "Unknown", "attribute_scores": {}}
        return final
        
    except Exception as e:
        print(f"Rank {RANK} batch error: {e}")
        return [{"artist_gender": "Unknown", "artist_region": "Unknown", 
                "attribute_scores": {}}] * len(lyrics_list)

def process_batch(indices: list, lyrics_list: list) -> list:
    results_parsed = infer_batch(lyrics_list)
    batch_results = []
    
    for i, idx in enumerate(indices):
        res = results_parsed[i]
        result = {"idx": idx}
        result["inferred_gender"] = res.get("artist_gender", "Unknown")
        result["inferred_region"] = res.get("artist_region", "Unknown")
        result["reasoning_steps"] = res.get("reasoning_steps", "")
        
        attr_scores = res.get("attribute_scores", {})
        total = 0
        for attr in ATTRIBUTE_NAMES:
            score = attr_scores.get(attr, 1)
            result[attr] = score
            total += score
        result["total_attribute_score"] = total
        result["toxicity_score"] = attr_scores.get("aggression_toxicity", 1)
        batch_results.append(result)
    
    return batch_results

# ====================== MAIN LOOP ======================
def get_rank_indices(total, rank, world_size):
    per_rank = total // world_size
    remainder = total % world_size
    start = rank * per_rank + min(rank, remainder)
    end = start + per_rank + (1 if rank < remainder else 0)
    return start, end

print_rank0(f"\n[REASONING-FIRST] Starting inference (batch={BATCH_SIZE})...")

start_idx, end_idx = get_rank_indices(total_rows, RANK, WORLD_SIZE)
my_indices = list(range(start_idx, end_idx))
print(f"Rank {RANK}: Processing rows {start_idx}-{end_idx-1} ({len(my_indices)} rows)")

if IS_DISTRIBUTED:
    dist.barrier()

my_results = []
num_batches = (len(my_indices) + BATCH_SIZE - 1) // BATCH_SIZE
pbar = tqdm(range(num_batches), desc=f"Rank {RANK}", disable=(RANK != 0 and IS_DISTRIBUTED))

for batch_idx in pbar:
    b_start = batch_idx * BATCH_SIZE
    b_end = min(b_start + BATCH_SIZE, len(my_indices))
    batch_indices = my_indices[b_start:b_end]
    batch_lyrics = [str(df.iloc[i]["lyrics"]) if pd.notna(df.iloc[i]["lyrics"]) else "" 
                    for i in batch_indices]
    
    batch_results = process_batch(batch_indices, batch_lyrics)
    my_results.extend(batch_results)
    
    if RANK == 0 and batch_idx == 0:
        for j, r in enumerate(batch_results[:DEBUG_FIRST_N]):
            print(f"\nROW {r['idx']}: Gender={r['inferred_gender']}, Region={r['inferred_region']}, "
                  f"Toxicity={r['toxicity_score']}")
            if r['reasoning_steps']:
                print(f"  Reasoning: {r['reasoning_steps'][:100]}...")

print(f"Rank {RANK}: Finished {len(my_results)} rows")

# ====================== GATHER & SAVE ======================
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
        idx = result.pop("idx")
        for key, value in result.items():
            df.at[idx, key] = value
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[REASONING-FIRST] Saved to {OUTPUT_PATH}")
    print(f"Gender:\n{df['inferred_gender'].value_counts()}")
    print(f"Region:\n{df['inferred_region'].value_counts()}")
    
    if 'artist_gender' in df.columns:
        df['gt_gender'] = df['artist_gender'].apply(normalize_gender)
        acc = (df['inferred_gender'] == df['gt_gender']).mean()
        print(f"Gender Accuracy: {acc:.4f}")

if IS_DISTRIBUTED:
    dist.barrier()
    cleanup_distributed()
print(f"Rank {RANK}: Done!")
