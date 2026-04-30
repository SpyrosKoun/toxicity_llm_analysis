import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from json_repair import repair_json

from ollama import chat

from prompts import render_prompt

MODEL = "gemma3:4b"
TEMPLATE_NAME = "improved_en_v1"
DATA_FILE = "data/final_labeled.csv"
OUTPUT_FILE = f"data/{MODEL}_{TEMPLATE_NAME}.csv"
MISMATCH_FILE = f"data/{MODEL}_{TEMPLATE_NAME}_mismatches.csv"

log_path = Path("logs") / (Path(OUTPUT_FILE).stem + ".log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

data_csv = pd.read_csv(DATA_FILE)
# data_csv = data_csv[:100]
results = []
mismatches = []

log.info(f"Loaded {len(data_csv)} samples from {DATA_FILE}")

for idx, row in tqdm(data_csv.iterrows(), total=len(data_csv)):
    text = row['text']
    prompt = render_prompt(template_name=TEMPLATE_NAME, text=text)

    try:
        response = chat(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
        )
        content = response.message.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        content = repair_json(content)
        
        parsed = json.loads(content)
        classification = parsed['classification']
        explanation = parsed['explanation']
    except (json.JSONDecodeError, KeyError) as e:
        mismatches.append((idx, text, content, str(e)))
        log.warning(f"Row {idx} response: {content}")
        log.error(f"Row {idx}: invalid response — {e}")
        classification = "error"
        explanation = f"Parse error: {e}"

    results.append({
        'text': text,
        'classification': classification,
        'explanation': explanation,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)
log.info(f"Saved {len(results_df)} results to {OUTPUT_FILE}")

if mismatches:
    mismatches_df = pd.DataFrame(mismatches, columns=['index', 'text', 'response', 'error'])
    mismatches_df.to_csv(MISMATCH_FILE, index=False)
    log.info(f"Saved {len(mismatches_df)} mismatches to {MISMATCH_FILE}")