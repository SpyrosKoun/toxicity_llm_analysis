import json
import logging

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from json_repair import repair_json

from prompts import render_prompt

# ── Backend selection ──────────────────────────────────────────────────────────
# Set to "ollama" or "huggingface"
BACKEND = "ollama"

OLLAMA_MODEL = "qwen2.5:7b"
HF_MODEL = "Qwen/Qwen3-4B"

MODEL = OLLAMA_MODEL if BACKEND == "ollama" else HF_MODEL
TEMPLATE_NAME = "improved_en_v1"
DATA_FILE = "data/final_labeled.csv"

_model_slug = MODEL.replace("/", "-")
OUTPUT_FILE = f"data/{_model_slug}_{TEMPLATE_NAME}.csv"
MISMATCH_FILE = f"data/{_model_slug}_{TEMPLATE_NAME}_mismatches.csv"

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


# ── Backend initialisation ─────────────────────────────────────────────────────

def _init_ollama():
    from ollama import chat as ollama_chat
    def infer(prompt: str) -> str:
        response = ollama_chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False
        )
        return response.message.content
    return infer


def _init_huggingface():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, device_map="auto")

    def infer(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        new_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]

    return infer


_backends = {
    "ollama": _init_ollama,
    "huggingface": _init_huggingface,
}

if BACKEND not in _backends:
    raise ValueError(f"Unknown backend '{BACKEND}'. Choose from: {list(_backends)}")

log.info(f"Initialising backend '{BACKEND}' with model '{MODEL}'")
infer = _backends[BACKEND]()


# ── Main loop ──────────────────────────────────────────────────────────────────

data_csv = pd.read_csv(DATA_FILE)
# data_csv = data_csv[:100]
results = []
mismatches = []

log.info(f"Loaded {len(data_csv)} samples from {DATA_FILE}")

for idx, row in tqdm(data_csv.iterrows(), total=len(data_csv)):
    text = row['text']
    prompt = render_prompt(template_name=TEMPLATE_NAME, text=text)

    try:
        raw = infer(prompt)
        content = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
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
