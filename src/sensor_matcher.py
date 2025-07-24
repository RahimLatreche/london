import os
import re
import faiss
import pickle
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Tuple

# ——————————————————————
# 1) Load env & paths
# ——————————————————————
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your .env")

openai_client = OpenAI(api_key=API_KEY)

BASE             = os.path.dirname(__file__)
MASTER_CSV       = os.path.join(BASE, "../data/sensor_database.csv")
META_CSV         = os.path.join(BASE, "../data/comerica_rtu_economizing_cooling_simultaneously_meta_data.csv")
INDEX_FILE       = os.path.join(BASE, "sensor_index.faiss")
SENSOR_META_FILE = os.path.join(BASE, "sensor_meta.pkl")

# ——————————————————————
# 2) Prime DataFrames
# ——————————————————————
DF_MASTER_FULL = pd.read_csv(MASTER_CSV)
DF_META_FULL   = pd.read_csv(META_CSV)

# ——————————————————————
# 3) Load FAISS index + ID map
# ——————————————————————
_index     = None
_sensor_ids = None

def _load_index():
    global _index, _sensor_ids
    if _index is None:
        _index = faiss.read_index(INDEX_FILE)
        with open(SENSOR_META_FILE, "rb") as f:
            _sensor_ids = pickle.load(f)

_load_index()

# ——————————————————————
# 4) Your concepts & stopwords
# ——————————————————————
PATTERNS = [
    "discharge fan",
    "outdoor damper",
    "outdoor temperature",
    "return temperature",
    "cooling valve",
    "face bypass damper",
]

STOPWORDS = {
    "and", "the", "more", "than", "below", "open", "threshold",
}

# **This multiplier gives us a wider net to catch 'dischargePressure'**
RAW_K_MULTIPLIER = 10

# ——————————————————————
# 5) FAISS lookup helper
# ——————————————————————
def match_sensors_vector(phrase: str, top_k: int = 5) -> List[str]:
    _load_index()
    resp = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=[phrase]
    )
    emb = np.array(resp.data[0].embedding, dtype="float32")[None, :]
    _, idxs = _index.search(emb, top_k)
    return [_sensor_ids[i] for i in idxs[0]]

# ——————————————————————
# 6) Core matcher
# ——————————————————————
def match_sensors(
    rule_text: str,
    equipment: str,
    top_k: int = 5
) -> Dict[str, List[str]]:
    text = rule_text.lower().strip()

    # a) Narrow to *exactly* those sensors your metadata file lists
    navnames    = set(DF_META_FULL["navName"])
    df_relevant = DF_MASTER_FULL[
        DF_MASTER_FULL["Display Name"].isin(navnames)
    ].copy()

    # b) Then scope to the requested equipment (AHU, RTU, etc.)
    df_relevant = df_relevant[
        df_relevant["Equipment"]
                  .str.contains(fr"\b{equipment}\b", na=False)
    ]
    if df_relevant.empty:
        raise ValueError(f"No sensors found for equipment={equipment!r}")

    valid_ids = set(df_relevant["Definition"])
    results   = {}
    matched   = set()

    # c) Phrase‑level matching, with a larger raw window
    for pat in PATTERNS:
        if re.search(rf"\b{re.escape(pat)}\b", text):
            raw      = match_sensors_vector(pat, top_k * RAW_K_MULTIPLIER)
            filtered = [sid for sid in raw if sid in valid_ids][:top_k]
            results[pat] = filtered
            matched.update(pat.split())

    # d) Single‑word fallback (≥4 letters, no stopwords)
    tokens   = re.findall(r"\b[a-z]{4,}\b", text)
    leftover = [w for w in set(tokens) if w not in matched and w not in STOPWORDS]

    for w in leftover:
        raw      = match_sensors_vector(w, top_k * RAW_K_MULTIPLIER)
        filtered = [sid for sid in raw if sid in valid_ids][:top_k]
        results[w] = filtered

    return results

# ——————————————————————
# 7) Pick the single best candidate
# ——————————————————————
def match_and_choose(
    rule_text: str,
    equipment: str,
    top_k: int = 5
) -> Tuple[Dict[str,List[str]], Dict[str,str]]:
    candidates = match_sensors(rule_text, equipment, top_k)
    best = {cond: (cands[0] if cands else None)
            for cond, cands in candidates.items()}
    return candidates, best

# ——————————————————————
# 8) Demo
# ——————————————————————
if __name__ == "__main__":
    RULE = (
        "Discharge fan is on, outdoor damper is open more than a threshold, "
        "cooling is on, and return temperature is below the outdoor temperature."
    )
    cands, best = match_and_choose(RULE, equipment="AHU", top_k=3)

    for cond in cands:
        print(f"{cond:20} → candidates: {cands[cond]},  best: {best[cond]}")
