# src/index_sensors.py

import os
import faiss
import pickle
import numpy as np
import pandas as pd
import requests

from typing import List, Tuple
from auth import get_access_token    # your OAuth2 helper in src/auth.py

# ——————————————————————
# 1) OAuth2 bearer + proxy settings
# ——————————————————————
ACCESS_TOKEN        = get_access_token()
EMBED_DEPLOYMENT_ID = "AOAIsharednonprodtxtembeddingada002"
EMBED_API_VERSION   = "2024-02-15-preview"

# Full URL template for embedding calls
EMBED_URL = (
    "https://api-test.cbre.com:443/"
    "t/digitaltech_us_edp/cbreopenaiendpoint/1/"
    "openai/deployments/"
    f"{EMBED_DEPLOYMENT_ID}/embeddings"
    f"?api-version={EMBED_API_VERSION}"
)

# ——————————————————————
# 2) File paths
# ——————————————————————
BASE       = os.path.dirname(__file__)
MASTER_CSV = os.path.join(BASE, "../data/sensor_database.csv")
INDEX_FILE = os.path.join(BASE, "sensor_index.faiss")
META_FILE  = os.path.join(BASE, "sensor_meta.pkl")

# ——————————————————————
# 3) Batch-Embedding via WSO2
# ——————————————————————
def embed_in_batches(texts: List[str], batch_size: int = 256) -> List[List[float]]:
    """
    Splits `texts` into chunks of up to `batch_size` and calls the WSO2 proxy
    embeddings endpoint. Returns a flat list of embedding vectors.
    """
    all_embs: List[List[float]] = []
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        body = {
            "model": "text-embedding-ada-002",
            "input": chunk
        }
        resp = requests.post(EMBED_URL, headers=headers, json=body)
        resp.raise_for_status()

        for record in resp.json()["data"]:
            all_embs.append(record["embedding"])
    return all_embs

# ——————————————————————
# 4) Build and persist FAISS index
# ——————————————————————
def build_index() -> None:
    # 4.1 Load your master sensor dictionary
    df = pd.read_csv(MASTER_CSV)

    # 4.2 Turn each row into a single text blob
    texts: List[str] = (
        df["Display Name"].fillna("") +
        " | " + df["Markers"].fillna("") +
        " | " + df.get("Units/Facets (Imperial)", pd.Series("")).fillna("") +
        " | " + df.get("Units/Facets (Metric)", pd.Series("")).fillna("")
    ).tolist()
    ids: List[str] = df["Definition"].tolist()

    # 4.3 Drop any rows where the combined text is empty
    nonblank: List[Tuple[str, str]] = [
        (txt, sid) for txt, sid in zip(texts, ids) if txt.strip()
    ]
    if not nonblank:
        raise RuntimeError("No non-empty texts found to embed")
    texts, ids = zip(*nonblank)

    # 4.4 Embed in batches
    embs = embed_in_batches(list(texts), batch_size=256)

    # 4.5 Build FAISS L2 index
    dim = len(embs[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embs, dtype="float32"))

    # 4.6 Persist index and ID mapping
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(list(ids), f)

    print(f"✅ Built index with {len(ids)} vectors.")

# ——————————————————————
# 5) CLI
# ——————————————————————
if __name__ == "__main__":
    build_index()



