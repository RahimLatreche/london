import os
import faiss
import pickle
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Tuple

# ——————————————————————
# 1) Load .env and instantiate client
# ——————————————————————
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Please set OPENAI_API_KEY in your .env file")
client = OpenAI(api_key=api_key)

# ——————————————————————
# 2) File paths
# ——————————————————————
BASE       = os.path.dirname(__file__)
MASTER_CSV = os.path.join(BASE, "../data/sensor_database.csv")
INDEX_FILE = os.path.join(BASE, "sensor_index.faiss")
META_FILE  = os.path.join(BASE, "sensor_meta.pkl")

def embed_in_batches(texts: List[str], batch_size: int = 256) -> List[List[float]]:
    """
    Splits `texts` into chunks of up to `batch_size` and calls the embeddings API.
    Returns a flat list of embeddings.
    """
    all_embs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        # resp.data is a list of CreateEmbeddingResponseData objects
        for record in resp.data:
            # each record has an .embedding attribute
            all_embs.append(record.embedding)
    return all_embs

def build_index() -> None:
    # ——————————————————————
    # 3) Load the master sensor CSV
    # ——————————————————————
    df = pd.read_csv(MASTER_CSV)

    # ——————————————————————
    # 4) Build per-sensor text: name | markers | imperial units | metric units
    # ——————————————————————
    texts: List[str] = (
        df["Display Name"].fillna("") +
        " | " + df["Markers"].fillna("") +
        " | " + df.get("Units/Facets (Imperial)", pd.Series("")).fillna("") +
        " | " + df.get("Units/Facets (Metric)", pd.Series("")).fillna("")
    ).tolist()
    ids: List[str] = df["Definition"].tolist()

    # ——————————————————————
    # 5) Filter out any blank texts
    # ——————————————————————
    nonblank: List[Tuple[str, str]] = [
        (txt, sid) for txt, sid in zip(texts, ids) if txt.strip()
    ]
    if not nonblank:
        raise RuntimeError("No non-empty texts found to embed")
    texts, ids = zip(*nonblank)
    texts, ids = list(texts), list(ids)

    # ——————————————————————
    # 6) Embed in batches to avoid request size limits
    # ——————————————————————
    embs = embed_in_batches(texts, batch_size=256)

    # ——————————————————————
    # 7) Build a FAISS L2 index
    # ——————————————————————
    dim = len(embs[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embs, dtype="float32"))

    # ——————————————————————
    # 8) Persist the index and ID list
    # ——————————————————————
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(list(ids), f)

    print(f"✅ Built index with {len(ids)} vectors.")

if __name__ == "__main__":
    build_index()
