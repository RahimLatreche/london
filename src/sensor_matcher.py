import os
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, List

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# File paths
MASTER_CSV = os.path.join(os.path.dirname(__file__), "../data/sensor_database.csv")
META_CSV   = os.path.join(os.path.dirname(__file__), "../data/comerica_rtu_economizing_cooling.csv")

# Load data
df_master = pd.read_csv(MASTER_CSV)
df_meta   = pd.read_csv(META_CSV)

def match_sensors(rule_text: str) -> Dict[str, List[str]]:
    """
    Given a free-text rule, split into keywords and find candidate sensors.
    Returns a mapping from each keyword to a list of matching sensor Definitions.
    """
    # Basic keyword extraction
    keywords = [kw.strip() for kw in rule_text.lower().replace(",", " ").split() if len(kw) >= 3]
    results: Dict[str, List[str]] = {}

    for kw in set(keywords):
        # search in Display Name and Markers columns
        mask = (
            df_master["Display Name"].str.lower().str.contains(kw, na=False) |
            df_master["Markers"].str.lower().str.contains(kw, na=False)
        )
        # collect the Definition values for matches
        matched_defs = df_master.loc[mask, "Definition"].dropna().unique().tolist()
        results[kw] = matched_defs

    return results

if __name__ == "__main__":
    # Example rule
    RULE = """
    Discharge fan is on, outdoor damper open > threshold,
    cooling on, return temp below outdoor temp by threshold.
    """
    matches = match_sensors(RULE)
    for kw, defs in matches.items():
        print(f"{kw}: {defs}")
