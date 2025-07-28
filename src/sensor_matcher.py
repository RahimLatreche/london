# src/sensor_matcher.py

import os
import re
import faiss
import pickle
import requests
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

from auth import get_access_token       # make sure src/auth.py defines this

# ——————————————————————
# 1) Fetch OAuth2 Bearer once
# ——————————————————————
ACCESS_TOKEN = get_access_token()

# ——————————————————————
# 2) Embedding proxy settings
#    from OpenAIInstructions.pdf (non‑prod)
# ——————————————————————
EMBED_DEPLOYMENT_ID = "AOAIsharednonprodtxtembeddingada002"
EMBED_API_VERSION   = "2024-02-15-preview"

def embed_via_wso2(phrase: str) -> List[float]:
    """
    Call the CBRE WSO2 proxy to get an embedding for `phrase`.
    """
    url = (
        f"https://api-test.cbre.com:443/"
        f"t/digitaltech_us_edp/cbreopenaiendpoint/1/"
        f"openai/deployments/{EMBED_DEPLOYMENT_ID}/embeddings"
        f"?api-version={EMBED_API_VERSION}"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    body = {"model": "text-embedding-ada-002", "input": [phrase]}
    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

# ——————————————————————
# 3) Paths & pre‑loads
# ——————————————————————
BASE             = os.path.dirname(__file__)
MASTER_CSV       = os.path.join(BASE, "../data/sensor_database.csv")
META_CSV         = os.path.join(BASE, "../data/comerica_rtu_economizing_cooling_simultaneously_meta_data.csv")
INDEX_FILE       = os.path.join(BASE, "sensor_index.faiss")
SENSOR_META_FILE = os.path.join(BASE, "sensor_meta.pkl")
LEARNED_PATTERNS_FILE = os.path.join(BASE, "learned_patterns.json")

DF_MASTER_FULL = pd.read_csv(MASTER_CSV)
DF_META_FULL   = pd.read_csv(META_CSV)

_index      = None
_sensor_ids = None

def _load_index():
    global _index, _sensor_ids
    if _index is None:
        _index = faiss.read_index(INDEX_FILE)
        with open(SENSOR_META_FILE, "rb") as f:
            _sensor_ids = pickle.load(f)

_load_index()

# ——————————————————————
# 4) Concepts & stopwords (expanded for full "Economizing and Cooling Simultaneously" rule)
# ——————————————————————
PATTERNS = [
    "discharge fan",
    "supply fan",  # Adding related fan patterns
    "exhaust fan",
    "return fan",
    "outdoor damper",
    "outdoor temperature", 
    "return temperature",
    "cooling valve",
    "face bypass damper",
    "outside airflow",
    "minimum outside airflow",
    "return co2",
    "return co2 setpoint",
    "cold water pump",
    "communication alarm",
    # NEW: Add more specific patterns to prevent single-word matches
    "outside airflow setpoint",
    "minimum outside airflow setpoint",
    "cooling valves",
    "cold water pumps",
    "zone co2",
    "zone co2 setpoint",
    # Remove control keywords from patterns
    # "andModes",
    # "orModes",
]

# Enhanced stopwords to filter out control words and problematic terms
STOPWORDS = {
    # original handful…
    "and", "the", "more", "than", "below", "open", "threshold",
    # plus supplements to filter out control‑flow words
    "unit", "has", "by", "for", "over", "periods", "when", "if",
    "only", "used", "also", "make", "sure", "associated", "points",
    "are", "all", "on", "any",
    # NEW: Control keywords that shouldn't be treated as sensors
    "andmodes", "ormodes", "modes", "active", "must", "should",
    "will", "based", "using", "following", "required", "during",
    "true", "false", "above", "between", "either",
    "spark", "rule", "finds", "check", "verify", "ensure",
    # Additional problematic single words from results
    "cold", "hot", "warm", "cool",  # Too generic without context
    # MORE stopwords from the bad matches
    "this", "that", "have", "found", "requires", "simultaneously",
    "parameter", "duration", "inside", "economizing", "cooling",
    "water", "pumps", "valves", "record", "from", "with", "into",
    "been", "were", "their", "these", "those", "through", "where",
    "which", "while", "within", "would", "could", "shall", "might",
    "being", "having", "doing", "saying", "going", "making", "getting",
    "finding", "periods", "times", "uses", "needs", "means", "ways",
    "types", "kinds", "forms", "parts", "areas", "cases", "things",
    "face",  # Too generic - need "face bypass damper" as full pattern
}

# ——————————————————————
# 5) Pattern Learning Functions
# ——————————————————————
def load_learned_patterns() -> Set[str]:
    """Load previously learned successful patterns."""
    patterns = set()
    
    # Load from multiple sources
    files_to_check = [
        os.path.join(BASE, "learned_patterns.json"),
        os.path.join(BASE, "auto_patterns.json"),
        os.path.join(BASE, "mined_patterns.json")
    ]
    
    for filepath in files_to_check:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # Handle different file formats
                    if 'patterns' in data:
                        if isinstance(data['patterns'], list):
                            patterns.update(data['patterns'])
                        elif isinstance(data['patterns'], dict):
                            # From mined_patterns.json
                            patterns.update(data['patterns'].get('high_confidence', []))
                            patterns.update(data['patterns'].get('compounds', []))
            except Exception as e:
                print(f"Warning: Failed to load patterns from {filepath}: {e}")
    
    return patterns

def load_learned_stopwords() -> Set[str]:
    """Load automatically learned stopwords."""
    stopwords = set()
    
    # Load from multiple sources
    files_to_check = [
        os.path.join(BASE, "learned_stopwords.json"),
        os.path.join(BASE, "auto_stopwords.json"),
        os.path.join(BASE, "mined_patterns.json")
    ]
    
    for filepath in files_to_check:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # Handle different file formats
                    if 'stopwords' in data:
                        if isinstance(data['stopwords'], list):
                            stopwords.update(data['stopwords'])
                        elif isinstance(data['stopwords'], dict):
                            # From mined_patterns.json
                            stopwords.update(data['stopwords'].get('generated', []))
            except Exception as e:
                print(f"Warning: Failed to load stopwords from {filepath}: {e}")
    
    return stopwords

def save_learned_patterns(patterns: Set[str]):
    """Save learned patterns for future use."""
    try:
        existing = load_learned_patterns()
        existing.update(patterns)
        with open(LEARNED_PATTERNS_FILE, 'w') as f:
            json.dump({'patterns': list(existing)}, f, indent=2)
    except:
        pass

def extract_common_patterns_from_display_names(min_count: int = 5) -> List[str]:
    """Extract common multi-word patterns from Display Names in the database."""
    # This would run periodically to update patterns
    display_names = DF_MASTER_FULL['Display Name'].dropna()
    
    # Extract 2-3 word combinations that appear frequently
    two_word_patterns = []
    three_word_patterns = []
    
    for name in display_names:
        words = name.lower().split()
        # Two-word patterns
        for i in range(len(words) - 1):
            two_word_patterns.append(f"{words[i]} {words[i+1]}")
        # Three-word patterns
        for i in range(len(words) - 2):
            three_word_patterns.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    # Count occurrences
    from collections import Counter
    two_word_counts = Counter(two_word_patterns)
    three_word_counts = Counter(three_word_patterns)
    
    # Return patterns that appear at least min_count times
    common_patterns = []
    common_patterns.extend([p for p, c in two_word_counts.items() if c >= min_count])
    common_patterns.extend([p for p, c in three_word_counts.items() if c >= min_count])
    
    return common_patterns

# Combine static patterns with learned patterns
DYNAMIC_PATTERNS = PATTERNS + list(load_learned_patterns())

# Combine static stopwords with learned stopwords
DYNAMIC_STOPWORDS = STOPWORDS.union(load_learned_stopwords())

RAW_K_MULTIPLIER = 10

# ——————————————————————
# 6) Fallback Mappings for Missing Metadata
# ——————————————————————
# When a pattern doesn't find matches in metadata, use these fallback mappings
# To customize: Run find_available_sensors_in_metadata() to see what's available,
# then update these mappings accordingly.

SENSOR_FALLBACK_MAPPINGS = {
    # Pattern in rule -> Sensor patterns to search for instead
    "discharge fan": [
        "discharge pressure",     # Fan status can be inferred from pressure
        "discharge air",         # General discharge air sensors
        "supply fan",           # Sometimes discharge = supply
        "fan status",          # Generic fan status
        "discharge",          # Broader search
    ],
    
    "cooling valve": [
        "cooling coil",          # Valve position might be on coil
        "chilled water valve",   # More specific name
        "cooling",              # Broader search
        "chw valve",           # Abbreviated form
    ],
    
    "face bypass damper": [
        "face damper",
        "bypass damper",
        "economizer damper",    # Sometimes used interchangeably
        "mixed air damper",
    ],
    
    "outside airflow": [
        "outdoor airflow",
        "oa flow",             # Common abbreviation
        "outside air cfm",
        "ventilation flow",
    ],
    
    "communication alarm": [
        "comm alarm",
        "communication status",
        "comm fault",
        "network alarm",
        "device status",
    ],
}

# ——————————————————————
# 7) FAISS‐based lookup (via WSO2 embeddings)
# ——————————————————————
def match_sensors_vector(phrase: str, top_k: int = 5) -> List[str]:
    _load_index()
    vec = embed_via_wso2(phrase)
    emb = np.array(vec, dtype="float32")[None, :]
    _, idxs = _index.search(emb, top_k)
    return [_sensor_ids[i] for i in idxs[0]]

# ——————————————————————
# 8) Core matcher (IMPROVED with Route 1 and Fallbacks)
# ——————————————————————
def match_sensors(
    rule_text: str,
    equipment: str,
    top_k: int = 5
) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict]]]:
    """
    Match sensors using Route 1: Vector search first, then filter.
    Now includes fallback mappings for patterns not in metadata.
    
    Returns:
        - results: Dict mapping patterns to matched sensor IDs
        - missing_metadata: Dict mapping patterns to sensors filtered out due to missing metadata
    """
    text = rule_text.lower().strip()
    
    # Get metadata navNames for filtering
    navnames = set(DF_META_FULL["navName"])
    
    # Get all sensor IDs that match the equipment type (from full master)
    equipment_mask = DF_MASTER_FULL["Equipment"].str.contains(
        fr"\b{equipment}\b", 
        case=False, 
        na=False
    )
    equipment_sensor_ids = set(DF_MASTER_FULL.loc[equipment_mask, "Definition"])
    
    if not equipment_sensor_ids:
        raise ValueError(f"No sensors found for equipment={equipment!r}")
    
    # Get sensor IDs that are both in equipment AND metadata
    metadata_sensor_ids = set(
        DF_MASTER_FULL.loc[
            DF_MASTER_FULL["Display Name"].isin(navnames), 
            "Definition"
        ]
    )
    valid_ids = equipment_sensor_ids & metadata_sensor_ids
    
    results = {}
    missing_metadata = {}
    matched = set()
    
    # Process multi‑word patterns
    for pat in DYNAMIC_PATTERNS:
        if re.search(rf"\b{re.escape(pat)}\b", text):
            # ROUTE 1: Search ALL sensors first
            raw = match_sensors_vector(pat, top_k * RAW_K_MULTIPLIER)
            
            # Filter by equipment
            equipment_filtered = [sid for sid in raw if sid in equipment_sensor_ids]
            
            # Filter by metadata availability
            metadata_filtered = [sid for sid in equipment_filtered if sid in valid_ids][:top_k]
            
            # Track what was filtered out due to missing metadata
            filtered_out = [
                sid for sid in equipment_filtered 
                if sid not in valid_ids
            ][:5]  # Keep top 5 filtered results
            
            # If no matches found in metadata, try fallback mappings
            if not metadata_filtered and pat in SENSOR_FALLBACK_MAPPINGS:
                print(f"No metadata matches for '{pat}', trying fallbacks...")
                fallback_patterns = SENSOR_FALLBACK_MAPPINGS[pat]
                
                for fallback_pat in fallback_patterns[:3]:  # Try top 3 fallbacks
                    raw_fallback = match_sensors_vector(fallback_pat, top_k * RAW_K_MULTIPLIER)
                    equipment_filtered_fb = [sid for sid in raw_fallback if sid in equipment_sensor_ids]
                    metadata_filtered_fb = [sid for sid in equipment_filtered_fb if sid in valid_ids][:top_k]
                    
                    if metadata_filtered_fb:
                        metadata_filtered = metadata_filtered_fb
                        print(f"  Using fallback: '{pat}' -> '{fallback_pat}' (found {len(metadata_filtered_fb)} matches)")
                        # Store that we used a fallback (useful for learning)
                        if not hasattr(match_sensors, 'fallback_usage'):
                            match_sensors.fallback_usage = {}
                        match_sensors.fallback_usage[pat] = fallback_pat
                        break
                
                if not metadata_filtered:
                    print(f"  No matches found even with fallbacks")
            
            results[pat] = metadata_filtered
            
            if filtered_out:
                # Get display names for filtered sensors
                filtered_info = []
                for sid in filtered_out:
                    try:
                        row = DF_MASTER_FULL.loc[DF_MASTER_FULL["Definition"] == sid].iloc[0]
                        filtered_info.append({
                            'definition': sid,
                            'display_name': row['Display Name'],
                            'markers': row.get('Markers', ''),
                            'equipment': row.get('Equipment', '')
                        })
                    except:
                        continue
                
                if filtered_info:
                    missing_metadata[pat] = filtered_info
            
            matched.update(pat.split())
    
    # Process single‑word fallback
    tokens = re.findall(r"\b[a-z]{4,}\b", text)
    # Filter more aggressively - only keep potential sensor-related words
    leftover = [
        w for w in set(tokens) 
        if w not in matched 
        and w.lower() not in DYNAMIC_STOPWORDS
        and len(w) >= 5  # Increased minimum length
        and not w.isdigit()  # No pure numbers
        and any(char in w for char in 'aeiou')  # Must have vowels (real words)
    ]
    
    # Additional filter: Only process words that might be sensor-related
    sensor_keywords = {'temp', 'press', 'flow', 'speed', 'alarm', 'fault', 'status', 
                      'sensor', 'meter', 'valve', 'damper', 'setpoint', 'mode'}
    leftover = [w for w in leftover if any(kw in w for kw in sensor_keywords)]
    
    for w in leftover:
        # Same Route 1 approach for single words
        raw = match_sensors_vector(w, top_k * RAW_K_MULTIPLIER)
        equipment_filtered = [sid for sid in raw if sid in equipment_sensor_ids]
        metadata_filtered = [sid for sid in equipment_filtered if sid in valid_ids][:top_k]
        
        filtered_out = [
            sid for sid in equipment_filtered 
            if sid not in valid_ids
        ][:5]
        
        results[w] = metadata_filtered
        
        if filtered_out:
            filtered_info = []
            for sid in filtered_out:
                try:
                    row = DF_MASTER_FULL.loc[DF_MASTER_FULL["Definition"] == sid].iloc[0]
                    filtered_info.append({
                        'definition': sid,
                        'display_name': row['Display Name'],
                        'markers': row.get('Markers', ''),
                        'equipment': row.get('Equipment', '')
                    })
                except:
                    continue
            
            if filtered_info:
                missing_metadata[w] = filtered_info
    
    return results, missing_metadata

# ——————————————————————
# 9) Best candidate pick (ENHANCED)
# ——————————————————————
def match_and_choose(
    rule_text: str,
    equipment: str,
    top_k: int = 5,
    return_missing: bool = True
) -> Tuple[Dict[str, List[str]], Dict[str, str], Optional[Dict[str, List[Dict]]]]:
    """
    Enhanced version that returns missing metadata information.
    
    Returns:
        - candidates: Dict of pattern -> list of sensor IDs
        - best: Dict of pattern -> best sensor ID
        - missing_metadata: Dict of pattern -> list of filtered sensors (if return_missing=True)
    """
    candidates, missing_metadata = match_sensors(rule_text, equipment, top_k)
    best = {c: (l[0] if l else None) for c, l in candidates.items()}
    
    if return_missing:
        return candidates, best, missing_metadata
    else:
        return candidates, best

# ——————————————————————
# 10) Utility function to update patterns based on successful matches
# ——————————————————————
def record_successful_match(pattern: str, sensor_id: str, confidence: float = 1.0):
    """
    Record a successful pattern match for future learning.
    This would be called when users confirm a match is correct.
    """
    # In a production system, this would:
    # 1. Log the successful match
    # 2. Update pattern weights
    # 3. Potentially add new patterns if confidence is high
    
    if confidence > 0.8 and len(pattern.split()) > 1:
        save_learned_patterns({pattern})

# ——————————————————————
# 11) Helper function to find available sensors in metadata
# ——————————————————————
def find_available_sensors_in_metadata(keywords: List[str]) -> Dict[str, List[str]]:
    """
    Find what sensors are available in metadata matching given keywords.
    Useful for updating fallback mappings.
    
    Example:
        find_available_sensors_in_metadata(["discharge", "pressure", "fan"])
    """
    available = {}
    navnames = DF_META_FULL["navName"].dropna()
    
    for keyword in keywords:
        matching = navnames[
            navnames.str.contains(keyword, case=False, na=False)
        ].unique().tolist()
        
        if matching:
            available[keyword] = matching[:10]  # Top 10 matches
            print(f"\n'{keyword}' - Found {len(matching)} matches:")
            for sensor in matching[:5]:
                print(f"  - {sensor}")
            if len(matching) > 5:
                print(f"  ... and {len(matching) - 5} more")
    
    return available

# ——————————————————————
# 12) Demo (UPDATED)
# ——————————————————————
if __name__ == "__main__":
    # First, explore what's available in metadata for discharge fan alternatives
    print("=== EXPLORING METADATA FOR ALTERNATIVES ===")
    available = find_available_sensors_in_metadata([
        "discharge", "pressure", "fan", "supply", "status"
    ])
    
    print("\n" + "="*50 + "\n")
    
    RULE = (
        "Discharge fan is on, outdoor damper is open more than a threshold, "
        "cooling is on, and return temperature is below the outdoor temperature."
    )
    
    # Use the enhanced version
    candidates, best, missing = match_and_choose(
        RULE, 
        equipment="AHU", 
        top_k=3,
        return_missing=True
    )

    print("=== MATCHES FOUND ===")
    for cond, defs in candidates.items():
        print(f"\nCondition: {cond!r}")
        unique_defs = list(dict.fromkeys(defs))
        if not unique_defs:
            print("   -> no candidates")
            continue
        for sid in unique_defs:
            disp = DF_MASTER_FULL.loc[
                DF_MASTER_FULL["Definition"] == sid, "Display Name"
            ].iat[0]
            meta_ids = DF_META_FULL.loc[
                DF_META_FULL["navName"] == disp, "id"
            ].tolist()
            print(f"   - Definition:    {sid}")
            print(f"     Display Name:  {disp}")
            print(f"     Metadata IDs:  {meta_ids}")
        print(f"   -> best overall: {best[cond]}")
    
    # Show missing metadata sensors
    if missing:
        print("\n\n=== SENSORS FOUND BUT MISSING FROM METADATA ===")
        for cond, filtered_sensors in missing.items():
            print(f"\nCondition: {cond!r}")
            for sensor in filtered_sensors:
                print(f"   - Definition:    {sensor['definition']}")
                print(f"     Display Name:  {sensor['display_name']}")
                print(f"     Equipment:     {sensor['equipment'][:50]}...")