# src/feedback_io.py

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Constants for feedback file paths
data_dir = Path(__file__).resolve().parent.parent / "data"
sensor_fb_file = data_dir / "sensor_feedback.json"
pattern_fb_file = data_dir / "pattern_feedback.json"

def _ensure_file(path: Path, default: Any) -> None:
    """Ensure file exists with default content if missing."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    if not path.exists():
        path.write_text(json.dumps(default, indent=2))

def load_sensor_feedback() -> List[Dict[str, Any]]:
    """
    Load existing sensor feedback, or initialize/restore defaults if missing,
    empty, or invalid.
    """
    default: List[Dict[str, Any]] = []
    # Ensure data folder exists
    sensor_fb_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        content = sensor_fb_file.read_text()
        if not content.strip():
            # Empty file → write default
            sensor_fb_file.write_text(json.dumps(default, indent=2))
            return default
        # Try parsing
        return json.loads(content)
    except (json.JSONDecodeError, OSError):
        # Corrupted or unreadable → reset to default
        sensor_fb_file.write_text(json.dumps(default, indent=2))
        return default

def append_sensor_feedback(records: List[Dict[str, Any]]) -> None:
    """
    Append a list of sensor feedback records to the feedback file.
    """
    all_fb = load_sensor_feedback()
    all_fb.extend(records)
    sensor_fb_file.write_text(json.dumps(all_fb, indent=2))

def load_pattern_feedback() -> Dict[str, Any]:
    """
    Load pattern feedback JSON, initialize or restore defaults if missing, empty, or invalid.
    """
    default: Dict[str, Any] = {"missing": []}
    # Ensure the data directory exists
    if not pattern_fb_file.parent.exists():
        pattern_fb_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        content = pattern_fb_file.read_text()
        # Empty file? initialize with default
        if not content.strip():
            pattern_fb_file.write_text(json.dumps(default, indent=2))
            return default
        # Try to parse existing JSON
        data = json.loads(content)
        data.setdefault("missing", [])
        return data
    except (json.JSONDecodeError, OSError):
        # Invalid or unreadable file: reset to default
        pattern_fb_file.write_text(json.dumps(default, indent=2))
        return default

def append_pattern_feedback(new_patterns: List[str]) -> None:
    """
    Add missing patterns to the pattern feedback.
    """
    data = load_pattern_feedback()
    for p in new_patterns:
        if p not in data["missing"]:
            data["missing"].append(p)
    pattern_fb_file.write_text(json.dumps(data, indent=2))

def make_sensor_record(condition: str, definition: str, was_correct: bool, correction: str = None) -> Dict[str, Any]:
    """
    Helper to build a sensor feedback record.
    """
    return {
        "condition": condition,
        "definition": definition,
        "was_correct": was_correct,
        "correction": correction or "",
        "timestamp": datetime.utcnow().isoformat()
    }