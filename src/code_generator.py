# src/code_generator.py
import os
import openai
import json
import logging
from typing import Dict, Any, List
import pandas as pd

# Configure module logger
dlogger = logging.getLogger(__name__)

def generate_python_code(
    rule_text: str,
    best: Dict[str, str],
    df_master: pd.DataFrame,
    df_meta: pd.DataFrame,
    duration_minutes: int = None,
    model: str = "gpt-4o-code"
) -> str:
    """
    Generate Python code that implements the given rule as a function.
    - rule_text: the original free-text rule description.
    - best: mapping from pattern to best sensor Definition code.
    - df_master: master sensor dataframe.
    - df_meta: metadata dataframe.
    - duration_minutes: optional duration for continuous violation.
    Returns: generated Python code as string.
    """
    # Build sensor list for the prompt, with safe lookups
    sensors: List[Dict[str, Any]] = []
    for pattern, def_id in best.items():
        # Safely find display name
        dfm = df_master.loc[df_master["Definition"] == def_id, "Display Name"]
        if dfm.empty:
            dlogger.warning(f"Definition code '{def_id}' not found in master DataFrame.")
            display_name = ""
            meta_ids: List[str] = []
        else:
            display_name = dfm.iat[0]
            # Safely find metadata IDs
            meta_ids = df_meta.loc[df_meta["navName"] == display_name, "id"].tolist()
        sensors.append({
            "pattern": pattern,
            "definition": def_id,
            "display_name": display_name,
            "metadata_ids": meta_ids
        })

    # Prepare the LLM prompt
    prompt_lines = [
        "You are a Python developer. Write a function `check_rule(df: pandas.DataFrame) -> pandas.DataFrame` that:",
        f"- takes a DataFrame `df` with a DateTimeIndex and sensor columns named by their Definition codes.",
        f"- implements the rule: '{rule_text}'.",
        f"- uses these sensors: {json.dumps(sensors, indent=2)}"
    ]
    if duration_minutes:
        prompt_lines.append(f"- the rule must hold continuously for {duration_minutes} minutes.")
    prompt_lines.append(
        "The function should return a DataFrame (or its index) of timestamps where the rule is true continuously."
    )
    prompt = "\n".join(prompt_lines)

    # Call the OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Generate production-quality Python code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        dlogger.error(f"Failed to generate code via OpenAI: {e}")
        raise



