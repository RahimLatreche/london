"""
Experiment: LLM-based Phrase Extraction
Run a few-shot prompt against CBRE OpenAI proxy to extract sensor phrases from a hardcoded rule description.
Usage:
 python src/experiment_patterns.py
"""
import json
import requests
from auth import get_access_token


# Rule description to extract from
rule_text = (
    "Finds periods when discharge fan is on, cooling valve is closed"
    "and discharge temperature sensor is under mixed air sensor by a threshold for over a duration."
    "Will also spark using the following, based on the points that are found:"
    "If a cooling stage was found, will make sure it is inactive during this period."
    "If target is a Dual Duct unit and has the `multiDuct` tag, will also account for points in multiple ducts."
    "If a cooling coil discharge temperature and/or heating coil discharge temperature exists, will determine the entering and leaving temperatures based off them."
)


# Few-shot prompt
prompt = """
Extract key sensor phrases from the following rule description. Each phrase should capture a noun or noun phrase representing one sensor (e.g., 'discharge fan', 'return temperature').
Do not include generic qualifiers or measurement terms like 'threshold', 'duration', or 'degree'.


Example 1:
Input: "chiller is off and condenser water pump is running"
Output: ["chiller", "condenser water pump"]


Example 2:
Input: "supply air temperature exceeds setpoint and heating coil valve opens"
Output: ["supply air temperature", "setpoint", "heating coil valve"]


Now extract from this:
Input: """ + rule_text + """
Output:
"""


# Setup proxy call
deployment_id = "AOAIsharednonprodgpt35turbo"
api_version = "2024-02-15-preview"


url = (
   f"https://api-test.cbre.com:443/"
   f"t/digitaltech_us_edp/cbreopenaiendpoint/1/"
   f"openai/deployments/{deployment_id}/chat/completions"
   f"?api-version={api_version}"
)
headers = {
   "Content-Type": "application/json",
   "Authorization": f"Bearer {get_access_token()}"
}
payload = {
   "messages": [{"role": "user", "content": prompt}],
   "max_tokens": 200,
   "temperature": 0
}


# Call the API
resp = requests.post(url, headers=headers, json=payload)
resp.raise_for_status()
content = resp.json()["choices"][0]["message"]["content"].strip()


# Parse LLM output as JSON list
try:
   phrases = json.loads(content)
except json.JSONDecodeError:
   # Fallback: wrap content in brackets if missing
   phrases = json.loads("[" + content + "]")


print("Extracted phrases:", phrases)




