# src/auth.py

import requests

def get_access_token() -> str:
    """
    Uses CBRE’s OAuth2 endpoint to fetch a fresh Bearer token.
    """
    client_id = "BruW9KHSkOPiybJ0shJRzSK2QPEa"
    client_secret = "eYjPDmyuJbgKjCzaGIX5d7nlyhwa"
    auth_token_endpoint = "https://api-test.cbre.com:443/token"

    resp = requests.post(
        auth_token_endpoint,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]



