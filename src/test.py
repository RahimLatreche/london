import requests
import json
from datetime import datetime, timedelta
import os



# function generating the token
def get_access_token():
    # Note: Client ID, Secrets and auth 2.0 endpoint, can be hardcoded here or can be setup in the environment variable
    client_id = "BruW9KHSkOPiybJ0shJRzSK2QPEa"  # Enter Client ID
    client_secret = "eYjPDmyuJbgKjCzaGIX5d7nlyhwa"  # Enter Client Secrets
    auth_token_endpoint = "https://api-test.cbre.com:443/token"  # Enter auth 2.0 URL to generate Bearer token

    response = requests.post(
        auth_token_endpoint,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )

    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception("Failed to obtain access token")
    

print("-------------------------------------")    
print("Access Token:", get_access_token())  # This will print the access token to the console for verification
print("-------------------------------------")    



def connect_wso2(access_token):
    # Chat Completions URL:
    # https://api-test.cbre.com:443/t/digitaltech_us_edp/cbreopenaiendpoint/1/openai/deployments/{deployment_id}/chat/completions

    # Embeddings URL:
    # https://api-test.cbre.com:443/t/digitaltech_us_edp/cbreopenaiendpoint/1/openai/deployments/{deployment_id}/embeddings

    proxy_url = "https://api-test.cbre.com:443/t/digitaltech_us_edp/cbreopenaiendpoint/1/openai/deployments/{deployment_id}/chat/completions"
    #provide url

    # request Body: for embedding models:
    # request_body = {
    #     "model": "text-embedding-ada-002",
    #     "input": "Hello, This is the test call to connect to Embedding model."
    # }

    deployment_id = "AOAIsharednonprodgpt35turbo16k" 
    #change deployment id as needed
    api_version = "2024-02-15-preview"
    request_body = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant and provide short precise answer. Convert the sentence in French: Hello, How are you?",
            },
            {"role": "user", "content": "test"},
        ],
        "temperature": 0.7,
        "max_tokens": 250,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    url_variable = proxy_url.format(deployment_id=deployment_id)
    url_with_param = f"{url_variable}?api-version={api_version}"

    # Using Static Bearer Token:
    # headers = {'Content-Type':'application/json', 'Authorization': bbe273f4-####-####-####-############'} # Add generated key value here

    # Using Dynamic Bearer Token:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    response = requests.post(
        url_with_param, headers=headers, data=json.dumps(request_body)
    )
    if response.status_code == 200:
        print("Response Status Code:", response.status_code)
        print("Response Content:", response.json())

    return response.status_code == 200


def main():
    # If Access_token is set in the environemnt variable, it will get the value of it otherwise it will generate the new token
    if "ACCESS_TOKEN" in os.environ:
        access_token = os.getenv("ACCESS_TOKEN")
    else:
        access_token = get_access_token()
    print(access_token)

    # This code will try to connect to WSO2 using the access_token, and if access_token is not valid, it will generate the new token and try again once.
    if not connect_wso2(access_token):
        access_token_new = get_access_token()
        connect_wso2(access_token_new)

    # Setting the access_token value in the environment variable for next run(if run within one hour, it will be valid)
    # os.environ["ACCESS_TOKEN"] = access_token


if __name__ == "__main__":
    main()