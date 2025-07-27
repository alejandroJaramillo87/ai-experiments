import requests
import json

# The URL of your local TensorRT-LLM server.
# This works because your docker-compose.yml maps port 8000 of the container
# to port 8000 on your host machine (127.0.0.1).
API_URL = "http://127.0.0.1:8000/v1/chat/completions"

# The header specifying the content type is JSON.
headers = {
    "Content-Type": "application/json"
}

# The payload, mimicking a request from Open WebUI.
# We will test both streaming and non-streaming.
payload = {
    "model": "mistral-7b-v0.3-int4-awq", # This should match the model you're serving
    "messages": [
        {
            "role": "user",
            "content": "Explain GPUs to me as if I were a child."
        }
    ],
    "max_tokens": 150,
    "temperature": 0.7,
    "stream": False # Set to False for a simple, non-streaming test
}

def test_inference():
    """Sends a request to the server and prints the response."""
    print("--- Sending request to TRT-LLM Server ---")
    print(f"URL: {API_URL}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        # Check if the request was successful
        response.raise_for_status()

        print("\n--- Server Response ---")
        # Pretty-print the JSON response from the server
        print(json.dumps(response.json(), indent=2))
        print("\nSUCCESS: The server responded correctly!")

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Could not connect to the server.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    test_inference()
