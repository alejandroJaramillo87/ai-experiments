import requests
import json

# 1. Update the URL to point to the CPU server's port
API_URL = "http://127.0.0.1:8001/v1/chat/completions"

# 2. Use a simple header, as the non-streaming default is application/json
headers = {
    "Content-Type": "application/json"
}

# 3. Update the payload for a simple, non-streaming test
payload = {
    "model": "qwen2.5-coder-32b-instruct-q5_k_m.gguf",
    "messages": [
        {
            "role": "user",
            "content": "Write a python function that calculates the factorial of a number."
        }
    ],
    "max_tokens": 150,
    "temperature": 0.2,
    "stream": False # Explicitly set to False for a simple test
}

def test_cpu_inference():
    """Sends a request to the CPU server and prints the response."""
    print("--- 🧪 Sending request to CPU Llama.cpp Server ---")
    print(f"URL: {API_URL}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        # Add a generous timeout for the CPU-bound task
        response = requests.post(API_URL, headers=headers, json=payload, timeout=600)
        response.raise_for_status()

        print("\n--- Server Response ---")
        print(json.dumps(response.json(), indent=2))
        print("\n✅ SUCCESS: The CPU server responded correctly!")

    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: The request timed out. The model may be too slow to respond.")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: Could not connect to the server.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_cpu_inference()