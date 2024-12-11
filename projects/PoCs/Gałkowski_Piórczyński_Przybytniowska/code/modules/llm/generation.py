import requests

LLM_API_URL = "http://localhost:8000/v1/chat/completions"


def generate_response(
    prompt: str, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
) -> str:
    response = requests.post(
        LLM_API_URL,
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    response.raise_for_status()
    return response.json()
