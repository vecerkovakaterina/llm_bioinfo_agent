import requests
import json


# mixtral:8x22b
#
def call_ollama_api(params):
    ollama_api_generate = "http://localhost:11434/api/generate"
    request = requests.post(ollama_api_generate, json=params)
    return request


def get_response_answer(ollama_api_generate_response):
    return json.loads(json.loads(ollama_api_generate_response.text)["response"])[
        "answer"
    ]


mixtral_call = get_response_answer(
    call_ollama_api(
        {
            "model": "mixtral:8x22b",
            "prompt": "Why is the sky blue?",
            "format": "json",
            "stream": False,
        }
    )
)

print(mixtral_call)
