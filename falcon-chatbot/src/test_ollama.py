import requests

def query_ollama(prompt: str, model: str = "gpt-oss:20b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    return response.json()["response"]

if __name__ == "__main__":
    answer = query_ollama("Hello! Can you explain what Ollama is in simple terms?")
    print("Model:", answer)
