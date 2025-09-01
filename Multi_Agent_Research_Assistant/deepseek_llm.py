import ollama
import re

def call_deepseek(prompt):
    if isinstance(prompt, str):
        messages = [{'role': 'user', 'content': prompt}]
    elif isinstance(prompt, list):
        messages = [{'role': m.get('role', 'user'), 'content': m.get('content', '')} for m in prompt]
    else:
        raise ValueError("Prompt must be str or list of dicts.")

    response = ollama.chat(model='deepseek-r1:latest', messages=messages)
    return preprocess_response(response['message']['content'])

def preprocess_response(response: str) -> str:
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
