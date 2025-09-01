from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import re
import traceback

app = FastAPI()

# Define request body model
class RequestModel(BaseModel):
    prompts: list[str]  # Accept multiple prompts

# Function to clean AI responses
def clean_response(text):
    """Remove <think> tags and unnecessary characters"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # Remove <think>...</think> blocks
    text = text.replace("\\n", "\n").strip()  # Fix escaped newlines and trim spaces
    return text

# API Endpoint to handle multiple prompts
@app.post("/generate/")
def generate_response(request: RequestModel):
    results = {}

    for prompt in request.prompts:
        response = None  #  Initialize response before try block

        try:
            response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}])

            # Extract message and content
            message = response.get("message", {})
            raw_content = message.get("content", "")

            if not raw_content:
                raise ValueError(f"Unexpected response format: {response}")

            cleaned_content = clean_response(raw_content)
            results[prompt] = {
                "content": cleaned_content
            }

        except Exception as e:
            error_trace = traceback.format_exc()
            raise HTTPException(status_code=500, detail={
                "error": str(e),
                "message": response["message"] if response else "No response received",  
                "content": clean_response(response["message"]["content"]) if response and "message" in response else "No content"
            })

    return {"responses": results}

# Root endpoint to check if the API is running
@app.get("/")
def root():
    return {"message": "FastAPI Ollama Assistant is running!"}