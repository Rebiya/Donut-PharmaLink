import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def list_gemini_models(api_key: str):
    client = genai.Client(api_key=api_key)

    models = client.models.list()

    print("Available Gemini models:\n")

    for model in models:
        if "gemini" in model.name.lower():
            print(model.name)

if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise SystemExit("Please set GEMINI_API_KEY in the environment.")

    list_gemini_models(api_key)