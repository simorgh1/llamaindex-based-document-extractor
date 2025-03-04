import google.generativeai as genai
import openai
from deepseek import DeepSeekAPI
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="List models for the given provider name (openai/gemini).")
parser.add_argument("--provider", type=str, help="Name of the provider to show available models for")
args = parser.parse_args()

def list_deepseek_models():
    api_client = DeepSeekAPI(os.environ["DEEPSEEK_API_KEY"])
    models = api_client.get_models()
    for model in models:
        print(model)

def list_openai_models():
    openai.api_key = os.environ["OPENAI_API_KEY"]

    models = openai.models.list()
    sorted_models = sorted(models, key=lambda x: x.id)
    for model in sorted_models:
        print(model.id)

def list_gemini_models():
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    models = genai.list_models()
    sorted_models = sorted(models, key=lambda x: x.name)

    for m in sorted_models:
        if "generateContent" in m.supported_generation_methods:
            print(m.name)

if(args.provider == "openai"):
    list_openai_models()
elif(args.provider == "gemini"):
    list_gemini_models()
elif(args.provider == "deepseek"):
    list_deepseek_models()
else:
    print("Please provide a valid provider name (openai/gemini).")
