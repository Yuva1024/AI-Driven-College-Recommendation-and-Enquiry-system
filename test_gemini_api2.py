import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

try:
    genai.configure(api_key=API_KEY)
    
    print("Available genai attributes and methods:")
    attrs = dir(genai)
    # Filter for likely model/API related methods
    for attr in attrs:
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    print("\n\nAvailable models:")
    try:
        models = genai.list_models()
        for model in models:
            print(f"  - {model.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
        
    print("\n\nTrying genai.chat()...")
    try:
        chat = genai.chat()
        print(f"Chat created, type: {type(chat)}")
        response = chat.send_message("What is Veltech University?")
        print(f"Response: {response.text[:100] if hasattr(response, 'text') else str(response)[:100]}")
    except Exception as e:
        print(f"Error: {e}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
