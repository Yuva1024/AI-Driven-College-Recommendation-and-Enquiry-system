import os
import google.generativeai as genai
from dotenv import load_dotenv

print("Testing Gemini API directly...")
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
print(f"API Key present: {bool(API_KEY)}")
print(f"API Key (first 20 chars): {API_KEY[:20] if API_KEY else 'None'}...")

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        print("✓ genai.configure() succeeded")
        
        # Test with GenerativeModel
        print("\nTesting GenerativeModel('gemini-2.5-flash-lite')...")
        try:
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            print("✓ Model created")
            
            response = model.generate_content("What is the placement rate of Veltech University Avadi?")
            print(f"✓ Got response: {response.text[:100]}...")
        except Exception as e:
            print(f"✗ GenerativeModel failed: {e}")
        
        # Test with generate_text
        print("\nTesting generate_text()...")
        try:
            response = genai.generate_text(prompt="What is the placement rate of Veltech University Avadi?")
            print(f"Response type: {type(response)}")
            print(f"Response attrs: {dir(response)[:10]}")
            if hasattr(response, 'result'):
                print(f"✓ Got result: {response.result[:100]}...")
            elif hasattr(response, 'text'):
                print(f"✓ Got text: {response.text[:100]}...")
            else:
                print(f"✓ Got response: {str(response)[:100]}...")
        except Exception as e:
            print(f"✗ generate_text failed: {e}")
            
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
else:
    print("✗ No API key found")
