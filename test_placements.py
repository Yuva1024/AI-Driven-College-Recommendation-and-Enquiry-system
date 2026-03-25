import requests
import json

# Flask app URL
url = "http://127.0.0.1:5000/"

# Test form data
data = {
    'cut_off': 180.0,
    'previous_year_cutoff': 175.0,
    'rank': 5000,
    'college_fees': 150000,
    'category': 'OC',
    'district': 'Chennai',
    'branch': 'CSE',
    'sports_quota': 'No',
    'top_n': 3
}

print("Submitting prediction request...")
print(f"Data: {data}")
print("-" * 60)

try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response Length: {len(response.text)} bytes")
    
    # Check if placements are in the response
    if "Placement" in response.text:
        print("✓ Placements section found in response")
        # Extract placements sections
        lines = response.text.split('\n')
        for i, line in enumerate(lines):
            if 'Placement' in line or 'placement' in line.lower():
                print(f"  Line {i}: {line.strip()[:100]}")
    else:
        print("✗ No placement data in response")
    
    print("\nChecking for Gemini results...")
    if "Gemini" in response.text or "gemini" in response.text.lower():
        print("✓ Found Gemini references in response")
    
except Exception as e:
    print(f"Error: {e}")

print("\nDone!")
