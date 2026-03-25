import pandas as pd
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

df = pd.read_csv("c:\\Users\\Yuvaraj\\Downloads\\Projects\\TNEA COLLEGE\\TNEA COLLEGE\\complete_engineering_colleges_dataset.csv")
names = df['college_name'].dropna().tolist()

query = "What is the annual fee for Sri Krishna College of Engineering?"

match1 = rf_process.extractOne(query, names, scorer=rf_fuzz.partial_ratio)
match2 = rf_process.extractOne(query, names, scorer=rf_fuzz.token_set_ratio)
match3 = rf_process.extractOne(query, names, scorer=rf_fuzz.partial_token_set_ratio)
match_lower = rf_process.extractOne(query.lower(), [n.lower() for n in names], scorer=rf_fuzz.partial_token_set_ratio)

print("partial:", match1)
print("token_set:", match2)
print("partial_token_set:", match3)
print("partial_token_set_lower:", match_lower)
