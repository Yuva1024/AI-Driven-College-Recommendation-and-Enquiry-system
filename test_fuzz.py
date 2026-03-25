import pandas as pd
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

df = pd.read_csv("c:\\Users\\Yuvaraj\\Downloads\\Projects\\TNEA COLLEGE\\TNEA COLLEGE\\complete_engineering_colleges_dataset.csv")
names = df['college_name'].dropna().tolist()

query = "What is the annual fee for Sri Krishna College of Engineering?"

match1 = rf_process.extractOne(query, names, scorer=rf_fuzz.partial_ratio)
match2 = rf_process.extractOne(query.lower(), names, scorer=rf_fuzz.partial_ratio)

lower_names = df['college_name'].str.lower().dropna().tolist()
match3 = rf_process.extractOne(query.lower(), lower_names, scorer=rf_fuzz.partial_ratio)

print("Match 1 (Query mixed, names mixed):", match1)
print("Match 2 (Query lower, names mixed):", match2)
print("Match 3 (Query lower, names lower):", match3)
