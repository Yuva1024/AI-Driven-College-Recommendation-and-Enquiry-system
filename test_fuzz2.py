import pandas as pd
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

df = pd.read_csv("c:\\Users\\Yuvaraj\\Downloads\\Projects\\TNEA COLLEGE\\TNEA COLLEGE\\complete_engineering_colleges_dataset.csv")
names = df['college_name'].dropna().tolist()

query = "What is the annual fee for Sri Krishna College of Engineering?"

match = rf_process.extractOne(query, names, scorer=rf_fuzz.token_set_ratio)

print("token_set_ratio:", match)
