import pandas as pd
from rapidfuzz import process as rf_process, fuzz as rf_fuzz, utils

df = pd.read_csv("c:\\Users\\Yuvaraj\\Downloads\\Projects\\TNEA COLLEGE\\TNEA COLLEGE\\complete_engineering_colleges_dataset.csv")
names = df['college_name'].dropna().tolist()

query = "What is the annual fee for Sri Krishna College of Engineering?"

matches = rf_process.extract(query.lower(), [n.lower() for n in names], scorer=rf_fuzz.partial_ratio, limit=5)
print("partial_ratio limit 5:")
for m in matches:
    print(m)

def custom_search(q):
    best_college = None
    max_score = 0
    q_clean = utils.default_process(q)
    for n in names:
        n_clean = utils.default_process(n)
        # remove " - city"
        if " - " in n_clean:
            n_clean = n_clean.split(" - ")[0]
        
        score = rf_fuzz.partial_ratio(q_clean, n_clean)
        # We need to make sure the match length is significant, to avoid matching just "college"
        if score > max_score and score > 80:
            best_college = n
            max_score = score
    return best_college, max_score

print("custom logic:", custom_search(query))
