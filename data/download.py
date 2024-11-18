#%%
import json
import requests
import pandas as pd
from datasets import load_dataset

#%%
# Download the go_emotions dataset
go_emotions = load_dataset("go_emotions")
data = go_emotions.data

df_train = data["train"].to_pandas().drop(columns=["id"])
df_val = data["validation"].to_pandas().drop(columns=["id"])
df_test = data["test"].to_pandas().drop(columns=["id"])

df_train.to_parquet("train.parquet", index=False)
df_val.to_parquet("val.parquet", index=False)
df_test.to_parquet("test.parquet", index=False)

#%%
# Download emoji sentiment data
df_emotag = pd.read_csv('https://raw.githubusercontent.com/abushoeb/EmoTag/refs/heads/master/data/EmoTag1200-scores.csv')
df_emotag.to_parquet("emotag.parquet", index=False)

# %%
url ="https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/british_spellings.json"
uk2us = requests.get(url).json() 
with open('spellings/british_spellings.json', 'w') as f:
    json.dump(uk2us, f)
url ="https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json"
us2uk = requests.get(url).json() 
with open('spellings/american_spellings.json', 'w') as f:
    json.dump(us2uk, f)