#%%
import json
import pandas as pd

#%%
go_emotions = {
    'train': pd.read_parquet("data/go_emotions/train.parquet"),
    'val': pd.read_parquet("data/go_emotions/val.parquet"),
    'test': pd.read_parquet("data/go_emotions/test.parquet"),
}

#%%
emotag = pd.read_parquet("data/emotag/emotag.parquet")

#%%
nrc_lexicon = pd.read_csv(
    "data/nrc_lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", 
    sep='\t', 
    header=None,
    names=["word", "emotion", "association"],
)
nrc_emotions = nrc_lexicon[~nrc_lexicon['emotion'].isin(['negative', 'positive'])]
nrc_sentiment = nrc_lexicon[nrc_lexicon['emotion'].isin(['negative', 'positive'])]

# %%
with open('data/spellings/british_spellings.json') as f:
    uk2us = json.load(f)
with open('data/spellings/american_spellings.json') as f:
    us2uk = json.load(f)