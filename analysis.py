#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:08:22 2026

@author: michael
"""

import lseg.data as ld
from textblob import TextBlob
# from arch import arch_model

### Sentiment Index Contstruction

ld.open_session()

df = ld.news.get_headlines(
    query='OPEC AND Language:LEN', 
    start='2025-03-08', 
    end='2026-03-08',
    count=100 # 20000 to be safe
    )

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity
df['sentiment_score'] = df['headline'].apply(get_sentiment)

sentiment_index = df.sentiment_score.resample('D').mean()

print(sentiment_index.head())



### GARCH Estimation
