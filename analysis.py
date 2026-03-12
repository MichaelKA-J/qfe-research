import pandas as pd
import lseg.data as ld
from textblob import TextBlob

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

pos_dummy = (sentiment_index > 0).astype(int)
neg_dummy = (sentiment_index < 0).astype(int)

sentiment_dummies_df = pd.DataFrame({
    'mean_sentiment': sentiment_index,
    'pos_dummy': pos_dummy,
    'neg_dummy': neg_dummy
})
print(sentiment_dummies_df.head())
