import numpy as np
import pandas as pd
from finvader import finvader

df = pd.read_csv(
    '/Users/michael/Documents/ECON 525/crude_oil_headlines_raw.csv',
    index_col='versionCreated',
    parse_dates=True
)

df = df.tail(5000)

def get_sentiment(text):
    return finvader(
        text,
        indicator='compound',
        use_sentibignomics=True,
        use_henry=True
    )

sentiment_score = df['headline'].apply(get_sentiment)
sentiment_index = sentiment_score.resample('D').mean().dropna()
sentiment_index = sentiment_index[-250:]

"""
pos_dummy = (sentiment_index > 0).astype(int)
neg_dummy = (sentiment_index < 0).astype(int)

sentiment_dummies_df = pd.DataFrame({
    'mean_sentiment': sentiment_index,
    'pos_dummy': pos_dummy,
    'neg_dummy': neg_dummy
})

print(sentiment_dummies_df.head())
"""

# the news var. has a constant negative bias, so we centered it
overall_mean = sentiment_index.mean()
above_avg_dummy = (sentiment_index > overall_mean).astype(int)
below_avg_dummy = (sentiment_index < overall_mean).astype(int)

sentiment_dummies_df = pd.DataFrame({
    'mean_sentiment': sentiment_index,
    'above_avg_dummy': above_avg_dummy,
    'below_avg_dummy': below_avg_dummy
    })
print(sentiment_dummies_df.head())

print(np.mean(sentiment_dummies_df.above_avg_dummy))
print(np.mean(sentiment_dummies_df.below_avg_dummy))
