#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:29:14 2026
@author: michael
"""
import time
import pandas as pd
import lseg.data as ld
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=FutureWarning)



### Crucial Console Commands
"""
df = pd.DataFrame()
df.head()
df.tail()
df.columns # column names
df.drop_duplicates()
"""



### Pagination In API Calls
"""
Suppose you have to pull a large dataset via an API...
the download speed gets exponentially slower as you try to pull more data...

Solution: 
pagination- pull the data 500 obs. at a time rather than all 50k at once
"""

# The Slow Way
"""
ld.open_session()
df = ld.news.get_headlines(
    query = "Topic:CRU AND Language:LEN",
    start = "2022-01-01",
    end = "2026-01-01",
    count = 50000)
ld.close_session()
"""

# The Fast Way
chunks = [] # lists are computationally faster than dataframes
END = pd.to_datetime('2026-03-08') # turns readable str to datetime datatype
STOP_DATE = pd.to_datetime('2022-03-08') 

ld.get_config()["http.request-timeout"] = 120 # protect from timeout error
start_time = time.time() # start the clock
while True:    
    chunk = ld.news.get_headlines(
        query = 'Topic:CRU AND Language:LEN',
        end = END,
        count = 500
    ) 

    chunks.append(chunk.headline)  
    END = chunk.index.min()
    print(datetime.now().strftime("%H:%M:%S.%f")) # the stopwatch
    
    
    if chunk.empty or END <= STOP_DATE: 
        break

end_time = time.time() # stop the clock
duration = end_time - start_time 

# concat is a clean way to turn list to df
chunks_df = pd.concat(chunks).drop_duplicates() 
print(f"\n{duration} seconds")
print(f"{len(chunks_df)} obs.\n")
print(chunks_df.tail())

chunks_df.to_csv("crude_oil_headlines_raw.csv")

"""
LSEG imposes a hard cap of 15 months worth of headlines
2 options: 
- use 250 days of trading data rather than 1000
- Get help from Roark either to get the API limit removed or to
get the data I need directly rather than through all this programming
"""
