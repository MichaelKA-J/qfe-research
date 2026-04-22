#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:58:34 2026

@author: michael
"""

import lseg.data as ld
import numpy as np

# data acquisition
ld.open_session()
df = ld.get_history(
    universe='LCOc1',
    fields=['SETTLE'], 
    start='2020-03-08 00:00:00',
    end='2026-03-08 00:00:00',
    interval='daily'
)
ld.close_session()

# solving a bug where I didn't know what field to put
'''
- pull prices without specifying field
- prices_df.columns 
- one of those column names should work
'''

# price data cleaning
'''
- inspect data before coding
- should check for nulls, blanks, negatives, duplicates, etc. 
'''
df.to_csv('oil_prices.csv') # inspect raw data before cleaning

# convert raw prices to log returns
'''
- did not add .values because I wanted to preserve the timestamp
- shift(1) method is much easier than manually indexing
- ln(x/y) = ln(x) - ln(y)
'''
returns = np.log(df.SETTLE / df.SETTLE.shift(1)).dropna()
returns = returns.rename("Return") 

# returns data cleaning
returns = returns * 100 # GARCH prefers % over decimal
returns = returns[returns != 0] # "boolean indexing" to drop zeros
returns = returns[-1250:]
returns.to_csv('oil_returns.csv')
