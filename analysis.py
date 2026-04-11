# import lseg.data as ld
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
pd.set_option('display.max_columns', None)

# set wd so I don't have to put the whole file path everytime
import os
os.chdir('/Users/michael/Documents/ECON 525/Research')



# data acquisition
'''
ld.open_session()
df = ld.get_history(
    universe='LCOc1',
    fields=['SETTLE'], 
    start='2020-03-08 00:00:00',
    end='2026-03-08 00:00:00',
    interval='daily'
)
ld.close_session()
'''
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
df.to_csv('oil_prices.csv') # inspect raw data before cleaning
'''
# convert raw prices to log returns
'''
- did not add .values because I wanted to preserve the timestamp
- shift(1) method is much easier than manually indexing
- ln(x/y) = ln(x) - ln(y)
'''
'''
returns = np.log(df.SETTLE / df.SETTLE.shift(1)).dropna()
returns = returns.rename("Return") 
'''
'''
# returns data cleaning
returns = returns * 100 # GARCH prefers % over decimal
returns = returns[returns != 0] # "boolean indexing" to drop zeros
returns = returns[-1250:]
returns.to_csv('oil_returns.csv')
'''


returns = pd.read_csv(
    "https://github.com/MichaelKA-J/qfe-research/"
    "blob/main/oil_returns.csv?raw=true",
    )

# plot returns
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
returns['Date'] = pd.to_datetime(returns['Date']) # 
def plot_returns():
    plt.ylabel('Daily % Return')
    plt.title(
        'Fig. 2: LCOc1 Daily % Returns, '
        'Last 1250 Trading Days Since 2026-03-06',
        )
    plt.gcf().autofmt_xdate() 
    plt.autoscale(tight = 'x') 
    plt.plot(returns.Date, returns.Return)
plot_returns()
# plt.savefig('oil_returns_plot.png')



# summary stats
'''
print(returns.Return.mean())
print(returns.Return.std())
print(returns.Return.skew())
print(returns.Return.kurt() + 3)
print(returns.Return.count())
'''



# baseline garch(1,1) estimation
am = arch_model(returns.Return.values)
res = am.fit(disp=False) # disp=False is nice for surpressing output
print(res.summary())



# run GJR GARCH
am_asym = arch_model(
    returns,
    mean='Constant',
    vol='GARCH',
    p=1,
    o=1,   
    q=1,
    dist='t'   
)
res_asym = am_asym.fit(disp=False)
print(res_asym.summary())



# backtesting setup 
window = 1000
horizon = 1
n_forecasts = 250

garch_forecasts = []
gjr_forecasts = []

r = returns.Return.values

# rolling window backtest 
for i in range(n_forecasts):
    train = r[i:i+window]
    
    # GARCH(1,1)
    am = arch_model(train, mean='Constant', vol='GARCH', p=1, q=1)
    res = am.fit(disp=False)
    fcast = res.forecast(horizon=horizon)
    garch_forecasts.append(fcast.variance.values[-1, 0])
    
    # GJR-GARCH(1,1) 
    am_asym = arch_model(train, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='t')
    res_asym = am_asym.fit(disp=False)
    fcast_asym = res_asym.forecast(horizon=horizon)
    gjr_forecasts.append(fcast_asym.variance.values[-1, 0])

# convert to arrays
garch_forecasts = np.array(garch_forecasts)
gjr_forecasts = np.array(gjr_forecasts)

# realized variance proxy (squared returns)
realized = r[window:window + n_forecasts]**2

# forecast errors 
garch_errors = realized - garch_forecasts
gjr_errors = realized - gjr_forecasts

# Mean Forecast Error (MFE) 
garch_mfe = np.mean(garch_errors)
gjr_mfe = np.mean(gjr_errors)

# Root Mean Squared Forecast Error (RMSFE)
garch_rmsfe = np.sqrt(np.mean(garch_errors**2))
gjr_rmsfe = np.sqrt(np.mean(gjr_errors**2))



# R^2 OOS 
# benchmark = historical mean of realized variance (rolling)
benchmark = []
for i in range(n_forecasts):
    benchmark.append(np.mean(r[i:i+window]**2))
benchmark = np.array(benchmark)

# OOS R^2 formula
garch_r2_oos = 1 - np.sum((realized - garch_forecasts)**2) / np.sum((realized - benchmark)**2)
gjr_r2_oos   = 1 - np.sum((realized - gjr_forecasts)**2) / np.sum((realized - benchmark)**2)

print("GARCH MFE:", garch_mfe)
print("GJR   MFE:", gjr_mfe)

print("\nGARCH RMSFE:", garch_rmsfe)
print("GJR   RMSFE:", gjr_rmsfe)

print("\nGARCH R2_OOS:", garch_r2_oos)
print("GJR   R2_OOS:", gjr_r2_oos)



# Clark-West test: does GJR-GARCH improve on GARCH? 
# Null: baseline GARCH forecast is at least as good as GJR-GARCH
# Alt : GJR-GARCH has lower MSPE than baseline GARCH

from scipy import stats

e_null = realized - garch_forecasts
e_alt  = realized - gjr_forecasts

f_cw = (e_null**2) - (e_alt**2) + (garch_forecasts - gjr_forecasts)**2

cw_mean = np.mean(f_cw)
cw_se = np.std(f_cw, ddof=1) / np.sqrt(len(f_cw))
cw_tstat = cw_mean / cw_se

# one-sided p-value
cw_pvalue = 1 - stats.t.cdf(cw_tstat, df=len(f_cw)-1)

print("\n Clark-West Test ")
print("t-stat:", cw_tstat)
print("one-sided p-value:", cw_pvalue)



'''
# APARCH backtest + CW test

aparch_forecasts = []

for i in range(n_forecasts):
    train = r[i:i+window]
    
    # APARCH(1,1)
    am_aparch = arch_model(
        train,
        mean='Constant',
        vol='APARCH',
        p=1,
        o=1,
        q=1,
        dist='t'
    )
    res_aparch = am_aparch.fit(disp=False)
    fcast_aparch = res_aparch.forecast(horizon=horizon)
    aparch_forecasts.append(fcast_aparch.variance.values[-1, 0])

# convert to array
aparch_forecasts = np.array(aparch_forecasts)

# forecast errors
aparch_errors = realized - aparch_forecasts

# Mean Forecast Error 
aparch_mfe = np.mean(aparch_errors)

# Root Mean Squared Forecast Error 
aparch_rmsfe = np.sqrt(np.mean(aparch_errors**2))

# R^2 OOS
aparch_r2_oos = 1 - np.sum((realized - aparch_forecasts)**2) / np.sum((realized - benchmark)**2)

print("\nAPARCH MFE:", aparch_mfe)
print("APARCH RMSFE:", aparch_rmsfe)
print("APARCH R2_OOS:", aparch_r2_oos)

# Clark-West test: does APARCH improve on GARCH?
# Null: baseline GARCH forecast is at least as good as APARCH
# Alt : APARCH has lower MSPE than baseline GARCH

e_null_aparch = realized - garch_forecasts
e_alt_aparch  = realized - aparch_forecasts

f_cw_aparch = (e_null_aparch**2) - (e_alt_aparch**2) + (garch_forecasts - aparch_forecasts)**2

cw_mean_aparch = np.mean(f_cw_aparch)
cw_se_aparch = np.std(f_cw_aparch, ddof=1) / np.sqrt(len(f_cw_aparch))
cw_tstat_aparch = cw_mean_aparch / cw_se_aparch

# one-sided p-value
cw_pvalue_aparch = 1 - stats.t.cdf(cw_tstat_aparch, df=len(f_cw_aparch)-1)

print("\nClark-West Test: GARCH vs APARCH")
print("t-stat:", cw_tstat_aparch)
print("one-sided p-value:", cw_pvalue_aparch)



# EGARCH backtest + DM test

egarch_forecasts = []

for i in range(n_forecasts):
    train = r[i:i+window]
    
    # EGARCH(1,1)
    am_egarch = arch_model(
        train,
        mean='Constant',
        vol='EGARCH',
        p=1,
        o=1,
        q=1,
        dist='t'
    )
    res_egarch = am_egarch.fit(disp=False)
    fcast_egarch = res_egarch.forecast(horizon=horizon)
    egarch_forecasts.append(fcast_egarch.variance.values[-1, 0])

# convert to array
egarch_forecasts = np.array(egarch_forecasts)

# forecast errors
egarch_errors = realized - egarch_forecasts

# Mean Forecast Error 
egarch_mfe = np.mean(egarch_errors)

# Root Mean Squared Forecast Error 
egarch_rmsfe = np.sqrt(np.mean(egarch_errors**2))

# R^2 OOS
egarch_r2_oos = 1 - np.sum((realized - egarch_forecasts)**2) / np.sum((realized - benchmark)**2)

print("\nEGARCH MFE:", egarch_mfe)
print("EGARCH RMSFE:", egarch_rmsfe)
print("EGARCH R2_OOS:", egarch_r2_oos)

# Diebold-Mariano test
# Null: GARCH and EGARCH have equal predictive accuracy
# Alt : EGARCH has lower expected loss than GARCH

from scipy import stats

# squared-error loss differential
loss_garch = (realized - garch_forecasts)**2
loss_egarch = (realized - egarch_forecasts)**2
d = loss_garch - loss_egarch

dm_mean = np.mean(d)
dm_se = np.std(d, ddof=1) / np.sqrt(len(d))
dm_stat = dm_mean / dm_se

# one-sided p-value for alt: EGARCH better than GARCH
dm_pvalue = 1 - stats.t.cdf(dm_stat, df=len(d)-1)

print("\nDiebold-Mariano Test: GARCH vs EGARCH")
print("DM statistic:", dm_stat)
print("one-sided p-value:", dm_pvalue)
'''
