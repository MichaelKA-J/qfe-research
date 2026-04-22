#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:30:39 2026

@author: michael
"""

from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)



# download already cleaned returns
returns = pd.read_csv(
    "https://github.com/MichaelKA-J/qfe-research/"
    "blob/main/oil_returns.csv?raw=true",
    )



# summary stats
print("Summary Stats:")
print(returns.Return.mean())
print(returns.Return.std())
print(returns.Return.skew())
print(returns.Return.kurt() + 3)
print(returns.Return.count())
print("\n")



# plot returns
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
returns['Date'] = pd.to_datetime(returns['Date']) # 
def plot_returns():
    plt.grid()
    plt.ylabel('Daily % Return')
    plt.title(
        'LCOc1 Daily % Returns, '
        'Last 1250 Trading Days Since 2026-03-06',
        )
    plt.gcf().autofmt_xdate() 
    plt.autoscale(tight = 'x') 
    plt.plot(returns.Date, returns.Return)
plot_returns()



# baseline garch(1,1) estimation
print("Baseline GARCH(1,1) Estimation")
am = arch_model(returns.Return.values)
res = am.fit(disp=False) # disp=False is nice for surpressing output
print(res.summary())
print("\n")



# run GJR GARCH
print("GJR-GARCH(1,1) Estimation")
am_asym = arch_model(
    returns.Return.values,
    mean='Constant',
    vol='GARCH',
    p=1,
    o=1,   
    q=1,
    dist='t'   
)
res_asym = am_asym.fit(disp=False)
print(res_asym.summary())
print("\n")



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


