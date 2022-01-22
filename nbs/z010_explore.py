# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python [conda env:q5]
#     language: python
#     name: conda-env-q5-py
# ---

# %%
# load and autoreload
from IPython import get_ipython
# noinspection PyBroadException
try:
    _magic = get_ipython().magic
    _magic('load_ext autoreload')
    _magic('autoreload 2')
except:
    pass
    

import os 
import glob
import sys
import pprint
import datetime as dt
import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt 
import xarray as xr 
import seaborn as sns
import cartopy as crt


# %%

def get_presL():
    F = '../data/CHC_2016-09-09T15_16_00_2016-12-31T23_59_00_15.0_471_minute.MTO'

    df = pd.read_csv(F,sep=';',skiprows=38)

    df['DT'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])

    df1 = df.set_index('DT')
    df1 = df1.rename({'AP':'P','AT':'T'},axis=1)
    
    df1.index = df1.index  - pd.Timedelta(hours=4)
    df1 = df1[['P','T','RH','WS']]
    df2 = df1.resample('15T',label='right').mean()
    df3 = df2.resample('1T',label='right').mean().interpolate(limit=15)
    return df3

def read_summit():
    F = '../data/CHC_cumbre_15M_2016_until_1104.txt'

    df = pd.read_csv(F,sep='\s+',skiprows=24)

    dt = df[['YYYY', 'MM', 'DD', 'HH', 'mm']].rename({'YYYY':'Year','MM':'Month','DD':'Day','HH':'Hour','mm':'Minute'},axis=1)


    df.index=pd.to_datetime(dt)

    df1 = df.rename({'Tair':'T','Pbar':'P'},axis=1)[['T','RH','P','WS']]

    df2 = df1[(df1>-999)&(df1<999)]

    df3 = df2.resample('15T').mean()
    df3 = df3.resample('1T',label='right').mean().interpolate(limit=15)
    return df3 

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

def cross_corr(V, dfm):
    res = {}
    for i in range(-120, 120):
        res[i] = crosscorr(dfm[V + 's'], dfm[V + 'i'], i)
    res = pd.Series(res)
    return res


def plot_ts(ax, dfm,V):
    dfm.plot(ax=ax)
    ax.set_title(V)


def merge_data(V, dfI, dfS):
    dfm = pd.merge(dfS[[V]], dfI[[V]], left_index=True, right_index=True, suffixes=['s', 'i'])
    return dfm


def plot_cross_val(V, ax, res):
    delta_t = res.index[res.argmax()]
    res.plot(ax=ax)
    ax.set_title(f'{V} [cross-correlation]')
    ax.set_xlabel('time delta [minutes]')
    ax.set_ylabel('correlation')
    ax.axvline(delta_t, c='.5')
    ax.text(delta_t, np.mean(res), f' {delta_t=}')

# %%
# %%
dfI = get_presL()

dfS = read_summit()



for V in ['T','RH','P','WS']:

    dfm = merge_data(V, dfI, dfS)
    res = cross_corr(V, dfm)

    f, (ax1,ax2) = plt.subplots(2,1,figsize=(6,6))
    plot_ts(ax1, dfm, V)

    plot_cross_val(V, ax2, res)
    f.savefig(f'./imgs/{V}.png')

# %%

# %%



