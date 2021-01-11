# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:30:30 2021

@author: AXZQ
"""

import numpy as np
import pandas as pd
import utils
import bottleneck as bn

def TS_ArgMax(s, backlength = 5):
    backlength = int(backlength)
    def argmax(x):
        if len(x) < backlength:
            return pd.Series([np.nan] * len(x), index = x.index)
        return pd.Series(bn.move_argmax(x, backlength), index = x.index)
    return s.groupby('code').apply(argmax).sort_index()

def TS_ArgMin(s, backlength = 5):
    backlength = int(backlength)
    def argmin(x):
        if len(x) < backlength:
            return pd.Series([np.nan] * len(x), index = x.index)
        return pd.Series(bn.move_argmin(x, backlength), index = x.index)
    return s.groupby('code').apply(argmin).sort_index()

def rolling_window(a, window):
    if len(a) < window:
        return pd.Series([np.nan] * len(a) , index = a.index)
    index = a.index
    a = a.values
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    result = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return pd.Series([np.nan] * (window - 1) + list(result), index = index)
    
def numpy_rank(x):
    index = np.argsort(x)
    rank = np.argsort(index) + 1
    return rank[-1]

def TS_Rank(s, backlength = 5):
    backlength = int(backlength)
    def rolling_rank(x):
        if len(x) < backlength:
            return pd.Series([np.nan] * len(x), index = x.index)
        return pd.Series(bn.move_rank(x, window = backlength), index = x.index)
    return (s.groupby('code').apply(rolling_rank).sort_index() + 1) * (backlength - 1) / 2 + 1
    
def TS_Min(s, backlength = 5):
    backlength = int(backlength)
    return s.groupby('code').rolling(backlength).min().droplevel(0).sort_index()

def TS_Max(s, backlength = 5):
    backlength = int(backlength)
    return s.groupby('code').rolling(backlength).max().droplevel(0).sort_index()

def stddev(s, backlength = 5):
    backlength = int(backlength)
    return s.groupby('code').rolling(backlength).std().droplevel(0).sort_index()

def SignedPower(s, power = 2):
    return np.power(s, power).sort_index()

def rank(s):
    return s.groupby('date').rank().sort_index()

def delta(s, backlength = 2):
    backlength = int(backlength)
    return s.groupby('code').diff(backlength).sort_index()

def correlation(s1, s2, backlength = 6):
    backlength = int(backlength)
    s = pd.concat([s1, s2], axis = 1)
    s.columns = columns = ['s1', 's2']
    return s.reset_index().set_index('date').groupby('code')[columns].rolling(backlength).corr().droplevel(2).iloc[::2, 1].sort_index()

def covariance(s1, s2, backlength = 6):
    backlength = int(backlength)
    s = pd.concat([s1, s2], axis = 1)
    columns = s.columns
    return s.reset_index().set_index('date').groupby('code')[columns].rolling(backlength).cov().droplevel(2).iloc[::2, 1].sort_index()
    
def delay(s, backlength = 5):
    backlength = int(backlength)
    return s.groupby('code').shift(backlength).sort_index()

def sum_(s, backlength = 10):
    backlength = int(backlength)
    return s.groupby('code').rolling(backlength).sum().droplevel(0).sort_index()

def min_(s, backlength = 10):
    backlength = int(backlength)
    return s.groupby('code').rolling(backlength).min().droplevel(0).sort_index()

def scale(s, center = 1):
    s = s.replace([np.inf, -np.inf], [np.nan, np.nan])
    return (s / np.abs(s).groupby('date').transform('sum') * center).sort_index()

def product(s, backlength = 5):
    backlength = int(backlength)
    return s.groupby('code').rolling(backlength).apply(np.prod).droplevel(0).sort_index()

def decay_linear(s, backlength = 5):
    backlength = int(backlength)
    expanding_s = s.groupby('code').apply(rolling_window, window = backlength)
    index = expanding_s.index
    weights = np.arange(1, backlength + 1)
    expanding_s = expanding_s * pd.Series([weights] * len(expanding_s), index = index)
    return pd.Series(np.stack(expanding_s.values).sum(axis = 1), index = index) / np.sum(weights)
    
def IndNeutralize(s, industry_data):
    result = utils.industry_market_value_neutral(s.rename('f').replace([np.inf, -np.inf], [np.nan, np.nan]), industry_data, industry = True, market_value = False).sort_index()
    if len(result.index.names) == 3:
        result = result.droplevel(level = 0).sort_index()
    return result.reindex(s.index)

def alpha1(data):
    return rank(TS_ArgMax(SignedPower(stddev(data['returns'], 20).mask(data['returns'] >= 0, data['adj_close']), 2), 5)) - 0.5

def alpha2(data):
    return -1 * correlation(rank(delta(np.log(data['volume']), 2)), rank(data['adj_close'] / data['adj_open']), 6)

def alpha3(data):
    return -1 * correlation(rank(data['adj_open']), rank(data['volume']), 10)

def alpha4(data):
    return -1 * TS_Rank(rank(data['adj_low']), 9)

def alpha5(data):
    return rank(data['adj_open'] - sum_(data['adj_vwap'], 10) / 10) * -1 * np.abs(rank(data['adj_close'] - data['adj_vwap']))

def alpha6(data):
    return -1 * correlation(data['adj_open'], data['volume'], 10)

def alpha7(data):
    return (-1 * TS_Rank(np.abs(delta(data['adj_close'], 7)), 60)).mask(sum_(data['volume'], 20) / 20 < data['volume'] * np.sign(delta(data['adj_close'], 7)), -1)

def alpha8(data):
    return -1 * rank(sum_(data['adj_open'], 5) * sum_(data['returns'], 5) - delay(sum_(data['adj_open'], 5) * sum_(data['returns'], 5)))

def alpha9(data):
    return delta(data['adj_close'], 1).mask(TS_Max(delta(data['adj_close'], 1), 5) >= 0, (-delta(data['adj_close'], 1))).mask(TS_Min(delta(data['adj_close'], 1), 5) < 0, delta(data['adj_close'], 1))

def alpha10(data):
    return rank(delta(data['adj_close'], 1).mask(TS_Max(delta(data['adj_close'], 1), 4) >= 0, (-delta(data['adj_close'], 1))).mask(TS_Min(delta(data['adj_close'], 1), 4) < 0, delta(data['adj_close'], 1)))

def alpha11(data):
    return rank(TS_Max(data['adj_vwap'] - data['adj_close'], 3)) + rank(TS_Min(data['adj_vwap'] - data['adj_close'], 3)) * rank(delta(data['volume'], 3))

def alpha12(data):
    return np.sign(delta(data['volume'], 1)) * -1 * delta(data['adj_close'], 1)

def alpha13(data):
    return -1 * rank(covariance(rank(data['adj_close']), rank(data['volume']), 5))

def alpha14(data):
    return -1 * rank(delta(data['returns'], 3)) * correlation(rank(data['adj_close']), rank(data['volume']), 5)

def alpha15(data):
    return -1 * sum_(rank(correlation(rank(data['adj_high']), rank(data['volume']), 3)), 3)

def alpha16(data):
    return -1 * rank(covariance(rank(data['adj_high']), rank(data['volume']), 5))

def alpha17(data):
    return -1 * rank(TS_Rank(data['adj_close'], 10)) * rank(delta(delta(data['adj_close'], 1), 1)) * rank(TS_Rank(data['volume'] / sum_(data['volume'], 20) * 20, 5))

def alpha18(data):
    return -1 * rank(stddev(np.abs(data['adj_close'] - data['adj_open']), 5) + (data['adj_close'] - data['adj_open']) + correlation(data['adj_close'], data['adj_open'], 10))

def alpha19(data):
    return (-1 * np.sign(((data['adj_close'] - delay(data['adj_close'], 7)) + delta(data['adj_close'], 7)))) * (1 + rank((1 + sum_(data['returns'], 250))))

def alpha20(data):
    return (((-1 * rank((data['adj_open'] - delay(data['adj_high'], 1)))) * rank((data['adj_open'] - delay(data['adj_close'], 1)))) * rank((data['adj_open'] - delay(data['adj_low'], 1))))

def alpha21(data):
    condition1 = sum_(data['adj_close'], 8) / 8 + stddev(data['adj_close'], 8) < sum_(data['adj_close'], 2) / 2
    condition2 = sum_(data['adj_close'], 8) / 8 - stddev(data['adj_close'], 8) > sum_(data['adj_close'], 2) / 2
    condition3 = data['volume'] >= sum_(data['volume'], 20) / 20
    return condition3.replace([True, False], [1, -1]).mask(condition2, 1).mask(condition1, -1)

def alpha22(data):
    return (-1 * (delta(correlation(data['adj_high'], data['volume'], 5), 5) * rank(stddev(data['adj_close'], 20))))

def alpha23(data):
    return (-1 * delta(data['adj_high'], 2)).mask(sum_(data['adj_high'], 20) / 20 >= data['adj_high'], 0)

def alpha24(data):
    return (-1 * delta(data['adj_close'], 3)).mask(delta((sum_(data['adj_close'], 100) / 100), 100) / delay(data['adj_close'], 100) <= 0.05, -1 * (data['adj_close'] - TS_Min(data['adj_close'])))

def alpha25(data):
    return rank(((((-1 * data['returns']) * sum_(data['volume'], 20) / 20) * data['adj_vwap']) * (data['adj_high'] - data['adj_low'])))

def alpha26(data):
    return (-1 * TS_Max(correlation(TS_Rank(data['volume'], 5), TS_Rank(data['adj_high'], 5), 5), 3))

def alpha27(data):
    return pd.Series(np.where(0.5 < rank((sum_(correlation(rank(data['volume']), rank(data['adj_vwap']), 6), 2) / 2)), -1, 1), index = data.index)

def alpha28(data):
    return scale(((correlation(sum_(data['volume'], 20) / 20, data['adj_low'], 5) + ((data['adj_high'] + data['adj_low']) / 2)) - data['adj_close']))

def alpha29(data):
    return (min_(product(rank(rank(scale(np.log(sum_(TS_Min(rank(rank((-1 * rank(delta((data['adj_close'] - 1), 5))))), 2), 1))))), 1), 5) + TS_Rank(delay((-1 * data['returns']), 6), 5))

def alpha30(data):
    return (((1 - rank(((np.sign((data['adj_close'] - delay(data['adj_close'], 1))) + np.sign((delay(data['adj_close'], 1) - delay(data['adj_close'], 2)))) + np.sign((delay(data['adj_close'], 2) - delay(data['adj_close'], 3)))))) * sum_(data['volume'], 5)) / sum_(data['volume'], 20))

def alpha31(data):
    return ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(data['adj_close'], 10)))), 10)))) + rank((-1 * delta(data['adj_close'], 3)))) + np.sign(scale(correlation(sum_(data['volume'], 20) / 20, data['adj_low'], 12))))
def alpha32(data):
    return (scale(((sum_(data['adj_close'], 7) / 7) - data['adj_close'])) + (20 * scale(correlation(data['adj_vwap'], delay(data['adj_close'], 5), 230))))

def alpha33(data):
    return rank((-1 * ((1 - (data['adj_open'] / data['adj_close'])))))

def alpha34(data):
    return rank(((1 - rank((stddev(data['returns'], 2) / stddev(data['returns'], 5)))) + (1 - rank(delta(data['adj_close'], 1)))))

def alpha35(data):
    return ((TS_Rank(data['volume'], 32) * (1 - TS_Rank(((data['adj_close'] + data['adj_high']) - data['adj_low']), 16))) * (1 - TS_Rank(data['returns'], 32)))

def alpha36(data):
    return (((((2.21 * rank(correlation((data['adj_close'] - data['adj_open']), delay(data['volume'], 1), 15))) + (0.7 * rank((data['adj_open'] - data['adj_close'])))) + 
        (0.73 * rank(TS_Rank(delay((-1 * data['returns']), 6), 5)))) + rank(np.abs(correlation(data['adj_vwap'], sum_(data['volume'], 20) / 20, 6)))) + 
    (0.6 * rank((((sum_(data['adj_close'], 200) / 200) - data['adj_open']) * (data['adj_close'] - data['adj_open'])))))

def alpha37(data):
    return (rank(correlation(delay((data['adj_open'] - data['adj_close']), 1), data['adj_close'], 200)) + rank((data['adj_open'] - data['adj_close'])))

def alpha38(data):
    return ((-1 * rank(TS_Rank(data['adj_close'], 10))) * rank((data['adj_close'] / data['adj_open'])))

def alpha39(data):
    return ((-1 * rank((delta(data['adj_close'], 7) * (1 - rank(decay_linear((data['volume'] / sum_(data['volume'], 20) * 20), 9)))))) * (1 + rank(sum_(data['returns'], 250))))

def alpha40(data):
    return ((-1 * rank(stddev(data['adj_high'], 10))) * correlation(data['adj_high'], data['volume'], 10))

def alpha41(data):
    return (((data['adj_high'] - data['adj_low']) ** 0.5) - data['adj_vwap'])

def alpha42(data):
    return (rank((data['adj_vwap'] - data['adj_close'])) / rank((data['adj_vwap'] + data['adj_close'])))

def alpha43(data):
    return (TS_Rank((data['volume'] / sum_(data['volume'], 20) * 20), 20) * TS_Rank((-1 * delta(data['adj_close'], 7)), 8))

def alpha44(data):
    return (-1 * correlation(data['adj_high'], rank(data['volume']), 5))

def alpha45(data):
    return (-1 * ((rank((sum_(delay(data['adj_close'], 5), 20) / 20)) * correlation(data['adj_close'], data['volume'], 2)) * rank(correlation(sum_(data['adj_close'], 5), sum_(data['adj_close'], 20), 2))))

def alpha46(data):
    return ((-1 * 1) * (data['adj_close'] - delay(data['adj_close'], 1))).mask((0.25 < (((delay(data['adj_close'], 20) - delay(data['adj_close'], 10)) / 10) - ((delay(data['adj_close'], 10) - data['adj_close']) / 10))), 1)

def alpha47(data):
    return  ((((rank((1 / data['adj_close'])) * data['volume']) / sum_(data['volume'], 20) * 20) * ((data['adj_high'] * rank((data['adj_high'] - data['adj_close']))) / (sum_(data['adj_high'], 5) / 5))) - rank((data['adj_vwap'] - delay(data['adj_vwap'], 5))))

def alpha48(data):
    return (IndNeutralize(((correlation(delta(data['adj_close'], 1), delta(delay(data['adj_close'], 1), 1), 250) * delta(data['adj_close'], 1)) / data['adj_close']).rename('f'), data[['industry', 'market_value']]) / sum_(((delta(data['adj_close'], 1) / delay(data['adj_close'], 1)) ** 2), 250))

def alpha49(data):
    cond = (delay(data['adj_close'], 20) - delay(data['adj_close'], 10)) / 10 - (delay(data['adj_close'], 10) - data['adj_close']) / 10 < -0.1
    return (data['adj_close'] - delay(data['adj_close'], 1)).mask(cond, -1)

def alpha50(data):
    return (-1 * TS_Max(rank(correlation(rank(data['volume']), rank(data['adj_vwap']), 5)), 5))

def alpha51(data):
    cond = (delay(data['adj_close'], 20) - delay(data['adj_close'], 10)) / 10 - (delay(data['adj_close'], 10) - data['adj_close']) / 10 < -0.05
    return (data['adj_close'] - delay(data['adj_close'], 1)).mask(cond, -1)

def alpha52(data):
    return ((((-1 * TS_Min(data['adj_low'], 5)) + delay(TS_Min(data['adj_low'], 5), 5)) * rank(((sum_(data['returns'], 240) - sum_(data['returns'], 20)) / 220))) * TS_Rank(data['volume'], 5))

def alpha53(data):
    return (-1 * delta((((data['adj_close'] - data['adj_low']) - (data['adj_high'] - data['adj_close'])) / (data['adj_close'] - data['adj_low'])), 9))

def alpha54(data):
    return ((-1 * ((data['adj_low'] - data['adj_close']) * (data['adj_open'] ** 5))) / ((data['adj_low'] - data['adj_high']) * (data['adj_close'] ** 5)))

def alpha55(data):
    return (-1 * correlation(rank(((data['adj_close'] - TS_Min(data['adj_low'], 12)) / (TS_Max(data['adj_high'], 12) - TS_Min(data['adj_low'], 12)))), rank(data['volume']), 6))

def alpha56(data):
    return (0 - (1 * (rank((sum_(data['returns'], 10) / sum_(sum_(data['returns'], 2), 3))) * rank((data['returns'] * data['market_value'])))))

def alpha57(data):
    return (0 - (1 * ((data['adj_close'] - data['adj_vwap']) / decay_linear(rank(TS_ArgMax(data['adj_close'], 30)), 2))))

def alpha58(data):
    return (-1 * TS_Rank(decay_linear(correlation(IndNeutralize(data['adj_vwap'], data[['industry', 'market_value']]), data['volume'], 3.92795), 7.89291), 5.50322))

def alpha59(data):
    return  (-1 * TS_Rank(decay_linear(correlation(IndNeutralize(((data['adj_vwap'] * 0.728317) + (data['adj_vwap'] * (1 - 0.728317))), data[['industry', 'market_value']]), data['volume'], 4.25197), 16.2289), 8.19648))

def alpha60(data):
    return (0 - (1 * ((2 * scale(rank(((((data['adj_close'] - data['adj_low']) - (data['adj_high'] - data['adj_low'])) / (data['adj_high'] - data['adj_low'])) * data['volume'])))) - scale(rank(TS_ArgMax(data['adj_close'], 10))))))

def alpha61(data):
    return (rank((data['adj_vwap'] - TS_Min(data['adj_vwap'], 16))) < rank(correlation(data['adj_vwap'], sum_(data['volume'], 180) / 180, 17)))

def alpha62(data):
    return ((rank(correlation(data['adj_vwap'], sum_(sum_(data['volume'], 20) / 20, 22.4101), 9.91009)) < rank(((rank(data['adj_open']) + rank(data['adj_open'])) < (rank(((data['adj_high'] + data['adj_low']) / 2)) + rank(data['adj_high']))))) * -1)

def alpha63(data):
    return ((rank(decay_linear(delta(IndNeutralize(data['adj_close'], data[['industry', 'market_value']]), 2.25164), 8.22237)) - rank(decay_linear(correlation(((data['adj_vwap'] * 0.318108) + (data['adj_open'] * (1 - 0.318108))), sum_(sum_(data['volume'], 180) / 180, 37.2467), 13.557), 12.2883))) * -1)

def alpha64(data):
    return ((rank(correlation(sum_(((data['adj_open'] * 0.178404) + (data['adj_low'] * (1 - 0.178404))), 12), sum_(sum_(data['volume'], 120) / 120, 12), 16)) < rank(delta(((((data['adj_high'] + data['adj_low']) / 2) * 0.178404) + (data['adj_vwap'] * (1 - 0.178404))), 3))) * -1)

def alpha65(data):
    return ((rank(correlation(((data['adj_open'] * 0.00817205) + (data['adj_vwap'] * (1 - 0.00817205))), sum_(sum_(data['volume'], 60) / 60, 8), 6)) < rank((data['adj_open'] - TS_Min(data['adj_open'], 13)))) * -1)

def alpha66(data):
    return ((rank(decay_linear(delta(data['adj_vwap'], 3), 7)) + TS_Rank(decay_linear(((((data['adj_low'] * 0.96633) + (data['adj_low'] * (1 - 0.96633))) - data['adj_vwap']) / (data['adj_open'] - ((data['adj_high'] + data['adj_low'])/ 2))), 11), 6)) * -1)

def alpha67(data):
    return  ((rank(decay_linear(delta(data['adj_vwap'], 3.51013), 7.23052)) + TS_Rank(decay_linear(((((data['adj_low'] * 0.96633) + (data['adj_low'] * (1 - 0.96633))) - data['adj_vwap']) / (data['adj_open'] - ((data['adj_high'] + data['adj_low']) / 2))), 11.4157), 6.72611)) * -1)

def alpha68(data):
    return ((TS_Rank(correlation(rank(data['adj_high']), rank(sum_(data['volume'], 15) / 15), 8), 13) < rank(delta(((data['adj_close'] * 0.518371) + (data['adj_low'] * (1 - 0.518371))), 1))) * -1)

def alpha69(data):
    return ((rank(TS_Max(delta(IndNeutralize(data['adj_vwap'], data[['industry', 'market_value']]), 2.72412), 4.79344)) ** TS_Rank(correlation(((data['adj_close'] * 0.490655) + (data['adj_vwap'] * (1 - 0.490655))), sum_(data['volume'], 20) / 20, 4.92416), 9.0615)) * -1)

def alpha70(data):
    return ((rank(delta(data['adj_vwap'], 1))))

def alpha71(data):
    s1 = TS_Rank(decay_linear(correlation(TS_Rank(data['adj_close'], 3), TS_Rank(sum_(data['volume'], 180) / 180, 12), 18), 4), 15)
    s2 = TS_Rank(decay_linear((rank(((data['adj_low'] + data['adj_open']) - (data['adj_vwap'] + data['adj_vwap']))) ** 2), 16), 4)
    return s1.mask(s1 < s2, s2)

def alpha72(data):
    return (rank(decay_linear(correlation(((data['adj_high'] + data['adj_low']) / 2), sum_(data['volume'], 40) / 40, 8), 10)) / rank(decay_linear(correlation(TS_Rank(data['adj_vwap'], 3), TS_Rank(data['volume'], 18), 6), 2)))

def alpha73(data):
    s1 = rank(decay_linear(delta(data['adj_vwap'], 4), 2))
    s2 = TS_Rank(decay_linear(((delta(((data['adj_open'] * 0.147155) + (data['adj_low'] * (1 - 0.147155))), 2) / ((data['adj_open'] * 0.147155) + (data['adj_low'] * (1 - 0.147155)))) * -1), 3), 16)
    return -s1.mask(s1 < s2, s2)

def alpha74(data):
    return ((rank(correlation(data['adj_close'], sum_(sum_(data['volume'], 30) / 30, 37), 15)) < rank(correlation(rank(((data['adj_high'] * 0.0261661) + (data['adj_vwap'] * (1 - 0.0261661)))), rank(data['volume']), 11))) * -1)

def alpha75(data):
    return (rank(correlation(data['adj_vwap'], data['volume'], 4)) < rank(correlation(rank(data['adj_low']), rank(sum_(data['volume'], 50) / 50), 12)))

def alpha76(data):
    s1 = rank(decay_linear(delta(data['adj_vwap'], 1), 11))
    s2 = TS_Rank(decay_linear(TS_Rank(correlation(IndNeutralize(data['adj_low'], data[['industry', 'market_value']]), sum_(data['volume'], 81) / 81, 8.14941), 19.569), 17.1543), 19.383)
    return -s1.mask(s1 < s2, s2)

def alpha77(data):
    s1 = rank(decay_linear(((((data['adj_high'] + data['adj_low']) / 2) + data['adj_high']) - (data['adj_vwap'] + data['adj_high'])), 20.0451))
    s2 = rank(decay_linear(correlation(((data['adj_high'] + data['adj_low']) / 2), sum_(data['volume'], 40) / 40, 3.1614), 5.64125))
    return s1.mask(s1 > s2, s2)

def alpha78(data):
    return (rank(correlation(sum_(((data['adj_low'] * 0.352233) + (data['adj_vwap'] * (1 - 0.352233))), 19.7428), sum_(sum_(data['volume'], 40) / 40, 19.7428), 6.83313)) ** rank(correlation(rank(data['adj_vwap']), rank(data['volume']), 5.77492)))

def alpha79(data):
    return (rank(delta(IndNeutralize(((data['adj_close'] * 0.60733) + (data['adj_open'] * (1 - 0.60733))).rename('f'), data[['industry', 'market_value']]), 1.23438)) < rank(correlation(TS_Rank(data['adj_vwap'], 3.60973), TS_Rank(sum_(data['volume'], 150) / 150, 9.18637), 14.6644)))

def alpha80(data):
    return ((rank(np.sign(delta(IndNeutralize(((data['adj_open'] * 0.868128) + (data['adj_high'] * (1 - 0.868128))), data[['industry', 'market_value']]), 4.04545))) ** TS_Rank(correlation(data['adj_high'], sum_(data['volume'], 10) / 10, 5.11456), 5.53756)) * -1)

def alpha81(data):
    return  ((rank(np.log(product(rank((rank(correlation(data['adj_vwap'], sum_(sum_(data['volume'], 10) / 10, 49.6054), 8.47743)) ** 4)), 14.9655))) < rank(correlation(rank(data['adj_vwap']), rank(data['volume']), 5.07914))) * -1)

def alpha82(data):
    s1 = rank(decay_linear(delta(data['adj_open'], 1.46063), 14.8717))
    s2 = TS_Rank(decay_linear(correlation(IndNeutralize(data['volume'], data[['industry', 'market_value']]), ((data['adj_open'] * 0.634196) + (data['adj_open'] * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)
    return -s1.mask(s1 < s2, s2)

def alpha83(data):
    return  ((rank(delay(((data['adj_high'] - data['adj_low']) / (sum_(data['adj_close'], 5) / 5)), 2)) * rank(rank(data['volume']))) / (((data['adj_high'] - data['adj_low']) / (sum_(data['adj_close'], 5) / 5)) / (data['adj_vwap'] - data['adj_close'])))

def alpha84(data):
    return SignedPower(TS_Rank((data['adj_vwap'] - TS_Max(data['adj_vwap'], 15.3217)), 20.7127), delta(data['adj_close'], 4.96796))

def alpha85(data):
    return (rank(correlation(((data['adj_high'] * 0.876703) + (data['adj_close'] * (1 - 0.876703))), sum_(data['volume'], 30) / 30, 9.61331)) ** rank(correlation(TS_Rank(((data['adj_high'] + data['adj_low']) / 2), 3.70596), TS_Rank(data['volume'], 10.1595), 7.11408)))

def alpha86(data):
    return ((TS_Rank(correlation(data['adj_close'], sum_(sum_(data['volume'], 20) / 20, 14.7444), 6.00049), 20.4195) < rank(((data['adj_open'] + data['adj_close']) - (data['adj_vwap'] + data['adj_open'])))) * -1)

def alpha87(data):
    s1 = rank(decay_linear(delta(((data['adj_close'] * 0.369701) + (data['adj_vwap'] * (1 - 0.369701))), 1.91233), 2.65461))
    s2 = TS_Rank(decay_linear(np.abs(correlation(IndNeutralize(sum_(data['volume'], 81) / 81, data[['industry', 'market_value']]), data['adj_close'], 13.4132)), 4.89768), 14.4535)
    return -s1.mask(s1 < s2, s2)

def alpha88(data):
    s1 = rank(decay_linear(((rank(data['adj_open']) + rank(data['adj_low'])) - (rank(data['adj_high']) + rank(data['adj_close']))), 8.06882))
    s2 = TS_Rank(decay_linear(correlation(TS_Rank(data['adj_close'], 8.44728), TS_Rank(sum_(data['volume'], 60) / 60, 20.6966), 8.01266), 6.65053), 2.61957)
    return s1.mask(s1 > s2, s2)

def alpha89(data):
    return (TS_Rank(decay_linear(correlation(((data['adj_low'] * 0.967285) + (data['adj_low'] * (1 - 0.967285))), sum_(data['volume'], 10) / 10, 6.94279), 5.51607), 3.79744) - TS_Rank(decay_linear(delta(IndNeutralize(data['adj_vwap'], data[['industry', 'market_value']]), 3.48158), 10.1466), 15.3012))
                                                                        
def alpha90(data):
    return ((rank((data['adj_close'] - TS_Max(data['adj_close'], 4.66719))) ** TS_Rank(correlation(IndNeutralize(sum_(data['volume'], 40) / 40, data[['industry', 'market_value']]), data['adj_low'], 5.38375), 3.21856)) * -1)

def alpha91(data):
    return ((TS_Rank(decay_linear(decay_linear(correlation(IndNeutralize(data['adj_close'], data[['industry', 'market_value']]), data['volume'], 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(data['adj_vwap'], sum_(data['volume'], 30) / 30, 4.01303), 2.6809))) * -1)

def alpha92(data):
    s1 = TS_Rank(decay_linear(((((data['adj_high'] + data['adj_low']) / 2) + data['adj_close']) < (data['adj_low'] + data['adj_open'])), 14.7221), 18.8683)
    s2 = TS_Rank(decay_linear(correlation(rank(data['adj_low']), rank(sum_(data['volume'], 30) / 30), 7.58555), 6.94024), 6.80584)
    return s1.mask(s1 > s2, s2)

def alpha93(data):
    return (TS_Rank(decay_linear(correlation(IndNeutralize(data['adj_vwap'], data[['industry', 'market_value']]), sum_(data['volume'], 81) / 81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((data['adj_close'] * 0.524434) + (data['adj_vwap'] * (1 - 0.524434))), 2.77377), 16.2664)))

def alpha94(data):
    return ((rank((data['adj_vwap'] - TS_Min(data['adj_vwap'], 11.5783))) ** TS_Rank(correlation(TS_Rank(data['adj_vwap'], 19.6462), TS_Rank(sum_(data['volume'], 60) / 60, 4.02992), 18.0926), 2.70756)) * -1)

def alpha95(data):
    return (rank((data['adj_open'] - TS_Min(data['adj_open'], 12.4105))) < TS_Rank((rank(correlation(sum_(((data['adj_high'] + data['adj_low']) / 2), 19.1351), sum_(sum_(data['volume'], 40) / 40, 19.1351), 12.8742)) ** 5), 11.7584))

def alpha96(data):
    s1 = TS_Rank(decay_linear(correlation(rank(data['adj_vwap']), rank(data['volume']), 3.83878), 4.16783), 8.38151)
    s2 = TS_Rank(decay_linear(TS_ArgMax(correlation(TS_Rank(data['adj_close'], 7.45404), TS_Rank(sum_(data['volume'], 60) / 60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)
    return -s1.mask(s1 < s2, s2)

def alpha97(data):
    return ((rank(decay_linear(delta(IndNeutralize(((data['adj_low'] * 0.721001) + (data['adj_vwap'] * (1 - 0.721001))), data[['industry', 'market_value']]), 3.3705), 20.4523)) - TS_Rank(decay_linear(TS_Rank(correlation(TS_Rank(data['adj_low'], 7.87871), TS_Rank(sum_(data['volume'], 60) / 60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)

def alpha98(data):
    return (rank(decay_linear(correlation(data['adj_vwap'], sum_(sum_(data['volume'], 5) / 5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(TS_Rank(TS_ArgMin(correlation(rank(data['adj_open']), rank(sum_(data['volume'], 15) / 15), 20.8187), 8.62571), 6.95668), 8.07206)))

def alpha99(data):
    return ((rank(correlation(sum_(((data['adj_high'] + data['adj_low']) / 2), 19.8975), sum_(sum_(data['volume'],60) / 60, 19.8975), 8.8136)) < rank(correlation(data['adj_low'], data['volume'], 6.28259))) * -1)

def alpha100(data):
    return (0 - (1 * (((1.5 * scale(IndNeutralize(rank(((((data['adj_close'] - data['adj_low']) - (data['adj_high'] - data['adj_close'])) / (data['adj_high'] - data['adj_low'])) * data['volume'])), data[['industry', 'market_value']]))) - scale(IndNeutralize((correlation(data['adj_close'], rank(sum_(data['volume'], 20) / 20), 5) - rank(TS_ArgMin(data['adj_close'], 30))), data[['industry', 'market_value']]))) * (data['volume'] / sum_(data['volume'], 20) / 20))))

def alpha101(data):
    return ((data['adj_close'] - data['adj_open']) / ((data['adj_high'] - data['adj_low']) + .001))
