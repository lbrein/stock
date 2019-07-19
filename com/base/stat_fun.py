# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ----- 统计公式
"""

from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import talib as ta

# 皮尔森相识度计算
def per(df0, df1):
    df0, df1 = regular(df0), regular(df1)
    l = min(len(df0), len(df1))
    r = pearsonr(df0[0:l], df1[0:l])[0]
    return 0 if np.isnan(r) else r

# 最大最小归一
def regular(df0):
    return (df0 -df0.min()) / (df0.max() - df0.min())

# fisher变换
def fisher(high, low, timeperiod=10, fish_ratio=0.33):
    mid = (high + low) / 2
    max, min = ta.MAX(mid, timeperiod=timeperiod), ta.MIN(mid, timeperiod=timeperiod)

    vv = 2 * ((mid - min) / (max - min) - 0.5)
    vv = vv.apply(lambda x: 0.999 if x > 0.99 else x)
    vv = vv.apply(lambda x: -0.999 if x < -0.99 else x)
    f0 = preKeep(vv, fish_ratio)
    vv2 = ta.LN((1 + f0) / (1 - f0))

    # fish 值
    fish = preKeep(vv2, 0.5)
    return fish

# 继承算法
def preKeep(vv, diff):
    pre = None
    f0 = []
    for n in vv:
        if not np.isnan(n):
            if pre is None:
                v = n
            else:
                v = diff * n + (1 - diff) * pre
            f0.append(v)
            pre = v
        else:
            f0.append(np.nan)
    return pd.Series(f0, index=vv.index)

# 返回除最大公约数后值
def subMax(a,b):
    p = 1.0
    a0, b0 = max([a,b]), min([a,b])
    while p!=0:
       p = a0 % b0
       a0 = b0
       b0 = p
    return (a/a0, b/a0)


def MaxDrawdown(return_list):
    '''最大回撤率'''
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    return (return_list[j] - return_list[i]) / (return_list[j])

if __name__ == '__main__':
   a = subMax(121, 66)
   print(a)
