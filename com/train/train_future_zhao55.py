# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ---  与tick 配套对比计算的 kline
    > 1m的 K线 模拟tick预测
    > 定期执行，用于筛选交易对
"""

from com.base.stat_fun import MaxDrawdown
from com.base.public import public, logger
import pandas as pd
import talib as ta
import numpy as np
from com.object.obj_entity import train_future, train_total, future_baseInfo
from com.data.interface_Rice import interface_Rice, tick_csv_Rice
import time
import uuid
from multiprocessing import Pool, Manager
from com.train.train_future_zhao import  train_future_zhao

# 回归方法
class train_future_zhao55(train_future_zhao):
    """

    """

    csvList = [
        "15_2.0_30_3_2.0_1.5",
        "15_2.0_15_3_2.0_2.5",
        "30_2.0_60_3_2.0_1.5",
    ]

    def __init__(self):
        # 费率和滑点
        super().__init__()

        self.totalMethod = 'single'
        self.stage = 'zhao55'
        self.method = 'ma55'
        self.isAll = 1
        self.isEmptyUse = False
        self.uidKey = "%s_%s_%s_%s_%s_" + self.stage

    # 核心策略部分
    def stageApply(self, df0, df1, period=40, uid=''):
        self.records = []
        """
            布林带策略：            
        """
        status, isOpen = 0, 0
        adjusts = [c[1] for c in self.adjustDates]

        for i in range(period, len(df0)):
            isRun, isstop = False, 0

            close, date, ma10, ma20, ma55, atrb, tu_s, td_s, tu_d, td_d, tu_s1, td_s1, isout3,isout5,interval = (df0.ix[i, key] for key in
                                                                     "close,datetime,ma10,ma20,ma55,atrb,tu_s,td_s,tu_d,td_d,tu_s1,td_s1,isout3,isout5,interval".split(
                                                                         ","))

            if isOpen == 0:
                """
                    开盘策略 
                    1、20分钟线策略
                    2、55分钟策略
                """
                dd = str(public.getTime(style='%H:%M:%S'))

                opt1 = (close > ma55 and (isout3 > 2 or isout5 >2)) or (close < ma55 and (isout3 < -2 or isout5 < -2))

                if opt1 and (abs(isout3) or abs(isout5)) > 2:
                    isOpen, isRun = 5 * np.sign(ma10-ma55), True

            elif isOpen!=0:

                """
                    交叉点: sarm结束点 
                    1、结束突变状态，变为布林带处理
                    2、根据节点状况开平新仓，状态为6 
                """
                # 平仓
                if self.preNode is not None:
                    # 掉仓
                    # 调仓
                    if date in adjusts:
                        self.adjustOrder(df0.iloc[i], date)

                        # 平仓并反向开仓
                    opt1 = (ma10 > ma20 and close > tu_s and isOpen < 0) or (
                                ma10 < ma20 and close < td_s and isOpen > 0)
                    opt2 = (close < td_s1 and isOpen > 0) or (close > tu_s1 and isOpen < 0)

                    if (opt1 and interval > 12) or opt2:
                            # 平仓并反向开仓
                         if self.order(df0.iloc[i], None, 0, uid, df0, isstop=2):
                              isOpen, isRun = np.sign(ma10 - ma20) * 3, True
                    else:
                        Pd, s = self.preNode[0]['price'], self.preNode[0]['createdate']
                        mp = self.getMax(df0, s, date, isOpen)
                        mp = close if np.isnan(mp) else mp

                        # 3倍 atr 平仓
                        if not np.isnan(mp) and (np.sign(isOpen) * (mp - close) > 3.0 * atrb):
                            isOpen, isRun, isstop = 0, True, 3


            if isRun:
                self.order(df0.iloc[i], None, isOpen, uid, df0, isstop=isstop)

        return self.records

def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = train_future_zhao()
        obj.isEmptyUse = True
        obj.Pool()

        obj = train_future_zhao55()
        obj.isEmptyUse = False
        obj.Pool()

if __name__ == '__main__':
    main()
