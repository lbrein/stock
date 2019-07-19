# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

 ----- kline预测 ，仅用于1分钟线模拟
       使用校对后的

"""

from com.base.stat_fun import per, fisher
from com.base.public import public, logger
import pandas as pd
import talib as ta
import numpy as np
from com.object.obj_entity import train_future, train_total, future_baseInfo
import os
from com.train.train_future_klineExpect import model_future_simTick_pair
from com.data.interface_Rice import interface_Rice, tick_csv_Rice

import time
import statsmodels.api as sm  # 协整


# 回归方法
class model_future_train_rice(model_future_simTick_pair):
    """

    """

    def __init__(self):
        super().__init__()
        self.klineTypeList = ['1m', '5m', '15m']
        self.method = 'quick'
        self.stage = 'pair'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method + "_" + self.stage

    cindex = 0
    def start(self, codes, time0, kt, shareDict):
        print("子进程启动:", self.cindex, codes, time.time() - time0)
        self.Base = future_baseInfo()
        self.klineType = kt
        self.codes = codes

        # 主力合约
        self.mCodes = mCodes = [n + '88' for n in codes]

        # 查询获得配置 - 费率和每手单量
        for doc in self.Base.getInfo(codes):
            self.baseInfo[doc["code"] + '88'] = doc

        cs0, cs1 = self.baseInfo[mCodes[0]], self.baseInfo[mCodes[1]]
        if cs0["nightEnd"] != cs1["nightEnd"]: return None

        shareDict.append(codes)

        # 子进程共享类
        self.Rice = interface_Rice()
        self.Train = train_future()
        self.Train.tablename = self.detail_tablename
        self.Total = train_total()
        self.Total.tablename = self.total_tablename

        # 查询获得分钟K线
        dfs = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate)

        print('kline load:', mCodes, len(dfs[mCodes[0]]), len(dfs[mCodes[1]]))

        # 计算tick 导致的滑点
        sh = [self.baseInfo[d + '88']['tick_size'] for d in codes]

        # 根据配置文件获取最佳交易手数对

        self.iniVolume = round(self.iniAmount / cs0["lastPrice"] / cs0["contract_multiplier"], 0)
        self.shift = [sh[i] * self.shiftScale for i in range(len(sh))]
        # 分参数执行
        results = []
        for period in self.periodList:
            for scale in self.scaleList:
                for conds in self.iterCond():
                    self.uid = self.uidKey % (
                        "_".join(self.codes), str(period), str(scale), self.klineType.replace('m', ''),
                       conds)
                    self.stopTimeDiff = self.stopTimeLine * period * int(self.klineType[:-1])
                    doc = self.total(dfs[mCodes[0]], dfs[mCodes[1]], scale=scale, period=period)
                    if doc is None and doc['amount'] == 0:  continue
                    logger.info(( doc['uid'], doc['count'], doc['income'], " time:", time.time() - time0))
                    results.append(doc)

        self.Total.insertAll(results)


    preNode, batchId = None, {}

    def total(self, df0, df1=None, scale=1, period=60):
        uid = self.uid
        df0["rel_price"] = close = df0["close"] / df1["close"]
        df0["datetime"] = df0.index

        s0, s1 = self.shift[0], self.shift[1]

        p_l = df0["p_l"] = (df0["close"] + s0) / (df1["close"] - s1)
        p_h = df0["p_h"] = (df0["close"] - s0) / (df1["close"] + s1)

        num = len(df0)

        ma = ta.MA(close[0:num], timeperiod=period)
        df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)
        # 上下柜
        top, lower = ma + scale * std, ma - scale * std
        # bullWidth

        df0["bullwidth"] = width = (4 * std / ma * 100).fillna(0)
        # 近三分钟width变动

        df0["widthDelta"] = wd1 = ta.MA(width - width.shift(1), timeperiod=3).fillna(0)
        df0["delta"] = (p_l - p_h) / std

        wd2 = wd1 - wd1.shift(1)
        wd2m = wd2 * wd2.shift(1)

        # 其他参数计算
        min, max = ta.MINMAX(close, timeperiod=period)
        df0["atr"] = ta.WMA((max.dropna() - min.dropna()), timeperiod=period / 2)
        # 协整
        result = sm.tsa.stattools.coint(df0["close"], df1["close"])

        df0.fillna(0, inplace=True)

        isOpen, preDate, prePrice = 0, None, 0

        doc, doc1, docs = {}, {}, []

        sline, wline = self.stopTimeDiff, self.widthDeltaLine
        for i in range(period, num):
            isRun, isstop = False, 0
            #  开仓2
            if isOpen == 0:
                cond1, cond2 = True, False
                if wline > 0:
                    # 布林宽带变化率
                    cond1 = not ((wd1[i] > wline) or (wd2[i] > (wline / 2)))
                    # 最大值
                    cond2 = wd2m[i] < 0

                # 突变状态开始
                # 大于上线轨迹
                if p_h[i] >= top[i] and not cond1:
                    isOpen = -1
                    isRun = True

                elif p_l[i] <= lower[i] and not cond1:
                    isOpen = 1
                    isRun = True

                elif p_h[i] >= top[i] and cond1 and cond2:
                    isOpen = -2
                    isRun = True

                elif p_l[i] <= lower[i] and cond1 and cond2:
                    isOpen = 2
                    isRun = True

            # 平仓
            else:
                # 回归ma则平仓  或  超过24分钟 或到收盘时间 强制平仓
                if (isOpen * ((p_h[i] if isOpen == 1 else p_l[i]) - ma[i])) >= 0:
                    isOpen = 0
                    isRun = True

                # 止损
                elif sline > 0 and self.preNode and len(self.preNode) == 2:
                    timediff = public.timeDiff(df0['datetime'].values, self.preNode[0]['createdate']) / 60
                    if timediff > sline:
                        isOpen, isstop = 0, 1
                        isRun = True

            if isRun:
                doc, doc1 = self.order(df0.iloc[i], df1.iloc[i], isOpen, uid, isstop=isstop)
                if doc1 is not None:
                    docs.append(doc)
                    docs.append(doc1)


        res = pd.DataFrame(docs).fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
        if self.saveDetail:
            self.Train.insertAll(res.to_dict(orient='records'))

        if len(res) > 0:
            return {
                "scale": scale,
                "code": self.codes[0],
                "code1": self.codes[1],
                "period": period,
                "count": int(len(docs) / 4),
                "amount": (doc["price"] * doc["vol"] + doc1["price"] * doc1["vol"]) if doc is not None else 0,
                "price": doc["price"] / doc1["price"],
                "income": res["income"].sum(),
                "uid": uid,
                "relative": per(df0["close"], df1["close"]),
                "std": std.mean(),
                "shift": (p_l - p_h).mean(),
                "delta": (p_l - p_h).mean() / std.mean(),
                "coint": 0 if np.isnan(result[1]) else result[1],
                "createdate": public.getDatetime()
            }
        else:
            return None



def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = model_future_train_rice()
        obj.Pool()


if __name__ == '__main__':
    main()
