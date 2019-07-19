# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

        -- 用一分钟k线模拟 Tick的 双Kline 回测模型
        --- 用于大于1分钟的k线回测模拟

"""

from com.base.stat_fun import per, fisher
from com.base.public import public, logger
import pandas as pd
import talib as ta
import numpy as np
from com.object.obj_entity import  train_future, train_total, future_baseInfo
import os
from com.data.interface_Rice import interface_Rice, tick_csv_Rice
from com.model.train_future_2_tick_new import model_future_quickTickTest
import itertools
import time
import uuid
from multiprocessing import Pool, Manager
import statsmodels.api as sm  #协整
import seaborn as sns

# 回归方法
class model_future_train_rice(object):
    """

    """
    iniAmount = 500000  # 单边50万
    def __init__(self):
        # 费率和滑点
        #self.iniAmount = 10
        self.baseInfo = {}
        self.saveDetail = True # 是否保存明细

        #self.lists=[('B','M'), ('J','I'), ('RU', 'BU'),  ("MA", "ZC"), ("OI", "P"), ("SR", "I")]
        #self.lists = [("SR", "I")]

        self.periodList = [30, 45, 60]
        self.scaleList = [2.0]
        self.shiftScaleList = [0.528]
        self.klineTypeList = ['5m', '15m', '30m', '60m']
        self.uidKey = "%s_%s_%s_0.8_%s_isnew_%s"

        self.startDate = public.getDate(-200)
        #self.endDate = '2018-06-29'
        #self.startDate = '2018-06-06'
        self.endDate = public.getDate()

        self.total_tablename = 'train_total_2'

    def Pool(self):
        time0 = time.time()

        pool = Pool(processes=4)
        shareDict = Manager().list([])

        # 从现有ticks文件获取列表
        Rice = tick_csv_Rice()
        Rice.setParam(model_future_quickTickTest().cvsParam)
        ticks = Rice.exitTicks() # 已存在的ticks

        for cs in ticks:
            rs = cs.split("_")
            for kt in self.klineTypeList:
                #self.start(rs, time0,  kt , shareDict)
                try:
                    pool.apply_async(self.start, (rs, time0,  kt , shareDict))
                    pass
                except Exception as e:
                    print(e)
                    continue
        pool.close()
        pool.join()

    # 时间调整 和 比例因子调整
    def dateAdjust(self, codes,  dfs, sh):
        self.Rice.setParam(model_future_quickTickTest().cvsParam)
        ticks = self.Rice.getTicks(codes)

        # 计算折算率
        ticks = ticks[ticks['isdeal'] == 1]
        ticks["p_l"] = ticks["a1"] / ticks["n_b1"]
        ticks["p_h"] = ticks["b1"] / ticks["n_a1"]

        r0 = (ticks["p_l"] - ticks["close"]).mean()
        r1 = (ticks["close"] - ticks["p_h"]).mean()
        s, e = ticks["datetime"].values[0], ticks["datetime"].values[-1]

        for key in dfs:
            dfs[key] = dfs[key][(dfs[key].index>=s)&(dfs[key].index<=e)]
            dfs[key].reset_index(drop=True)

        dd0, dd1 = dfs[self.mCodes[0]]["close"], dfs[self.mCodes[1]]["close"]

        k = self.shiftScaleList[0]

        d0 = ((dd0 + k * sh[0])/(dd1 - k * sh[1]) - dd0/dd1).mean()
        d1 = (dd0/dd1 - (dd0 - k * sh[0])/(dd1 + k * sh[1])).mean()

        self.shiftScaleList =[round((r0/d0 + r1/d1)/ 2 * 0.95 *  k, 3) ]
        return dfs

    cindex = 0
    def start(self, codes, time0, kt, shareDict):
        print("子进程启动:", self.cindex,  codes, time.time() - time0)
        self.klineType = kt
        # 主力合约
        self.codes = codes
        self.mCodes = mCodes = [n +'88' for n in codes]

        # 查询获得配置 - 费率和每手单量
        self.Base = future_baseInfo()
        for doc in self.Base.getInfo(codes):
            self.baseInfo[doc["code"] + '88'] = doc

        cs0, cs1 = self.baseInfo[mCodes[0]], self.baseInfo[mCodes[1]]
        if cs0["nightEnd"]!= cs1["nightEnd"]: return None

        # 计算tick 导致的滑点
        sh =[self.baseInfo[d+'88']['tick_size'] for d in codes]

        # 子进程共享类
        self.Rice = tick_csv_Rice()
        self.Train = train_future()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename

        # 查询获得N分钟K线
        dfs_l = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate)
        # 获得1分钟K线作为滚动线
        dfs = self.Rice.kline(mCodes, period='1m', start=self.startDate, end=self.endDate)

        # 按时间截取并调整
        dfs= self.dateAdjust(codes, dfs, sh)
        print('kline load:', mCodes, len(dfs[mCodes[0]]), len(dfs[mCodes[1]]))

        # 根据配置文件获取最佳交易手数对
        self.iniVolume = round(self.iniAmount / cs0["lastPrice"]/ cs0["contract_multiplier"], 0)

        # 分参数执行
        results = []
        for period in self.periodList:
            for scale in self.scaleList:
               for ss in self.shiftScaleList:

                    self.shiftScale = ss
                    self.shift = [sh[i] * ss for i in range(len(sh))]

                    doc = self.total(dfs, dfs_l, scale=scale, period=period)
                    if doc is None or doc['amount']==0:  continue
                    logger.info((doc['uid'], doc['income'],  " time:", time.time()-time0))
                    results.append(doc)
                    # 结果写入共享阈
                #shareDict.append(doc)
        self.Total.insertAll(results)

    #  混合K线
    def k_ma(self, d, p, close, period, m):
        df = close[close.index < d][-period-1:]
        df = df.append(pd.Series([p],index=[d]))
        if m ==0:
           res = ta.MA(df, timeperiod=period)
        else:
            res = ta.STDDEV(df, timeperiod=period, nbdev=1)

        return res.values[-1]

    preNode, batchId = None, {}
    def total(self, dfs, dfs2=None, scale=1, period=60):
        uid = self.uidKey % ("_".join(self.codes), str(period), str(scale), self.klineType[:-1], str(self.shiftScale))
        df0, df1 = dfs[self.mCodes[0]], dfs[self.mCodes[1]]
        df0["rel_price"] = close = df0["close"] / df1["close"]
        df0["datetime"] = df0.index
        # 模拟滑点
        num = len(df0)

        s0, s1 = self.shift[0] , self.shift[1]
        p_l =  df0["p_l"] = (df0["close"] + s0) / (df1["close"]-s1)
        p_h = df0["p_h"] = (df0["close"] - s0) / (df1["close"]+s1)

        close2 = dfs2[self.mCodes[0]]["close"] / dfs2[self.mCodes[1]]["close"]
        #ma = ta.MA(close, timeperiod=period)
        #df0["rel_std"] = sd = ta.STDDEV(close, timeperiod=period, nbdev=1)

        df0['ma'] = ma = df0.apply(lambda row: self.k_ma(row['datetime'],row['rel_price'], close2, period,0) , axis=1)
        df0['rel_std'] = sd = df0.apply(lambda row: self.k_ma(row['datetime'],row['rel_price'], close2, period,1), axis=1)
        # 上下柜

        top, lower = ma + scale * sd, ma - scale * sd
        # bullWidth
        df0["bullwidth"] = width = (4 * sd / ma * 100).fillna(0)
        # 近三分钟width变动

        df0["widthDelta"] = ta.MA(width - width.shift(1), timeperiod = 3).fillna(0)
        df0["delta"] = (p_l - p_h) / sd

        # 其他参数计算
        min, max = ta.MINMAX(close, timeperiod=period)
        df0["atr"] = ta.WMA((max.dropna() - min.dropna()), timeperiod = period/2)

        # 协整
        result = sm.tsa.stattools.coint(df0["close"], df1["close"])

        df0.fillna(0, inplace=True)

        isOpen, preDate, prePrice = 0, None, 0
        doc, doc1, docs = {}, {}, []
        for i in range(num):

            isRun = False
            if i < period and np.isnan(ma[i]): continue
            # 开仓
            if isOpen == 0:
                # 大于上线轨迹
                if p_h[i] >= top[i]:
                    isOpen = -1
                    isRun = True

                elif p_l[i] <= lower[i]:
                    isOpen = 1
                    isRun = True

            # 平仓
            else:
               # 回归ma则平仓  或  超过24分钟 或到收盘时间 强制平仓
               if  (isOpen * ((p_h[i] if isOpen == 1 else p_l[i]) - ma[i])) >= 0:
                     isOpen = 0
                     isRun = True

               # 止损

            if isRun:
                doc,doc1 = self.order(df0.iloc[i], df1.iloc[i], isOpen, uid)
                if doc1 is not None:
                     docs.append(doc)
                     docs.append(doc1)

        res = pd.DataFrame(docs)
        #res.fillna(0,inplace=True)
        if len(res) > 0:
            if self.saveDetail:
                print(self.Train.tablename, len(res), 'save')
                self.Train.insertAll(docs)

            return {
                "scale": scale,
                "code": self.codes[0],
                "code1":self.codes[1],
                "period": period,
                "count": int(len(docs)/4),
                "amount": (doc["price"] * doc["vol"] + doc1["price"] * doc1["vol"]) if doc is not None else 0 ,
                "price": doc["price"]/doc1["price"],
                "income": res["income"].sum(),
                "uid": uid,
                "relative": per(df0["close"], df1["close"]),
                "std":  res['rel_std'].mean(),
                "shift": (p_l - p_h).mean(),
                "delta": res['delta'].mean(),
                "coint": 0 if np.isnan(result[1]) else result[1],
                "createdate": public.getDatetime()
            }

        else:
            return None

    batchId=None
    def order(self, n0, n1, mode, uid):
        # baseInfo 配置文件，查询ratio 和 每手吨数

        b0, b1 = self.baseInfo[self.mCodes[0]], self.baseInfo[self.mCodes[1]]
        if mode != 0 :
             self.batchId = uuid.uuid1()

        # 交易量
        v0 = self.iniVolume * b0["contract_multiplier"]
        v1 = int( v0 * n0["close"]/n1["close"] / b1["contract_multiplier"]) * b1["contract_multiplier"]

        # 费率
        fee0 = (self.iniVolume *  b0["ratio"])  if b0["ratio"]> 0.5 else  ((b0["ratio"]) *  n0["close"] * v0)
        fee1 = (v1 / b1["contract_multiplier"] * b1["ratio"]) if b1["ratio"] > 0.5 else (b1["ratio"] * n1["close"] * v1)

        doc = {
            "createdate": n0["datetime"],
            "code": self.codes[0],
            "price": n0["close"],
            "vol": self.preNode[0]["vol"] if self.preNode else v0 ,
            "mode":  mode if not self.preNode else -self.preNode[0]["mode"],
            "isopen": 0 if mode == 0 else 1,
            "fee": fee0,
            "income": 0,
            "rel_price": n0["rel_price"],
            "rel_std": n0["rel_std"],
            "bullwidth": n0["bullwidth"],
            "widthDelta": n0["widthDelta"],
            "delta": n0["delta"],
            "batchid": self.batchId,
            "atr": n0["atr"],
            'p_l': n0["p_l"],
            'p_h': n0["p_h"],
            "uid": uid
        }

        doc1 ={}
        doc1.update(doc)
        doc1.update({
            "code": self.codes[1],
            "price": n1["close"],
            "vol": self.preNode[1]["vol"] if self.preNode else v1,
            "mode": -mode if not self.preNode else -self.preNode[1]["mode"],
            "isopen": 0 if mode == 0 else 1,
            "fee": fee1,
           })

        if mode == 0 and self.preNode:
            p0, p1 = self.preNode[0], self.preNode[1]
            doc["income"] = p0["mode"] * (n0["close"] - p0["price"]  - 2 * p0["mode"] * self.shift[0]) * p0["vol"] - p0["fee"]
            doc1["income"] = p1["mode"] * (n1["close"] - p1["price"] - 2 * p1["mode"] * self.shift[1]) * p1["vol"] - p1["fee"]
            self.preNode = None

        else:
            doc["income"] = -doc["fee"]
            doc1["income"] = -doc1["fee"]
            self.preNode = (doc, doc1)

        return doc, doc1


def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = model_future_train_rice()
        obj.Pool()

if __name__ == '__main__':
    main()
