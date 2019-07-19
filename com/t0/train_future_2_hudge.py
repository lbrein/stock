# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

   # 商品跨品种对冲策略
    策略1：异常追涨杀跌，以5分钟k线为基准计算
    策略2：跳空预测


"""

from com.base.public import public, logger
import pandas as pd
import talib as ta
import numpy as np
from com.object.obj_entity import train_future, train_total, future_baseInfo, stock_uniform

from com.data.interface_Rice import interface_Rice

import time
import uuid

from multiprocessing import Pool
import copy
import datetime

from multiprocessing.managers import BaseManager

# 回归方法
class MyManager(BaseManager):
    pass

MyManager.register('interface_Rice', interface_Rice)
MyManager.register('future_baseInfo', future_baseInfo)


def Manager2():
    m = MyManager()
    m.start()
    return m
# 回归方法
class train_future_2_hudge(object):
    """

    """
    cvsList = ['RB']
    csvKey = '%s_14_%s_2_2_3.5_0.5'

    def __init__(self):
        self.saveDetail = True  # 是否保存明细
        self.topUse = False
        self.isEmptyUse = True    # 是否清空记录

        self.testDays = 10
        self.processCount = 5

        self.period = 14
        self.bullwidth = 2

        self.iniAmount = 50000  # 单边50万

        # self.iniAmount = 2000000 * 0.0025  # 单边50万
        self.stopLine = 2
        self.klineList = ['3m', '5m']

        # 起始时间
        self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测

        self.endDate = public.getDate(diff=1)

        self.total_tablename = 'train_total_2'
        self.detail_tablename = 'train_future_2'
        self.uniform_tablename = 'future_uniform'
        self.method = 'hudge2'

        self.uidKey = "%s_%s_%s_" + self.method
        self.isAll = 0

    def iterCond(self):
        # 多重组合参数输出
        for i in range(1):
            yield '%s_%s' % (str(0), str(0))
        #"""
        """    
        keys = ['atrLine', 'stdcTick', 'startTick']
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                self.__setattr__(keys[1], s1)
                for s2 in self.__getattribute__(keys[2] + 'List'):
                    self.__setattr__(keys[2], s2)
                    yield '%s_%s_%s' % (str(s0), str(s1), str(s2))
        """

    def switch(self):
        # 生成all
        if self.isAll == 1:
            self.isEmptyUse = True
            self.total_tablename = 'train_total_0'
            self.detail_tablename = 'train_future_0'
        self.empty()

    def empty(self):
        if self.isEmptyUse:
            Train = train_future()
            Total = train_total()
            Total.tablename = self.total_tablename
            Train.tablename = self.detail_tablename
            Train.empty()
            Total.empty()


    def Pool(self):
        time0 = time.time()
        # 清空数据库
        self.switch()

        pool = Pool(processes=self.processCount)
        Uni = stock_uniform()
        Uni.tablename = self.uniform_tablename
        lists = Uni.top(90)

        print(lists)
        share = Manager2()
        Base, Rice = None, share.interface_Rice()

        for rs in lists:
            # 检查时间匹配
            if not 'RB' in rs: continue
            codes = rs
            for kline in self.klineList:
                self.start(codes, time0, kline, Base, Rice)
                # try:
                    #pool.apply_async(self.start, (codes, time0, kline, Base, Rice))

                # except Exception as e:
                #     print(e)
                #     continue

        pool.close()
        pool.join()

    cindex = 0

    def subInit(self, codes, time0, kline=None, Base=None, Rice=None):
        self.codes = codes
        self.Rice = Rice if Rice is not None else interface_Rice()
        # 查询获得配置 - 费率和每手单量
        self.Base = future_baseInfo() if Base is None else Base
        self.BS = {}

        for doc in self.Base.getInfo(codes):
            self.BS[doc["code"]] = doc

        cs = [self.BS[m] for m in codes]
        self.Rice.setTimeArea(cs[0]["nightEnd"])

        # 根据配置文件获取最佳交易手数对
        self.iniVolume = round(self.iniAmount / cs[0]["lastPrice"] / cs[0]["contract_multiplier"], 0)
        if self.iniVolume == 0: self.iniVolume = 1

        #
        self.Train = train_future()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename
        self.Train.tablename = self.detail_tablename

    def start(self, codes, time0, kline=None, Base=None, Rice=None):
        print("子进程启动:", self.cindex, codes, kline)
        self.subInit(codes, time0, kline=kline, Base=Base, Rice=Rice)

        self.klineType = kline
        self.mCodes = [c+'88' for c in codes]

        print(self.mCodes)
        # 查询获得N分钟K线
        dfs = self.Rice.kline(self.mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=1)

        # 分参数执行
        docs = self.total(dfs, period=self.period)
        if docs is None or len(docs) == 0: return

        logger.info((self.codes, self.klineType, len(docs), " time:", time.time() - time0))
        self.Total.insertAll(docs)

    pointColumns = ['mode', 'type', 's', 'o']
    def point(self, row):
        powm = 0

        columns = self.pointColumns
        return pd.Series([powm], index=columns)

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    def getMax(self, df0, s, e, mode):
        if mode > 0:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'high'].max()
        else:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'low'].min()

    def getLastMax(self, row, high, low):
        no, mio, mao, mas = (row[key] for key in "No,mio,mao,mas".split(","))
        return high[mao:no].max() if mas > 0 else low[mio:no].min()

    preNode, batchId = None, {}

    def total(self, dfs, period=14):
        # 计算参数
        self.dfs = dfs
        df0, df1 = (dfs[self.mCodes[i]] for i in [0, 1])

        df = pd.DataFrame(index=df0.index)

        df['close'] = close = df0['close'] / df1['close']
        df['volume'] = (df0['volume'] + df1['volume'])/2

        df["ma"] = ma = ta.MA(close, timeperiod=period)
        df["std"] = ta.STDDEV(close, timeperiod=period, nbdev=1)
        df["bias"] = (close - ma) /ma

        print(close.mean(), close.std())

        # 循环 scale
        docs = []
        for conds in self.iterCond():
            uid = self.uidKey % ('_'.join(self.codes),  self.klineType,  conds)

            df1 = df0.apply(lambda row: self.point(row), axis=1)
            for key in self.pointColumns:  df0[key] = df1[key]

            isIn = False
            for c in self.codes:
              if  c in self.cvsList:
                  isIn=True
                  break

            tot = self.detect(df, period= period, uid=uid)
            print(tot)
            if tot is not None and tot['amount'] != 0:
                tot.update(
                    {
                        "method": self.method,
                        "code": self.codes[0],
                        "code1": self.codes[1],
                        "period": period,
                        "uid": uid,
                        "createdate": public.getDatetime()
                    }
                )
                docs.append(tot)

        return docs

    def detect(self, df0, period=14, uid=''):
        docs = self.stageApply(df0,  period=period, uid=uid)
        if self.saveDetail:
            self.Train.insertAll(docs)

        res = pd.DataFrame(docs)

        if len(res) > 0:
            if self.saveDetail:
                self.Train.insertAll(docs)

            diff = res[res['diff'] > 0]['diff'].mean()

            # 计算最大回测
            sum = res['income'].cumsum() + self.iniAmount
            inr = res['income'] / self.iniAmount
            # 计算夏普指数

            sha = (res['income'].sum() / self.iniAmount - 0.02 * self.testDays / 252) / inr.std()
            return {
                "count": int(len(docs) / 2),
                "amount": self.iniAmount,
                "price": res['price'].mean(),
                "income": res["income"].sum(),
                "maxdown": ((sum.shift(1) - sum) / sum.shift(1)).max(),
                "sharprate": sha,
                "timediff": int(0 if np.isnan(diff) else diff)
            }
        else:
            return None

    # 核心策略部分
    def stageApply(self, df0, dfs, period=15, uid=''):
        isOpen, preDate, prePrice = 0, None, 0
        doc, docs = {}, []
        self.preNode = None
        for i in range(period, len(df0)):
            isRun, isstop, sp = False, 0, None
            powm, atr, kdjm, close, price = (df0.ix[i, key] for key in
                                             "powm,atr,kdjm,close,price".split(","))

            if isOpen == 0:
                if powm != 0:
                    isRun, isOpen, sp = True, int(-powm), price

            elif self.preNode is not None:
                if isOpen * kdjm < 0:
                    isRun, isOpen, isstop = True, 0, 1
                else:
                    # 止盈止损
                    preP = self.preNode[0]["price"]
                    if isOpen * ( preP - close) > 1.5 * atr:
                        isRun, isOpen, isstop = True, 0, 4

                    # 止盈反转
                    elif isOpen * powm > 0:
                        isRun, isOpen, isstop = True, 0, 3
                        doc = self.order(df0.iloc[i], isOpen, uid, dfs, isstop=isstop)
                        if doc is not None:
                            docs.append(doc)
                            isRun, isOpen, sp = True, int(-powm), price

            if isRun:
                doc = self.order(df0.iloc[i], isOpen, uid, dfs, isstop=isstop)
                if doc is not None:
                    docs.append(doc)
        return docs

    batchId = None

    # 下单

    def order(self, cur, mode, uid, dfs, isstop):
        # 使用uuid作为批次号
        if self.isOpen[self.uid] == mode and mode != 0:
            self.batchId[self.uid] = uuid.uuid1()

        # print(self.uid, mode, self.batchId[self.uid])

        # future_baseInfo 参数值
        b0, b1 = self.config[self.codes[0]], self.config[self.codes[1]]

        # 每次交易量
        v0 = self.iniVolume * b0["contract_multiplier"]
        v1 = round(v0 * cur["close"] / b1["contract_multiplier"], 0) * b1["contract_multiplier"]

        # 开仓 1/ 平仓 -1
        status = 0 if mode == 0 else 1

        # 买 / 卖 ,  若mode=0. 则按持仓反向操作
        isBuy = mode if not self.preNode[self.uid] else -self.preNode[self.uid][0]["mode"]

        # 费率
        fee0 = (self.iniVolume * b0["ratio"]) if b0["ratio"] > 0.5 else (
                b0["ratio"] * v0 * (cur["a1"] if isBuy == 1 else cur["b1"]))
        fee1 = (v1 / b1["contract_multiplier"] * b1["ratio"]) if b1["ratio"] > 0.5 else \
            (b1["ratio"] * v1 * (cur["n_a1"] if isBuy == -1 else cur["n_b1"]))

        doc = {
            "createdate": cur["datetime"],
            "code": self.codes[0],
            "price": cur["a1"] if isBuy == 1 else cur["b1"],
            "vol": v0,
            "mode": isBuy,
            "isopen": status,
            "isstop": self.isNotStop[self.uid][0],
            "fee": fee0,
            "income": 0,
            "rel_price": cur["p_l"] if isBuy == 1 else cur["p_h"],  # 实际交易价格(比价)
            "rel_std": param["std"],  # 标准差
            "batchid": self.batchId[self.uid],
            "uid": self.uid,
            "price_diff": isBuy * ((cur["p_l"] if isBuy == 1 else cur["p_h"]) - cur['close']),  #
            "delta": param["delta"],  # 价差/标准差
            "shift": param["shift"],  # 价差/tick_size
            "widthDelta": param["widthDelta"],
            "bullwidth": param["width"],  # 布林带宽度
            # "atr": param["atr"],  # 真实波幅
        }

        doc1 = {}
        doc1.update(doc)
        doc1.update({
            "code": self.codes[1],
            "price": cur["n_a1"] if isBuy == -1 else cur["n_b1"],
            "vol": v1,
            "mode": -isBuy,
            "fee": fee1,
        })

        #  计算income
        if mode == 0 and self.preNode[self.uid]:
            p0, p1 = self.preNode[self.uid][0], self.preNode[self.uid][1]
            doc["income"] = p0["mode"] * (doc["price"] - p0["price"]) * p0["vol"] - p0["fee"]
            doc1["income"] = p1["mode"] * (doc1["price"] - p1["price"]) * p1["vol"] - p1["fee"]
            self.preNode[self.uid] = None
        else:
            doc["income"] = -doc["fee"]
            doc1["income"] = -doc1["fee"]
            self.preNode[self.uid] = (doc, doc1)

        return [doc, doc1]


def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = train_future_2_hudge()
        obj.Pool()


if __name__ == '__main__':
    main()
