# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

   # 5分钟短线策略回测
    策略1：异常追涨杀跌，以5分钟k线为基准计算
    策略2：跳空预测


"""

from com.base.stat_fun import per, fisher
from com.base.public import public, logger, public_basePath
import pandas as pd
import talib as ta
import numpy as np
from com.object.obj_entity import train_future, train_total, future_baseInfo
import os
from com.data.interface_Rice import interface_Rice, tick_csv_Rice, rq
import itertools
import time
import uuid
from multiprocessing.managers import BaseManager
from multiprocessing import Pool
import copy
import json


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
class train_future_system(object):
    """

    """

    csvList = ['SC', 'FG', 'AU', 'IF']

    def __init__(self):
        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.topUse = False
        self.isEmptyUse = True  # 是否清空记录

        self.iniAmount = 500000  # 单边50万
        # self.iniAmount = 2000000 * 0.0025  # 单边50万
        self.stopLine = 2
        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.klineList = ['1m', '3m', '5m', '15m']

        # 起始时间
        self.startDate = '2019-04-06'
        self.endDate = public.getDate(diff=0)
        print(self.endDate)
        self.total_tablename = 'train_total_4'
        self.detail_tablename = 'train_future_4'
        self.method = 'system'
        self.uidKey = "%s_%s_" + self.method

    def empty(self):
        if self.isEmptyUse:
            Train = train_future()
            Total = train_total()
            Total.tablename = self.total_tablename
            Train.tablename = self.detail_tablename
            Train.empty()
            Total.empty()

    def category(self):
        cfg = public_basePath + "/com/data/future_category.json"
        with open(cfg, 'r', encoding='utf-8') as f:
            params = json.load(f)

        keys = copy.deepcopy(params).keys()

        for cat in keys:
            params[cat + 'List'] = []
            for s in params[cat]:
                params[cat + 'List'] += params[cat][s]

        return params

    def Pool(self):
        time0 = time.time()
        # 清空数据库
        self.empty()
        self.Base = future_baseInfo()
        self.Cat = self.category()
        for kline in self.klineList:
            self.start(kline)
        pass

    cindex = 0

    def start(self, kline):
        # 主力合约
        self.codes = codes = self.Cat['shortList'] + self.Cat['longList']
        self.mCodes = mCodes = [n + '88' for n in self.codes]

        # 查询获得配置 - 费率和每手单量
        self.Base = future_baseInfo()

        self.BS = {}
        for doc in self.Base.getInfo(codes):
            self.BS[doc["code"]] = self.BS[doc["code"] + '88'] = doc

        cs = [self.BS[m] for m in self.mCodes]
        # 子进程共享类
        self.Rice = interface_Rice()
        self.Rice.setTimeArea(cs[0]["nightEnd"])

        self.Train = train_future()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename
        self.Train.tablename = self.detail_tablename

        if len(self.indexCodeList) > 0:
            self.Rice.setIndexList(self.indexCodeList)

        self.klineType = kline
        # 查询获得N分钟K线
        print(self.startDate, self.endDate)

        dfs = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=1)

        print('kline load:', mCodes, [len(dfs[m]) for m in mCodes])

        # 分参数执行
        self.total(dfs)

    preNode, batchId = None, {}

    def total(self, dfs, period=14):
        # 计算参数
        indexs = dfs['A'].index
        Pa = pd.Panel(dfs, items=self.codes, major_axis=indexs)

        for c in self.codes:
            df = Pa[c]
            sign = 1 if c in self.Cat['longList'] else -1
            Pa.loc[c, :, 'volc'] = df['volume'] / ta.MA(df['volume'], timeperiod=15)
            Pa.loc[c, :, 'volc'].fillna(1)
            Pa.loc[c, :, 'diff'] = (df['close'] - df['open']) / df['open'] * 100
            Pa.loc[c, :, 'r'] = Pa.loc[c, :, 'diff'].apply(lambda x: 0 if np.isnan(x) else 1 if sign * x > 0 else -1)
            Pa.loc[c, :, 'r5'] = ta.SUM(Pa.loc[c, :, 'r'], timeperiod=5)

            docs = df.to_dict(orient='dict' )


        print(Pa.shape)

        res = pd.DataFrame(index=Pa.major_axis)
        res.loc[:, 'sum'] = Pa.loc[:, :, 'r'].apply(lambda x: x.sum(), axis=1)
        res.loc[:, 'count'] = Pa.loc[:, :, 'r'].apply(lambda x: x.abs().sum(), axis=1)
        res.loc[:, 'rate'] = res.loc[:, 'sum'] / res.loc[:, 'count']
        res.loc[:, 'diff'] = Pa.loc[:, :, 'diff'].apply(lambda x: x.mean(), axis=1)
        res.loc[:, 'std'] = Pa.loc[:, :, 'diff'].apply(lambda x: x.std(), axis=1)
        res.loc[:, 'volc'] = Pa.loc[:, :, 'volc'].apply(lambda x: x.mean(), axis=1)
        res.loc[:, 'sum5'] = Pa.loc[:, :, 'r5'].apply(lambda x: x.mean(), axis=1)
        res.loc[:, 'rate5'] = ta.MA(res['rate'], timeperiod=5)

        file = self.Rice.basePath + 'result_%s_system.csv' % self.klineType
        res.to_csv(file, index=1)
        """

        # 循环 scale
        #docs = self.stageApply(df0, period=period)
        docs = []
        if self.saveDetail:
             self.Train.insertAll(docs)
        """

    # 核心策略部分
    def stageApply(self, df0, period=15, uid=''):
        isOpen, preDate, prePrice = 0, None, 0
        doc, docs = {}, []
        self.preNode, self.jumpNode, self.jump_index = None, None, 0

        for i in range(period, len(df0)):
            isRun, isstop, sp = False, 0, None

            jumps, open, close, atr, dd = (df0.ix[i, key] for key in
                                           "jumps,open,close,atr,datetime".split(","))

            if isOpen == 0:
                if jumps != 0:
                    isRun, isOpen, sp = True, int(jumps), open

            elif self.preNode is not None:
                pN = self.preNode[0]
                sign = np.sign(pN['mode'])
                pmax = self.getMax(df0, pN['createdate'], dd, sign)

                if sign * jumps < 0:
                    doc = self.order(df0.iloc[i], 0, uid, df0, isstop=2, startPrice=open)
                    if doc is not None:
                        docs.append(doc)
                        isRun, isOpen, sp = True, int(jumps), open

                elif sign * (pmax - close) > atr:
                    isRun, isOpen, isstop = True, 0, 1

            if isRun:
                doc = self.order(df0.iloc[i], isOpen, uid, df0, isstop=isstop, startPrice=sp)
                if doc is not None:
                    docs.append(doc)
        return docs

    batchId = None

    def calcIncome(self, n0, p0, df0):
        # 计算收益，最大/最小收益
        high = df0[(p0['createdate'] <= df0.index) & (df0.index <= n0['datetime'])]['high'].max()
        low = df0[(p0['createdate'] <= df0.index) & (df0.index <= n0['datetime'])]['low'].min()
        close = n0["close"]
        sign = p0["mode"] / abs(p0["mode"])

        # 收入
        income = sign * (close - p0["price"]) * p0["vol"] - p0["fee"]
        # 最大收入
        highIncome = sign * ((high if sign > 0 else low) - p0["price"]) * p0["vol"] - p0["fee"]
        # 最大损失
        lowIncome = sign * ((high if sign < 0 else low) - p0["price"]) * p0["vol"] - p0["fee"]

        return income, highIncome, lowIncome, high, low

    def order(self, n0, mode, uid, df0, isstop=0, startPrice=None):
        # BS 配置文件，查询ratio 和 每手吨数
        b0 = self.BS[self.mCode]
        if mode != 0:
            self.batchId = uuid.uuid1()
        # 交易量
        v0 = self.iniVolume * b0["contract_multiplier"]
        # 费率
        fee0 = (self.iniVolume * b0["ratio"] * 1.1) if b0["ratio"] > 0.5 else ((b0["ratio"] * 1.1) * n0["close"] * v0)

        ps = startPrice if startPrice is not None else n0["close"]
        amount = v0 * ps * float(b0['margin_rate'])

        doc = {
            "createdate": n0["datetime"],
            "code": self.codes[0],
            "price": ps,
            "vol": self.preNode[0]["vol"] if self.preNode else v0,
            "hands": self.iniVolume,
            "amount": amount if not self.preNode else self.preNode[0]["amount"],
            "mode": int(mode) if not self.preNode else -self.preNode[0]["mode"],
            "isopen": 0 if mode == 0 else 1,
            "fee": fee0,
            "income": 0,
            "isstop": isstop,
            "shift": n0['slope'] if 'slope' in n0 else 0,
            "rel_std": n0["std"] if 'std' in n0 else 0,
            "delta": n0['diss'] if 'diss' in n0 else 0,
            "batchid": self.batchId,
            "diff": 0 if mode != 0 else self.Rice.timeDiff(str(self.preNode[0]['createdate']), str(n0["datetime"])),
            "uid": uid
        }

        if mode == 0 and self.preNode:
            p0 = self.preNode[0]
            doc['income'], doc['highIncome'], doc['lowIncome'], doc['atr'], doc['macd'] = self.calcIncome(n0, p0, df0)
            doc["diff"] = int(public.timeDiff(str(n0['datetime']), str(p0['createdate'])) / 60)

            self.preNode = None
        else:
            doc["income"] = -doc["fee"]
            self.preNode = [doc]
        return doc


def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = train_future_system()
        obj.Pool()


if __name__ == '__main__':
    main()
