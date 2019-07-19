# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

   # 5分钟短线策略回测
    策略1：异常追涨杀跌，以5分钟k线为基准计算
    策略2：跳空预测


"""

from com.base.stat_fun import per, fisher
from com.base.public import public, logger
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
class train_future_iland(object):
    """

    """

    csvList = ['SC', 'FG', 'AU', 'IF']

    def __init__(self):
        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.topUse = False
        self.isEmptyUse = True   # 是否清空记录

        self.testDays = 1200

        self.bullwidth = 2
        self.periodwidth = 14
        self.processCount = 5

        self.preDaysList = [10]
        self.preDays = 5
        self.dissLineList = [0.05]
        self.dissLine = 0.5

        self.iniAmount = 500000  # 单边50万

        # self.iniAmount = 2000000 * 0.0025  # 单边50万
        self.stopLine = 2
        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.klineList = ['1d']

        # 起始时间
        self.startDate = '2013-01-01'
        self.endDate = public.getDate(diff=0)
        print(self.endDate)
        self.total_tablename = 'train_total_4'
        self.detail_tablename = 'train_future_4'
        self.method = 'jump'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method
        self.isAll = 0

    def iterCond(self):
        # 多重组合参数输出
        #for i in range(1):
        #    yield '%s_%s' % (str(self.preDays), str(self.keepDays))

        keys = ['dissLine', 'preDays']
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                 self.__setattr__(keys[1], s1)
                 yield '%s_%s' % (str(s0),str(s1))

        #        for s2 in self.__getattribute__(keys[2] + 'List'):
        #            self.__setattr__(keys[2], s2)


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
        share = Manager2()
        Base = share.future_baseInfo()
        Rice = None
        lists = Base.getUsedMap(hasIndex=True, isquick=True)

        print(len(lists))

        for rs in lists:
            # 检查时间匹配
            #if not rs in self.csvList: continue

            codes = [rs]
            for kline in self.klineList:
                # try:
                #self.start(codes, time0, kline, Base, Rice)
                pool.apply_async(self.start, (codes, time0, kline, Base, Rice))
                #break
            # except Exception as e:
            #     print(e)
            #     continue

        pool.close()
        pool.join()

    cindex = 0

    def start(self, codes, time0, kline=None, Base=None, Rice=None):
        print("子进程启动:", self.cindex, codes, kline)
        # 主力合约
        self.codes = codes
        self.code, self.mCode = codes[0], codes[0] + '88'

        self.mCodes = mCodes = [n + '88' for n in codes]
        # 查询获得配置 - 费率和每手单量

        self.Base = future_baseInfo()

        self.BS = {}

        for doc in self.Base.getInfo(codes):
            self.BS[doc["code"]] = self.BS[doc["code"] + '88'] = doc

        cs = [self.BS[m] for m in self.mCodes]
        self.klineType = self.BS[self.code]['quickkline'] if kline is None else kline

        print(self.code, self.klineType)

        # 子进程共享类
        self.Rice = Rice if Rice is not None else interface_Rice()
        self.Rice.setTimeArea(cs[0]["nightEnd"])

        self.Train = train_future()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename
        self.Train.tablename = self.detail_tablename

        if len(self.indexCodeList) > 0:
            self.Rice.setIndexList(self.indexCodeList)

        # 查询获得N分钟K线
        dfs = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=1)

        print('kline load:', mCodes, [len(dfs[m]) for m in mCodes])

        # 根据配置文件获取最佳交易手数对
        self.iniVolume = round(self.iniAmount / cs[0]["lastPrice"] / cs[0]["contract_multiplier"], 0)
        if self.iniVolume == 0: self.iniVolume = 1

        # 分参数执行
        docs = self.total(dfs, period=self.periodwidth)
        if docs is None or len(docs) == 0: return

        logger.info((self.codes, self.klineType, len(docs), " time:", time.time() - time0))
        self.Total.insertAll(docs)

    pointColumns = ['jumps', 'diss']

    def jump0(self, row):
        open, std, high_p, low_p, high_5, low_5 = (row[key] for key in
                                                         "open,std,high_p,low_p,high_5,low_5".split(","))
        opt0 = open > high_p and open > high_5
        opt1 = open < low_p and open < low_5

        sign = 1 if open > high_p else -1

        diss = (open - high_p)/std if sign > 0 else (low_p -open)/std
        powm =  sign if (opt0 or opt1) and (diss > self.dissLine and diss < 1)   else 0
        columns = self.pointColumns
        return pd.Series([powm, diss], index=columns)

    def getMax(self, df0, s, e, mode):
        s = str(s)[:10]
        if mode > 0:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:, 'high'].max()
        else:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:, 'low'].min()

    preNode, batchId = None, {}

    def total(self, dfs, period=14):
        # 计算参数
        df0 = dfs[self.mCodes[0]]

        close = df0["close"]
        df0["datetime"] = df0.index
        df0['mcode'] = rq.get_dominant_future(self.code, start_date=self.startDate, end_date=self.endDate)

        df0["std"] = ta.STDDEV(close.shift(1), timeperiod=period, nbdev=1)
        df0['high_p'] = df0['high'].shift(1)
        df0['low_p'] = df0['low'].shift(1)
        df0['atr'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=period)

        # 循环 scale
        docs = []
        for conds in self.iterCond():
            uid = self.uidKey % ('_'.join(self.codes), str(period),  self.klineType, str(self.bullwidth), conds)

            df0['high_5'] = ta.MAX(df0['high'].shift(1), timeperiod=self.preDays)
            df0['low_5'] = ta.MIN(df0['low'].shift(1), timeperiod=self.preDays)
            df1 = df0.apply(lambda row: self.jump0(row), axis=1)
            for key in self.pointColumns:  df0[key] = df1[key]

            if self.code in self.csvList:
                file = self.Rice.basePath + '%s_pre.csv' % (uid)
                print(uid, '---------------------------- to_cvs', file, df0.columns)
                df0.to_csv(file, index=0)

            #tot = None
            tot = self.detect(df0, period=period, uid=uid)
            if tot is not None and tot['amount'] != 0:
                tot.update(
                    {
                        "method": self.method,
                        "code": self.code,
                        "period": period,
                        "uid": uid,
                        "createdate": public.getDatetime()
                    }
                )
                docs.append(tot)
        return docs

    def detect(self, df0, period=14, uid=''):
        docs = self.stageApply(df0, period=period, uid=uid)
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

            doc = {
                "count": int(len(docs) / 2),
                "amount": self.iniAmount,
                "price": res['price'].mean(),
                "income": res["income"].sum(),
                # "delta": res['delta'].mean(),
                "maxdown": ((sum.shift(1) - sum) / sum.shift(1)).max(),
                "sharprate": sha,
                "timediff": int(0 if np.isnan(diff) else diff)
            }
            print(self.code, doc['count'],doc['income'])
            return doc
        else:
            return None

    def getJump0(self, df0, start, mode):
        for i in range(start-3, start):
            print(i)

        df = df0.iloc[start-3:start]
        print(df)


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
                     doc = self.order(df0.iloc[i], 0, uid,  df0, isstop=2, startPrice=open)
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
        obj = train_future_iland()
        obj.Pool()


if __name__ == '__main__':
    main()
