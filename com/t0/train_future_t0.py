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
from com.data.interface_Rice import interface_Rice, tick_csv_Rice
import itertools
import time
import uuid
from multiprocessing.managers import BaseManager
from multiprocessing import Pool
import copy
import datetime

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
class train_future_t0(object):
    """

    """

    csvList = ['SC']

    csvKey = '%s_14_%s_2_2_3.5_0.5'

    def __init__(self):
        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.topUse = False
        self.isEmptyUse = True    # 是否清空记录

        self.testDays = 2
        self.processCount = 5

        self.period = 14
        self.bullwidth = 2

        self.atrLineList = [2]
        self.bullLineList = [3.5]
        self.startTickList =[0.5]
        self.stdcTickList = [1.25]

        self.bullLine = 3.5
        self.atrLine = 2
        self.startTick = 0.25
        self.stdcTick = 1.25

        self.iniAmount = 50000  # 单边50万

        # self.iniAmount = 2000000 * 0.0025  # 单边50万
        self.stopLine = 2
        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.klineList = ['3m']

        # 起始时间
        self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测

        self.endDate = public.getDate(diff=1)
        print(self.endDate)
        self.total_tablename = 'train_total_1'
        self.detail_tablename = 'train_future_1'
        self.method = 't0'
        self.uidKey = "%s_%s_%s_" + self.method
        self.isAll = 0

    def iterCond(self):
        # 多重组合参数输出
        for i in range(1):
            yield '%s_%s' % (str(self.bullLine), str(self.atrLine))
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
        share = Manager2()
        Base = share.future_baseInfo()
        Rice = None
        lists = Base.getUsedMap(hasIndex=True, isquick=True)

        print(len(lists))
        for rs in lists:
            # 检查时间匹配
            if not rs in self.csvList: continue

            codes = [rs]
            self.start(codes, time0, None, Base, Rice)
            # try:
                #pool.apply_async(self.start, (codes, time0, kline, Base, Rice))

            # except Exception as e:
            #     print(e)
            #     continue

        pool.close()
        pool.join()

    cindex = 0

    def start(self, codes, time0, kline=None, Base=None, Rice=None):
        print("子进程启动:", self.cindex, codes, kline)
        self.codes = codes

        # 查询获得配置 - 费率和每手单量
        self.Base = future_baseInfo() if Base is None else Base
        self.BS = {}
        for doc in self.Base.getInfo(codes):
            self.BS[doc["code"]] = doc
        cs = [self.BS[m] for m in codes]

        self.klineType = 'tick'

        # 子进程共享类
        self.Rice = Rice if Rice is not None else interface_Rice()
        # 主力合约
        self.mCodes = self.Rice.getMain(codes)

        self.Rice.setTimeArea(cs[0]["nightEnd"])

        self.Train = train_future()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename
        self.Train.tablename = self.detail_tablename

        if len(self.indexCodeList) > 0:
            self.Rice.setIndexList(self.indexCodeList)

        # 查询获得N分钟K线
        dfs = self.Rice.kline(self.mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=1)

        df0 = dfs[self.mCodes[0]]
        #print('kline load:', mCodes, [len(dfs[m]) for m in mCodes])

        # 根据配置文件获取最佳交易手数对
        self.iniVolume = round(self.iniAmount / cs[0]["lastPrice"] / cs[0]["contract_multiplier"], 0)
        if self.iniVolume == 0: self.iniVolume = 1

        # 分参数执行
        docs = self.total(dfs, period=self.period)

        if docs is None or len(docs) == 0: return

        logger.info((self.codes, self.klineType, len(docs), " time:", time.time() - time0))
        self.Total.insertAll(docs)

    pointColumns = ['mode', 'type', 's', 'o']
    def calcMode(self, row):
        last, delta, vol, ra, a1, b1 , ap, bp = (row[key] for key in
                                         "last,delta,vol,raise,a1,b1,ap,bp".split(","))

        type =  0 if (delta == 0 and vol == 0) else \
                1 if (delta == 0 and vol > 0) else \
                2 if (delta > 0 and vol == delta) else \
                3 if (delta > 0 and vol != delta) else \
                4 if (delta < 0 and (vol + delta) == 0) else \
                5 if (delta < 0 and (vol + delta) != 0) else \
                6

        mode = 1 if (last >= ap or last >=a1) else -1 if (last<=bp or last<=b1) else 0

        s = abs(delta)/2
        o = vol/2 - s

        columns = self.pointColumns
        return pd.Series([mode, type, s, o], index=columns)

    def curMA(self, row, df0):
        td, e = row['trading_date'], row['datetime']
        df = df0[(df0['trading_date'] == td) & (df0['datetime'] <=e)]
        return (df['last'] * df['volume']).sum()/df['volume'].sum() if df['volume'].sum()!=0 else 0

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
        df0 = dfs[self.mCodes[0]]
        # print(df0.iloc[-1])
        code = self.codes[0]
        #b0 = self.BS[code]
        #rate, mar, mul = (b0[c] for c in ['ratio', 'margin_rate', 'contract_multiplier'])

        df0["datetime"] = df0.index

        print(df0.columns)

        df0["curMa"] = df0.apply(lambda row: self.curMA(row, df0),  axis=1)
        df0["ap"], df0["bp"] = df0['a1'].shift(1), df0['b1'].shift(1)
        df0["vol"] = df0['volume'].diff()
        df0["delta"] = df0['open_interest'].diff()
        df0["raise"] = df0['last'].diff()

        # 循环 scale
        docs = []
        for conds in self.iterCond():
            uid = self.uidKey % ('_'.join(self.codes),  self.klineType,  conds)

            df1 = df0.apply(lambda row: self.calcMode(row), axis=1)
            for key in self.pointColumns:  df0[key] = df1[key]

            if code in self.csvList:
                file = self.Rice.basePath + '%s_pre.csv' % (uid)
                print(uid, '---------------------------- to_cvs', file, df0.columns)
                df0.to_csv(file, index=0)

            return
            tot = self.detect(df0, period=period, uid=uid)
            print(tot)
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
            print(self.iniAmount)
            sha = (res['income'].sum() / self.iniAmount - 0.02 * self.testDays / 252) / inr.std()

            return {
                "count": int(len(docs) / 2),
                "amount": self.iniAmount,
                "price": res['price'].mean(),
                "income": res["income"].sum(),
                # "delta": res['delta'].mean(),
                "maxdown": ((sum.shift(1) - sum) / sum.shift(1)).max(),
                "sharprate": sha,
                "timediff": int(0 if np.isnan(diff) else diff)
            }
        else:
            return None

    # 核心策略部分
    def stageApply(self, df0, period=15, uid=''):
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
                        doc = self.order(df0.iloc[i], isOpen, uid, df0, isstop=isstop, startPrice=price)
                        if doc is not None:
                            docs.append(doc)
                            isRun, isOpen, sp = True, int(-powm), price

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
        obj = train_future_t0()
        obj.Pool()


if __name__ == '__main__':
    main()
