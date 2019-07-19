# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

   # 期货日线策略：
      超大ATR反向买入

"""

from com.base.public import public
import pandas as pd
import talib as ta
import numpy as np
from com.object.obj_entity import future_baseInfo
from com.data.interface_Rice import rq
from com.train.train_future_iland import train_future_iland, Manager2
import time
from multiprocessing import Pool

# 回归方法


# 回归方法
class train_future_maxAtr(train_future_iland):

    csvList = ['SC', 'FG', 'AU']


    def __init__(self):
        super().__init__()

        # 费率和滑点
        self.isEmptyUse = False     # 是否清空记录
        # self.iniAmount = 2000000 * 0.0025  # 单边50万

        self.fallLineList = [0.05]
        self.fallLine = 0.1
        self.klineList = ['30m', '60m']

        self.stopLineList = [2.5, 3, 3.5]
        self.stopLine = 3

        # 起始时间
        self.startDate = '2016-01-01'
        self.total_tablename = 'train_total_3'
        self.detail_tablename = 'train_future_3'
        self.method = 'maxAtr'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method
        self.emptyFilter = " method ='%s'" % self.method

    def iterCond(self):
        # 多重组合参数输出

        #for i in range(1):
        #    yield '%s' % (str(self.stopLine))

        keys = ['stopLine', 'fallLine']
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                 self.__setattr__(keys[1], s1)
                 yield '%s_%s' % (str(s0), str(s1))

        #        for s2 in self.__getattribute__(keys[2] + 'List'):
        #            self.__setattr__(keys[2], s2)


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

    pointColumns = ['powm', 'diss']

    def jump0(self, row):
        close, high, low, open, atrc = (row[key] for key in
                                                         "close,high,low,open,atrc".split(","))

        sign = 1 if close > open else -1 if close < open else 0
        max = high if sign > 0 else low
        fall = 0 if (sign == 0 or max == open) else (max - close) / (max - open)
        powm = 0
        if atrc > self.stopLine and fall > self.fallLine:
             powm = sign

        columns = self.pointColumns
        return pd.Series([powm, fall], index=columns)

    preNode, batchId = None, {}
    def total(self, dfs, period=14):
        # 计算参数
        df0 = dfs[self.mCodes[0]]

        close = df0["close"]
        df0["datetime"] = df0.index
        df0['mcode'] = rq.get_dominant_future(self.code, start_date=self.startDate, end_date=self.endDate)

        df0['atr'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=period)
        df0['atrc'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=1) / df0['atr']

        # kdj顶点
        kdjK, kdjD = ta.STOCH(df0["high"], df0["low"], close,
                              fastk_period=9, slowk_period=3, slowk_matype=1, slowd_period=3,
                              slowd_matype=1)

        df0["kdj_d2"] = kdj_d2 = kdjK - kdjD
        df0["kdjm"] = kdj_d2 * kdj_d2.shift(1)
        df0["kdjm"] = df0.apply(lambda row: self.turn(row['kdjm'], row['kdj_d2'], 1), axis=1)

        # 循环 scale
        docs = []
        for conds in self.iterCond():
            uid = self.uidKey % ('_'.join(self.codes), str(period),  self.klineType, str(self.bullwidth), conds)

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

    # 核心策略部分
    def stageApply(self, df0, period=15, uid=''):
        isOpen, preDate, prePrice = 0, None, 0
        doc, docs = {}, []
        self.preNode, self.jumpNode, self.jump_index = None, None, 0

        for i in range(period, len(df0)):
            isRun, isstop, sp = False, 0, None

            powm, close, kdjm, atr, dd = (df0.ix[i, key] for key in
                                             "powm,close,kdjm,atr,datetime".split(","))

            if isOpen == 0:
                if powm != 0:
                    isRun, isOpen, sp = True, int(powm), close

            elif self.preNode is not None:
                 pN = self.preNode[0]
                 sign = np.sign(pN['mode'])
                 pmax = self.getMax(df0, pN['createdate'], dd, sign)

                 if sign * powm < 0:
                     doc = self.order(df0.iloc[i], 0, uid,  df0, isstop=1, startPrice=close)
                     if doc is not None:
                         docs.append(doc)
                         isRun, isOpen, sp = True, int(powm), close

                 elif sign * kdjm < 0:
                     isRun, isOpen, isstop = True, 0, 2

                 elif sign * (pmax - close) > atr:
                     sp = close
                     #sp = pmax - sign * (self.stopLine + 0.2) * atr
                     #if sp < close * sign: sp = close
                     isRun, isOpen, isstop = True, 0, 3

            if isRun:
                doc = self.order(df0.iloc[i], isOpen, uid, df0, isstop=isstop, startPrice=sp)
                if doc is not None:
                     docs.append(doc)
        return docs

def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = train_future_maxAtr()
        obj.Pool()


if __name__ == '__main__':
    main()
