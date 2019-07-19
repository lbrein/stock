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

from com.data.interface_Rice import interface_Rice, tick_csv_Rice
import time
import uuid
from multiprocessing import Pool

from com.train.train_future_follow5 import train_future_follow5, Manager2

# 回归方法
class train_future_follow1(train_future_follow5):
    """

    """

    csvList = ['RB', 'AP', 'FG', 'AU', 'B']

    def __init__(self):
        super().__init__()

        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.topUse = False
        self.isEmptyUse = True  # 是否清空记录

        self.testDays = 20
        self.processCount = 5
        self.iniAmount = 500000  # 单边50万

        self.volcLineList = [4, 6]
        self.volcLine = 5

        self.stopLine = 3
        self.maxperiodList = [5, 15, 30]
        self.maxperiod = 5

        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.klineList = ['1m', '5m', '10m']

        # 起始时间
        self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测
        self.endDate = public.getDate(diff=0)
        self.total_tablename = 'train_total_0'
        self.detail_tablename = 'train_future_0'
        self.method = 'follow1'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method

    def iterCond(self):
        # 多重组合参数输出
        #for i in range(1):
        #    yield '%s_%s' % (str(self.bullLine), str(self.atrLine))

        keys = ['volcLine', 'maxperiod']
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                self.__setattr__(keys[1], s1)
                #for s2 in self.__getattribute__(keys[2] + 'List'):
                    #self.__setattr__(keys[2], s2)
                yield '%s_%s' % (str(s0), str(s1))

    def Pool(self):
        time0 = time.time()
        # 清空数据库
        self.switch()

        pool = Pool(processes=self.processCount)
        share = Manager2()
        Base = share.future_baseInfo()
        Rice = None
        lists = Base.getUsedMap(hasIndex=True, isquick=True)

        for rs in lists:
            # 检查时间匹配
            #print(rs)
            if not rs in self.csvList: continue
            codes = [rs]
            # try:
            #kline = '1m'
            for kline in self.klineList:
                print(codes, kline)
                #self.start(codes, time0, kline, Base, Rice)
                pool.apply_async(self.start, (codes, time0, kline, Base, Rice))
            #break
            # except Exception as e:
            #     print(e)
            #     continue

        pool.close()
        pool.join()

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    pointColumns = ['powm', 'diss', 'curPrice', 'curMA']
    def point(self, row, b0, df0):
        close,open,high,low,volr, min10, max10, atr, date = \
            (row[key] for key in
             "close,open,high,low,volr,min10,max10,atr,datetime".split(","))

        tick, kline = b0['tick_size'], int(b0['quickkline'][:-1]) * 2
        if kline =='': kline = '3'

        ss = '0' + str(kline) if kline < 10 else str(kline)
        tt = str(public.parseTime(str(date), style='%H:%M:%S'))

        VL = self.volcLine
        if '09:00:00' <= tt <= ('09:%s:00' % ss) or '21:00:00' <= tt <= ('09:%s:00' % ss):
            VL = VL - 2

        preDate = public.getDate(diff=-1, start=str(date)[:10]) + ' 21:00:00'
        df1 = df0[(df0['datetime']>=preDate) & (df0['datetime']<=date)]
        curMA = round((df1['close'] * df1['volume']).sum() / df1['volume'].sum() / tick, 0) * tick

        sign = np.sign(close - open)
        trend = (high - close) if sign > 0 else (close - low)
        opt0 = (sign > 0 and close > max10) or (sign < 0 and close < min10)

        powm, price = 0, 0

        if opt0 and volr > VL:
              price = max10 + 2 * tick if sign< 0 else min10 - 2 * tick
              if sign * (open - price) > 0:
                  price = open + sign * 5 * tick

              if sign * (price - curMA) > 0:
                  powm = sign

        columns = self.pointColumns
        return pd.Series([powm, trend, price,curMA], index=columns)

    def total(self, dfs, period=14):
        # 计算参数
        df0 = dfs[self.mCodes[0]]
        # print(df0.iloc[-1])

        b0 = self.BS[self.code]
        #rate, mar, mul = (b0[c] for c in ['ratio', 'margin_rate', 'contract_multiplier'])

        close = df0["close"]
        df0["datetime"] = df0.index
        df0["datetime"] = df0['datetime'].astype('str')

        #print(df0.index[-1])

        df0['delta'] = df0['volr'] = df0['volume'] / ta.MA(df0['volume'], timeperiod=30)
         #df0['predate'] = df0["datetime"].apply(lambda x: public.getDate(diff=-1, start=str(x).split(' ')[0])+' 21:00:00')

        df0['atr'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=30)

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

            df0['max10'] = ta.MAX(df0['high'].shift(1), timeperiod=self.maxperiod)
            df0['min10'] = ta.MIN(df0['low'].shift(1), timeperiod=self.maxperiod)

            df1 = df0.apply(lambda row: self.point(row, b0, df0), axis=1)
            for key in self.pointColumns:  df0[key] = df1[key]

            if self.code in self.csvList:
                file = self.Rice.basePath + '%s_pre.csv' % (uid)
                print(uid, '---------------------------- to_cvs', file, df0.columns)
                df0.to_csv(file, index=0)

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
        self.preNode = None
        for i in range(period, len(df0)):
            isRun, isstop, sp = False, 0, None
            powm, atr, close, cp, kdjm, date = (df0.ix[i, key] for key in
                                             "powm,atr,close,curPrice,kdjm,datetime".split(","))

            tt = public.parseTime(str(date), style='%H:%M:%S')

            if '09:00:00'<= tt < '09:05:00' or '21:00:00'<= tt < '21:05:00' : continue
            if isOpen == 0 :
                if powm!=0:
                    isRun, isOpen = True, int(powm)
                    sp = cp

            elif self.preNode is not None:
                preP, s = self.preNode[0]['price'], self.preNode[0]['createdate']
                mp = self.getMax(df0, s, date, isOpen)
                sign = np.sign(isOpen)
                mp = close if np.isnan(mp) else mp

                if isOpen * kdjm < 0:
                    isRun, isOpen, isstop = True, 0, 1

                elif sign * (preP - close) > 1.25 * atr:
                    isRun, isOpen, isstop = True, 0, 2

                elif not np.isnan(mp) and (sign * (mp - close) > 2 * atr):
                    isOpen, isRun, isstop = 0, True, 3
                    pass

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
        obj = train_future_follow1()
        obj.Pool()


if __name__ == '__main__':
    main()
