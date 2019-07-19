# -*- coding: utf-8 -*-
"""
Created on 12 19 -2018
@author: lbrein


      反弹策略回测： 添加obv参数


"""

from com.base.public import public, logger
import numpy as np
import pandas as pd
import talib as ta
import uuid
from com.data.interface_Rice import interface_Rice
from com.object.obj_entity import stock_orderForm, stock_baseInfo
from com.object.mon_entity import mon_trainOrder
from multiprocessing import Pool, Manager
import time
import copy

# 选股
class train_etf50_kline(object):

    def __init__(self):
        self.period = '1d'
        self.pageCount =50

        self.timePeriodsList = [60]
        self.timePeriods = 150

        self.dropLineList = [1.0, 1.5]
        self.dropLine = 0.20

        self.sarStartList = [0.02, 0.03, 0.04, 0.05]
        self.sarStart = 0.02
        self.sarEndList = [0.02, 0.03, 0.04, 0.05]
        self.sarEnd = 0.05

        self.iniAmount = 250000
        self.shift = 0.002
        self.ratio = 0.0006
        self.atrDropLine = 2

        self.saveMongo = True
        self.methodName = 'obv'

        self.startDate = public.getDate(diff=-200)  # 60天数据回测
        self.endDate = public.getDate(diff=0)
        self.iterCondList = ['timePeriods', 'dropLine', 'sarStart', 'sarEnd']

    def empty(self):
        Record = stock_orderForm()
        Record.tablename = 'stock_orderForm_train'
        Record.empty(filter=" method='%s'" % self.methodName)
        if self.saveMongo:
            TrainOrder = mon_trainOrder()
            TrainOrder.empty({'method': '%s' % self.methodName})

    def pool(self):
        pool = Pool(processes=6)
        self.empty()

        Base = stock_baseInfo()
        lists = Base.getCodes(isBound=0)

        for k in range(0, len(lists), self.pageCount):
            codes = lists[k:k+self.pageCount]
            #self.start(codes, int(k/self.pageCount+1))
            try:
                print(k)
                pool.apply_async(self.start, (codes, int(k/self.pageCount+1)))
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

    def iterCond(self):
        # 多重组合参数输出
        keys = self.iterCondList
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                self.__setattr__(keys[1], s1)

                for s2 in self.__getattribute__(keys[2] + 'List'):
                    self.__setattr__(keys[2], s2)

                    for s3 in self.__getattribute__(keys[3] + 'List'):
                        self.__setattr__(keys[3], s3)

                        yield '%s_%s_%s_%s' % (str(s0), str(s1), str(s2), str(s3))


    def getMax(self, df0, s, e, mode):
        if mode > 0:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'close'].max()
        else:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'close'].min()

    # 分段布林策略
    def start(self, codes, n):
        time0 = time.time()
        print('process %s start:' % str(n))
        self.Rice = interface_Rice()
        self.Record = stock_orderForm()

        self.Record.tablename = 'stock_orderForm_train'
        self.TrainOrder = mon_trainOrder()

        self.codes = codes
        res = self.Rice.kline(codes, period=self.period, start=self.startDate, end=self.endDate, pre=90)

        for code in codes:
             for conds in self.iterCond():

                    self.uid = '%s_%s_pop' % (code.replace('.', '_'), conds)
                    self.batchid = uuid.uuid1()

                    df = res[code]
                    # 计算统一特征
                    df['createTime'] = df.index
                    df = self.add_stock_index(df)

                    df['code'] = code
                    df['mode'] = df.apply(lambda row: self.point(row), axis=1)

                    if code.find('300538') > -1 or code.find('002365') > -1:
                        file = self.Rice.basePath + '%s.csv' % self.uid
                        print(file)
                        df.to_csv(file, index=1)

                    self.saveStage(df)

        print('process %s end: %s ' % (str(n),str(time.time()-time0)))

    def point(self, row):
        drop, sm, sm5, sard, obvm, bias20 = (row[k] for k in 'drop,sarm,sarm5,sard,obvm,bias20'.split(','))
        mode = 0
        opt0 = (sm == 1 and obvm > 0) or (sard > 0 and obvm==2)
        if self.dropLine!=-1:
            opt1 = drop > 0.2 and bias20 < self.dropLine
        else:
            opt1 = drop > 0.2

        if (opt0) and opt1:
            mode = 1

        elif sm5 == -1:
            mode = -1

        return mode

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    def mid(self, row, close):
        c1 = close[close.index <= row['createTime']][row['miw']:]
        return c1.max()

    def isObvm(self, row):
        obv, obv60, obv_t = row['obv'], row['obv60'], row['obv_t']
        return 2 if obv > obv60 and obv_t < 0 else 1 if obv > obv60 else -1

    def add_stock_index(self, df0):

        close = df0["close"]

        df0['max90'] = mx =  ta.MAX(close, timeperiod=self.timePeriods)
        df0['drop'] = (mx-close) / mx

        sar = ta.SAR(df0['high'], df0['low'], acceleration=self.sarStart, maximum=0.2)
        df0['sard'] = sard = close - sar
        df0['sarm'] = sard * sard.shift(1)
        df0['sarm'] = df0.apply(lambda row: self.turn(row['sarm'], row['sard'], 1), axis=1)

        sar5 = ta.SAR(df0['high'], df0['low'], acceleration=self.sarEnd, maximum=0.2)
        df0['sard5'] = sard5 = close - sar5
        df0['sarm5'] = sard5 * sard5.shift(1)
        df0['sarm5'] = df0.apply(lambda row: self.turn(row['sarm5'], row['sard5'], 1), axis=1)

        # 计算能量潮和60日均线
        df0['obv'] = obv = ta.OBV(close, df0['volume'])
        df0['obv60'] = obv60 = ta.MA(obv, timeperiod=60)
        df0['obvs'] = ta.LINEARREG_SLOPE(obv/obv60, timeperiod=3)

        df0['obvr'] = (obv-obv60)/obv60
        df0['obv_t'] = (obv-obv60) * (obv.shift(1)-obv60.shift(1))
        df0['obvm'] = df0.apply(lambda row: self.isObvm(row), axis=1)

        # 乖离度
        for p in [5, 10, 20]:
            df0['ma' + str(p)] = ma = ta.MA(close, timeperiod=p)
            df0["bias" + str(p)] = (close - ma) / ma * 100

        # 鳄鱼线

        # 14根ATR
        df0['atr'] = atr = ta.ATR(df0['high'], df0['low'], close, timeperiod=1)
        df0['atrb'] = atr14 = ta.ATR(df0['high'], df0['low'], close, timeperiod=14)
        df0['atrr'] = atr/atr14

        return df0

    # 返回买入后最大值close
    def getMax(self, df0, s, e, mode):
        if mode > 0:
            return df0[(df0.index>=s) & (df0.index<e)].ix[:-1, 'close'].max()
        else:
            return df0[(df0.index >= s) & (df0.index < e)].ix[:-1,'close'].min()

    #
    def saveStage(self, df2):
        self.preNode =  None
        period, ini = 60, self.iniAmount
        self.mon_records,  self.records = [], []

        for i in range(period, len(df2)):
            mode, close, date , atrb = (df2.ix[i, key] for key in "mode,close,createTime,atrb".split(","))

            isBuy, isRun = -1, False
            pN = self.preNode

            # 部分加仓
            if pN is None and mode ==1 :
                isBuy, isRun, mode = 1, True, 1

            elif pN is not None:
                if mode==-1:
                   isBuy, isRun, mode = -1, True, -1

                else:
                   # ATR 止损
                   s, e = pN['createdate'], date
                   px = self.getMax(df2, s, e, 1)
                   if (px - close) > self.atrDropLine * atrb:
                       isBuy, isRun, mode = -1, True, -2

            if isRun:
                self.order(df2.iloc[i], isBuy, mode)

        # 保存明细
        if len(self.records) > 0:
           self.Record.insertAll(self.records)

           if self.saveMongo and len(self.mon_records) > 0:
               #print("monsave", self.uid, self.mon_records)
               self.TrainOrder.col.insert_many(self.mon_records)

    preTick = None
    def mon_saveTick(self, n0, doc):
        tick = copy.deepcopy(n0.to_dict())
        tick.update(doc)
        for key in ['sarm', 'sarm5', 'mode', 'obvm']:
            if key in tick: tick[key] = int(tick[key])

        if doc['isBuy'] == -1:
            self.mon_records.append(tick)
            if self.preTick is not None:
                self.preTick['income'] = doc['income']
                self.preTick['enddate'] = doc['createdate']
                self.mon_records.append(copy.deepcopy(self.preTick))

            self.preTick = None
        else:
            self.preTick = tick


    def order(self, n0, isBuy, mode):
        pN = self.preNode
        now = public.getDatetime()
        vol, fee, amount,income, p0 = 0,0,0,0, 0
        price = n0["close"]
        if isBuy > 0 :
            self.batchid =  uuid.uuid1()
            p0 = price * (1+ self.shift)
            vol = int(self.iniAmount/p0)
            amount = vol * p0
            fee = vol * p0 * self.ratio
            income = -fee

        elif isBuy < 0:
            p0 = price * (1 - self.shift)
            vol = pN['vol']
            amount = vol * p0
            fee = vol * p0 * self.ratio
            income = amount - pN['amount']-fee

        doc = {
            "code": n0['code'],
            "name": n0['code'],
            "createdate": n0['createTime'],
            "price": p0,
            "vol": vol,
            "mode": int(mode),
            "isBuy": int(isBuy),
            "fee": fee,
            "amount": amount,
            "income": income,
            "method": self.methodName,
            "batchid": self.batchid,
            "uid": self.uid
        }

        self.records.append(doc)
        if self.saveMongo:
            self.mon_saveTick(n0, doc)

        # 设置上一个记录
        if isBuy > 0:
            self.preNode = doc
        else:
            self.preNode = None

        return True



def main():
    actionMap = {
        "start": 1,  #
        "filter": 0,
        "stat": 0,
        "atr": 0,
    }
    obj = train_etf50_kline()

    if actionMap["start"] == 1:
        obj.pool()


    if actionMap["filter"] == 1:
        obj.pool_filter()




if __name__ == '__main__':
    main()
