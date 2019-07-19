# -*- coding: utf-8 -*-
"""
Created on 12 19 -2018
@author: lbrein

      zhao 基金长线策略： bias 》 1 或bias 《 -1
      日线，每分钟持续跟踪

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
        self.pageCount = 50

        self.timePeriodsList = [150]
        self.timePeriods = 150

        self.dropLineList = [0.6]
        self.dropLine = 0.20

        self.sarStartList = [0.02, 0.35, 0.05]
        self.sarStart = 0.02
        self.sarEndList = [0.02, 0.35, 0.05]
        self.sarEnd = 0.05

        self.iniAmount = 250000
        self.shift = 0.002
        self.ratio = 0.0006

        self.saveMongo = False
        self.methodName = 'filter_5'

        self.startDate = public.getDate(diff=-300)  # 60天数据回测
        self.endDate = public.getDate(diff=0)
        self.iterCondList = ['timePeriods', 'dropLine', 'sarStart', 'sarEnd']

    def empty(self):
        Record = stock_orderForm()
        Record.tablename = 'stock_orderForm_train'
        Record.empty(filter=" method='%s'" % self.methodName)
        if self.saveMongo:
            TrainOrder = mon_trainOrder()
            TrainOrder.empty({'method': '%s' % self.methodName})

    def pool_filter(self):
        #Rice = interface_Rice()
        #Rice.int_stockBase()
        Base = stock_baseInfo()
        Base.iniBound()

        lists = Base.getCodes(isBound=0)
        pool = Pool(processes=6)

        for k in range(0, len(lists), self.pageCount):
            codes = lists[k: k + self.pageCount]
            pool.apply_async(self.subFilter, (codes, k))

        pool.close()
        pool.join()

    def subFilter(self, codes, k):
        Rice = interface_Rice()
        Base = stock_baseInfo()
        period = 180
        line = 0.35
        res = Rice.kline(codes, period=self.period, start=self.startDate, end=self.endDate, pre=90)

        codeList = []
        for code in codes:
            df = res[code]
            # 计算跌幅和回跌幅度
            close = df['close']
            mx = close[-period:].max()
            mi = close[-period:].min()
            miw = ta.MININDEX(close, timeperiod=period).values[-1]
            mid = close[miw:].max()

            # 超过M5
            ma5 = ta.MA(close, timeperiod=5)
            last = close.values[-1]

            opt1 = (mx-mi)/mx > line and (mid-mi)/(mx-mi) < 0.372
            opt2 = (last > ma5.values[-1] or last > ma5.values[-2])
            if opt1 and opt2:
                codeList.append(code)
                #print(code, (mx-mi)/mx, (mid-mi)/(mx-mi))

        print(k, len(codeList))
        Base.updateBound(codeList)


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


    # 分段布林策略
    def start(self, codes, n):
        time0 = time.time()
        print('process %s start:' % str(n))
        self.Rice = interface_Rice()
        self.Record = stock_orderForm()
        self.Record.tablename = 'stock_orderForm_train'
        self.TrainOrder = mon_trainOrder()

        self.codes = codes
        res = self.Rice.kline(codes, period=self.period, start=self.startDate, end=self.endDate,pre=90)

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

                    if code.find('603383') > -1 or code.find('601139') > -1:
                        file = self.Rice.basePath + '%s.csv' % self.uid
                        print(file)
                        df.to_csv(file, index=1)

                    self.saveStage(df)

        print('process %s end: %s ' % (str(n),str(time.time()-time0)))

    def point(self, row):
        mx, mi, mid, sm, sm5 = (row[key] for key in 'max90,min90,mid,sarm,sarm5'.split(','))
        line = self.dropLine

        if not ((mx-mi)/mx > line and (mid-mi)/(mx-mi) < 0.372):
            return 0

        return 1 if sm == 1 else -1 if sm5==-1 else 0

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    def mid(self, row, close):
        c1 = close[close.index <= row['createTime']][row['miw']:]
        return c1.max()

    def add_stock_index(self, df0, index_list=None):

        close = df0["close"]

        df0['max90'] = mx = ta.MAX(close, timeperiod = self.timePeriods)
        df0['min90'] = min = ta.MIN(close, timeperiod= self.timePeriods)
        df0['miw'] = ta.MININDEX(close, timeperiod=self.timePeriods)
        df0['mid'] = df0.apply(lambda row: self.mid(row,  close), axis=1)

        # 穿越
        sar = ta.SAR(df0['high'], df0['low'], acceleration=self.sarStart, maximum=0.2)
        df0['sard'] = sard = close - sar
        df0['sarm'] = sard * sard.shift(1)
        df0['sarm'] = df0.apply(lambda row: self.turn(row['sarm'], row['sard'], 1), axis=1)

        sar5 = ta.SAR(df0['high'], df0['low'], acceleration=self.sarEnd, maximum=0.2)
        df0['sard5'] = sard5 = close - sar5
        df0['sarm5'] = sard5 * sard5.shift(1)
        df0['sarm5'] = df0.apply(lambda row: self.turn(row['sarm5'], row['sard5'], 1), axis=1)

        return df0

    #
    def saveStage(self, df2):
        self.preNode =  None
        period, ini = 60, self.iniAmount
        self.mon_records,  self.records = [], []

        for i in range(period, len(df2)):
            mode, close = (df2.ix[i, key] for key in "mode,close".split(","))

            isBuy, isRun = -1, False
            pN = self.preNode
            # 部分加仓
            if pN is None and mode ==1 :
                isBuy, isRun, mode = 1, True, mode

            elif pN is not None and mode==-1:
                isBuy, isRun, mode = -1, True, mode

            if isRun:
                # print(i, isBuy, pos, vol)
                self.order(df2.iloc[i], isBuy, mode)

        #print(self.uid, len(self.records))
        # 保存明细
        if len(self.records) > 0:
           self.Record.insertAll(self.records)

           if self.saveMongo and len(self.mon_records) > 0:
               #print("monsave", self.uid, self.mon_records)
               self.TrainOrder.col.insert_many(self.mon_records)

    def mon_saveTick(self, n0, doc):
        tick = copy.deepcopy(n0.to_dict())
        tick.update(doc)
        for key in ['sarm', 'sarm5', 'miw', 'mode']:
            if key in tick: tick[key] = int(tick[key])

        if doc['isBuy'] == -1:
            self.mon_records.append(tick)
            if self.preTick is not None:
                self.preTick['income'] = doc['income']
                self.preTick['enddate'] = doc['createdate']
                #self.preTick['isBuy'] = doc['isstop']
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
    }
    obj = train_etf50_kline()

    if actionMap["start"] == 1:
        obj.pool()


    if actionMap["filter"] == 1:
        obj.pool_filter()


if __name__ == '__main__':
    main()
