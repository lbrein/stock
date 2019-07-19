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
import tushare as ts
from com.data.interface_Rice import interface_Rice, rq
from com.object.obj_entity import stock_orderForm, stock_baseInfo, stock_record_t0
from com.object.mon_entity import mon_trainOrder
from multiprocessing import Pool, Manager
import time
import copy

# 选股
class train_stock_tick(object):

    csvList = ['002907.XSHE', '000606.XSHE', '002137.XSHE', '600809.XSHG']
    def __init__(self):
        self.period = '1m'
        self.pageCount = 5
        self.iniAmount = 250000
        self.ratio = 0.00015

        self.bigLineList = [1.5]
        self.volcLineList = [10]

        self.volcLine = 6
        self.bigLine = 1.5
        self.stopLine = 0.2

        self.timePeriodsList= [14]
        self.timePeriods = 14

        self.saveMongo = True
        self.methodName = 'tick1'
        self.recordTableName = 'stock_orderForm_1'
        #self.startDate = public.getDate(diff=-200)  # 60天数据回测
        #self.endDate = public.getDate(diff=0)

        self.startDate = '2019-02-11'
        self.endDate = '2019-06-12'

        self.iterCondList = ['volcLine', 'bigLine']
        self.uidKey = "%s_%s_%s_" + self.methodName

    def empty(self):
        Record = stock_orderForm()
        Record.tablename = self.recordTableName
        Record.empty(filter=" method='%s'" % self.methodName)
        if self.saveMongo:
            TrainOrder = mon_trainOrder()
            TrainOrder.empty({'method': '%s' % self.methodName})

    def pool(self):
        pool = Pool(processes=5)
        self.empty()
        T0 = stock_record_t0()
        lists = T0.getCodes()
        print(len(lists))

        for k in range(0, len(lists), self.pageCount):
            codes =lists[k: k+self.pageCount]
            In = False
            for c in codes:
                if c in self.csvList:
                    In = True
                    break
            #if not In : continue

            print(codes)
            #self.start(codes, int(k/self.pageCount+1))
            try:
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
                yield '%s_%s' % (str(s0), str(s1))

                """
                for s2 in self.__getattribute__(keys[2] + 'List'):
                    self.__setattr__(keys[2], s2)

                    for s3 in self.__getattribute__(keys[3] + 'List'):
                        self.__setattr__(keys[3], s3)

                        yield '%s_%s_%s_%s' % (str(s0), str(s1), str(s2), str(s3))
        """

    # 分段布林策略
    def start(self, codes, n):
        time0 = time.time()
        print('process %s start:' % str(n))

        self.Rice = interface_Rice()
        self.Record = stock_orderForm()

        self.Record.tablename = self.recordTableName
        self.TrainOrder = mon_trainOrder()

        res = self.Rice.kline(codes, period=self.period, start=self.startDate, end=self.endDate, pre=1, type='stock')
        self.klineColumns = res[codes[0]].columns

        Ticks = self.Rice.kline(codes, period='tick', start=self.startDate, end=self.endDate, pre=2, type='stock')
        #Ticks = self.Rice.kline(['SC1907','RB1907'], period='tick', start=self.startDate, end=self.endDate, pre=2, type='future')
        #print(Ticks)

        for code in codes:
            self.code = code
            df = res[code]
            tk = Ticks[code]
            tk['datetime'] = tk.index

            # 计算统一特征
            df['datetime'] = df.index
            #df['code'] = code

            for conds in self.iterCond():
                self.uid = self.uidKey % (self.code, str(self.timePeriods), conds)
                df, tk = self.total(df, tk)
                self.saveStage(df, tk)

        print('process %s end: %s ' % (str(n), str(time.time()-time0)))

    def curMA(self, row, df0):
        e = row['datetime']
        #print(str(e)[:10])
        s = public.str_date(public.getDate(diff=-1, start=str(e)[:10]) + ' 09:30:00')
        df = df0[(df0['datetime'] >= s) & (df0['datetime'] <=e)]

        return (df['close'] * df['volume']).sum() / df['volume'].sum() if df['volume'].sum() != 0 else 0

    def getTick(self, code, d):
        c = code[:6]
        return ts.get_tick_data(c, date=d, src='tt')

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    def tickMode(self, tick):
        if np.isnan(tick['raise']): return 0
        mode = 0
        if tick['raise'] != 0:
            mode = int(np.sign(tick['raise']))
        elif tick['a1'] != tick['pa1']:
            mode = 1 if tick['last'] == tick['pa1'] else -1 if tick['last'] == tick['pb1'] else 0
        else:
            mode = 1 if tick['last'] == tick['a1'] else -1 if tick['last'] == tick['b1'] else 0
        return mode

    def total(self, df0, tk, period=14):
        # 计算参数
        close = df0["close"]
        df0["datetime"] = df0.index
        df0["curMa"] = df0.apply(lambda row: self.curMA(row, df0), axis=1)

        df0["ma"] = ma = ta.MA(close, timeperiod=period)
        df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)
        df0['mv'] = ta.MA(df0['volume'], timeperiod=period)
        df0['volc120'] = df0['volume']/ta.MA(df0['volume'], timeperiod=120)
        df0['volc'] = df0['volume'] / df0['mv']
        df0['bias'] = (close - ma)/ma
        df0['min5'], df0['max5'] = ta.MINMAX(close, timeperiod=10)

        # df1 = df0.apply(lambda row: self.point(row, df0, b0['tick_size']), axis=1)
        # for key in self.pointColumns:  df0[key] = df1[key]

        # tick
        tk['vol'] = tk['volume'] - tk['volume'].shift(1)
        tk = tk[tk['vol']>0]

        #print(tk)
        tk['raise'] = tk['last'] - tk['last'].shift(1)
        tk['rdiff'] = tk['a1'] - tk['b1']

        for k in ['a', 'b']:
            tk['r' + k] = tk[k + '1'] - tk[k + '1'].shift(1)
            #tk['p%s1'% k] = tk[k + '1'].shift(1)

            rrr = tk['r' + k].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            for n in [3]:
                key = 'r%s%s' % (k, str(n))
                tk[key+'_v'] = ta.SUM(tk['r'+k], timeperiod=n)
                tk[key] = ta.SUM(rrr, timeperiod=n)

        #print(tk.iloc[-1])

        tk['mode'] = tk.apply(lambda row: self.tickMode(row), axis=1)
        tk['datetime'] = tk['datetime'].apply(lambda x: str(x)[:str(x).find(".")] if str(x).find(".") > -1 else x)

        #tk['dd5'] = tk['datetime'].shift(5).fillna('2019-01-01 00:00:00')
        #tk['diff'] = tk.apply(lambda row: public.timeDiff(str(row['datetime']), str(row['dd5'])), axis=1)
        tk['modem'] = ta.SUM(tk['mode'], timeperiod=5)
        tk['vol_t'] = ta.SUM(tk['vol'] * tk['mode'], timeperiod=5)

        tk['modem3'] = ta.SUM(tk['mode'], timeperiod=3)
        tk['vol_t3'] = ta.SUM(tk['vol'] * tk['mode'], timeperiod=3)

        print(self.code, self.csvList)

        if self.code in self.csvList:
            # and uid== (self.csvKey % ('_'.join(self.codes), str(period)) + self.method):
            file = self.Rice.basePath + '%s1_kline.csv' % (self.uid)
            file1 = self.Rice.basePath + '%s1_tick.csv' % (self.uid)
            print(self.uid, '---------------------------- to_cvs', file)

            df0.to_csv(file, index=0)
            columns = ['datetime', 'last', 'high', 'low',  'volume',
                        'a1',  'b1', 'a1_v', 'b1_v', 'change_rate', 'vol', 'ra', 'rb', 'ra2', 'ra3', 'rb2', 'rb3',
                       'raise', 'mode', 'modem', 'vol_t', 'modem3', 'vol_t3', 'diff']
            tk.to_csv(file1, index=0, columns=columns)

        return df0[self.startDate:], tk
    #
    def point(self, row):
        #print(row)
        mp, close, ma, vol, mv, mode, rs, dd, diff, rdiff, modem, min5, max5, vol_t, modem3, vol_t3 = \
            (row[key] for key in
             "curMa,close,ma,vol,mv,mode,raise,datetime,diff,rdiff,modem,min5,max5,vol_t,modem3,vol_t3".split(","))

        tt = public.parseTime(str(dd), style='%H:%M:%S')
        BL, VL = self.bigLine, self.volcLine
        if '09:30:00' <= tt < '09:35:00':
            BL = 3 * BL
            VL += 10

        # 开启条件
        #opt0 = mode * (close - mp) > 0 and volume_c > BL * mv and False
        sign = np.sign(close-mp)
        max = max5 if sign> 0 else min5
        opt0 = modem * (close - mp) > 0 and abs(modem) == 3 and abs(vol_t3) > 0.5 * mv and abs(diff) <= 15 and close * sign > max and False
        opt1 = modem * (close - mp) > 0 and abs(modem) == 5 and \
               3 * mv > abs(vol_t) >= 0.7 * mv and abs(diff)<=15 and vol < 0.8 * mv and rdiff < 0.04

        # 乖离度
        BIAS = (close - ma) / ma * 1000
        powm = 2 * sign if opt1 else sign if opt0 else 0

        return powm, BIAS

    def stop(self, row, sign):
        mp, close, a1, b1, ra2, rb2, dd, mv,modem, vol_t, diff = \
            (row[key] for key in
             "curMa,close,a1,b1,ra2,rb2,datetime,mv,modem,vol_t,diff".split(","))

        rr = ra2 if sign < 0 else rb2
        #rsa, rsb = sum([np.sign(r) for r in ra[-2:]]), sum([np.sign(r) for r in rb[-2:]])
        #print(dd, sign, ra, rb,  fall)

        tt = public.parseTime(str(dd), style='%H:%M:%S')
        opt0 = - sign * modem > 2 and - sign * vol_t * 2 > mv
        opt1 = ra2 * sign == -2 or rb2 * sign == -2
        opt2 = '14:59:40' <= tt < '14:59:50'
        #opt2 = fall > 0.20 and abs(a1-b1) / close < 0.003
        stop = -2 * sign if opt1 else - sign if opt0 else -4 * sign if opt2 else 0

        return stop, 0

    def getMax(self, df0, s, e, mode):
        if mode > 0:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'high'].max()
        else:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'low'].min()

    def saveStage(self, df2, tk):

        self.preNode, isOpen =  None, 0
        period, ini = 60, self.iniAmount
        self.mon_records,  self.records = [], []

        for i in range(period, len(df2)):
            close, volc, date = (df2.ix[i, key] for key in "close,volc,datetime".split(","))

            if isOpen == 0 and volc < 1.5:
                continue

            row = df2.iloc[i].to_dict()
            for tick in self.Rice.tickSim(tk, date):
                row.update(tick.to_dict())
                isRun = False

                # 部分加仓
                if isOpen == 0:
                    row['powm'], row['BIAS'] = powm, BIAS = self.point(row)
                    if powm != 0:
                        isOpen, isRun, mode = 1, True, powm

                elif isOpen == 1:
                    pN = self.preNode
                    row['stop'], row['fall'] = stop, fall = self.stop(row, np.sign(pN['mode']))
                    if pN['mode'] * stop < 0:
                        isOpen, isRun, mode = 0, True, stop

                    #elif pN['mode'] * kdjm < 0:
                        #isOpen, isRun, mode = 0, True, int(-np.sign(pN['mode']) * 2)

                if isRun:
                    self.order(row, isOpen, mode)

        #print(self.uid, len(self.records))
        # 保存明细
        if len(self.records) > 0:
           print('saved', self.uid, len(self.records), len(self.mon_records))
           self.Record.insertAll(self.records)

           #if self.saveMongo and len(self.mon_records) > 0:
               #self.TrainOrder.col.insert_many(self.mon_records)

    def mon_saveTick(self, n0, doc):
        tick = copy.deepcopy(n0)
        tick.update(doc)
        tick['datetime'] = str(tick['datetime'])
        tick['trading_date'] = str(tick['trading_date'])

        for key in ['kdjm', 'powm', 'mode', 'isBuy','volume_mode', 'stop']:
            if key in tick: tick[key] = int(tick[key])
        #print(tick)
        self.TrainOrder.col.insert_one(tick)
        #self.mon_records.append(tick)

    def mon_getTick(self, code):
        columns ="datetime,code,close,open,high,low,volume_m,volume_c,volume_mode,raise,raise_a,raise_b,curMa,mv,volc,volcm,income,mode,price,vol,fall".split(",")
        Mon, Rice = mon_trainOrder(), interface_Rice()
        df = Mon.getTick('tick', code=code, columns=columns)
        file = Rice.basePath + '%s_order.csv' % (code)
        df.to_csv(file, index=0)

    def order(self, n0, isOpen, mode):
        pN = self.preNode
        #now = public.getDatetime()
        vol, fee, amount,income, p0 = 0, 0, 0, 0, 0
        price = n0["close"]

        if 'a1' in n0:
            price = n0['a1'] if mode > 0 else n0['b1']
        
        # 费率
        if mode > 0:
            ratio = 0.00015 + (0.00002 if self.code[0] == '6' else 0.0)
        else:
            ratio = 0.00015 + 0.001 + (0.00002 if self.code[0] == '6' else 0.0)
        
        if isOpen == 1:
            self.batchid = uuid.uuid1()
            p0 = price
            vol = int(self.iniAmount/p0/100) * 100
            amount = vol * p0
            fee = amount * ratio
            income = -fee

        elif isOpen == 0 and pN is not None:
            #print('-------------', self.code, n0['datetime'],  pN)
            p0 = price
            vol = pN['vol']
            amount = vol * p0
            fee = amount * ratio
            income = np.sign(pN['mode']) * (amount - pN['amount']) - fee

        d = str(n0['datetime'])
        d = d[:d.find('.')] if d.find('.')> -1  else d
        doc = {
            "code": self.code,
            "name": self.code,
            "createdate": d,
            "price": p0,
            "vol": vol,
            "mode": int(mode),
            "isBuy": int(isOpen),
            "fee": fee,
            "pow": n0['vol_t'],
            "interval": int(n0['diff']),
            "width": n0['bias'],
            "last": n0['vol'],
            "ini_price": n0['volc'],
            "ini_vol": n0['mv'],
            "diss_a": n0['rdiff'],
            "vol_120": n0['volc120'],

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
        if isOpen > 0:
            self.preNode = doc
        else:
            self.preNode = None

        return True


def main():
    actionMap = {
        "start": 1,  #
        "mon": 1
    }
    obj = train_stock_tick()

    if actionMap["start"] == 1:
        obj.pool()

    if actionMap["mon"] == 1:
        obj.mon_getTick('002907.XSHE')

if __name__ == '__main__':
    main()
