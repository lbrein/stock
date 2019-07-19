# -*- coding: utf-8 -*-
"""
Created on 1-16 -2018 
@author: lbrein
   
 ---------- -- 计算并设置 币币组合的比价路径 --------------- 
"""

from com.base.public import public, logger
import numpy as np
import pandas as pd
import talib as ta
import uuid
import math
from com.data.interface_Rice import interface_Rice
from com.object.obj_entity import sh50_price_s, sh50_orderForm
from multiprocessing import Pool, Manager
import math


# 选股
class train_etf50_kline(object):

    def __init__(self):

        self.codes = ['510050.XSHG']

        self.klineType = '30m'
        self.klineTypeList = ['15m', '30m']
        self.timePeriodsList = [15, 30, 60]
        self.timePeriods = 60
        self.powlineList = [0.5]
        self.powline = 1.0
        self.turnlineList = [1.5, 1.75, 2.0]
        self.turnline = 1.0
        self.paramList = ['powline', 'turnline']
        self.scaleDiff2 = 0.8

        self.iniAmount = 40000
        self.multiplier = 10000
        self.ratio = 2
        self.scale = 2.0
        self.shift = 0.0005

        self.index_list = ["MA"]
        self.startDate = public.getDate(diff=-360)  # 60天数据回测
        self.endDate = public.getDate(diff=0)
        self.method = 'bull4'

    def empty(self):
        Record = sh50_orderForm()
        Record.tablename = 'sh50_orderForm_train'
        # Record.empty()

    def iterCond(self):
        # 多重组合参数输出
        keys = self.paramList
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                self.__setattr__(keys[1], s1)

                # for s2 in self.__getattribute__(keys[2] + 'List'):
                #    self.__setattr__(keys[2], s2)
                yield '%s_%s' % (str(s0), str(s1))

    def pool(self):
        pool = Pool(processes=5)
        self.empty()

        for kt in self.klineTypeList:
            for tms in self.timePeriodsList:
                #self.start(self.codes, kt, tms)
                try:
                    pool.apply_async(self.start, (self.codes, kt, tms))
                    pass
                except Exception as e:
                    print(e)
                continue
        pool.close()
        pool.join()


    # 分段布林策略
    def start(self, codes, kline, tms):
        print((public.getDatetime(), ':  start ', codes, kline, tms))
        self.Rice = interface_Rice()
        self.Record = sh50_orderForm()
        self.Record.tablename = 'sh50_orderForm_train'

        self.codes = codes
        self.Price = sh50_price_s()
        self.klineType = kline
        self.timePeriods = tms

        res = self.Rice.kline(codes, period=kline, start=self.startDate, end=self.endDate)
        df = res[codes[0]]
        # 计算统一特征
        for conds in self.iterCond():

            self.uid = "%s_%s_%s_%s" % (codes[0].replace('.', '-'), tms, kline[:-1], conds)

            df = self.add_stock_index(df, index_list=self.index_list)
            df['createTime'] = df.index

            saveList = ['30_5_0.25_1.75', '15_15_0.25_1.75', '15_30_0.25_1.75', '30_60_0.25_1.75']
            if self.uid[12:] in saveList:
                cs = []
                bans = 'ma,bullwidth,open,sard,rsi,high,low,std,top,lower,p_h,p_l,,close,volume,wmean,width,kdj_d2,createTime'.split(
                    ',')
                for c in df.columns:
                    if c not in bans:
                        cs.append(c)

                file = self.Rice.basePath + '%s.csv' % self.uid
                df.to_csv(file, index=1, columns=cs)

            # print(('start ', self.uid))
            self.saveStage(df)

    def sd(self, x):
        s = round(x * 1.5, 0) / 10
        return np.sign(s) * 0.8 - 0.1 if abs(s) > 0.8 else s - 0.1

    def stand(self, ser):
        return ser / ser.abs().mean()

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    pointColumns = ['pow', 'powm', 'trend', 'isopen']

    def point(self, row, ktype=15):
        close, wd1, s1, s2, atr, atr1, high, low, open, close, top, lower, p_h, p_l, tu, td = (row[key] for key in
                                                                                               "close,widthDelta,slope,slope30,atr,atr1,high,low,open,close,top,lower,p_h,p_l,tu,td".split(
                                                                                                   ","))
        if close > open:
            trend = 1 if high == open else ((close - open) / (high - open) + 0.5)
        else:
            trend = -1 if low == open else ((close - open) / (open - low) - 0.5)

        tr = (4 - atr * atr1) if atr > 2 else atr * atr1
        if tr < 0: tr = 0
        # pow
        pow = (wd1 + abs(s1 + s2 + trend * tr)) / 4

        powm = 2 * np.sign(s1) if (pow > self.turnline and (close > tu or close < td)) else np.sign(s1) if (
                pow > 0.25 or abs(s1) > 0.5) else 0

        isopen = -1 if p_h > top else 1 if p_l < lower else 0

        columns = self.pointColumns

        return pd.Series([pow, powm, trend, isopen], index=columns)


    def add_stock_index(self, df0, index_list=None):
        ktype, period, scale = self.klineType[:-1], self.timePeriods, self.scale
        close = df0["close"]
        df0["ma"] = ma = ta.MA(close, timeperiod=period)
        df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)
        # 上下柜
        # bullWidth
        df0["bullwidth"] = width = (4 * std / ma * 100).fillna(0)

        # 近三分钟width变动
        df0["widthDelta"] = self.stand(ta.LINEARREG_SLOPE(width, timeperiod=3))
        df0["slope"] = self.stand(ta.LINEARREG_SLOPE(ta.MA(close, timeperiod=3), timeperiod=2))
        df0["slope30"] = self.stand(ta.LINEARREG_SLOPE(ta.MA(close, timeperiod= 60 if period >= 15 else 30), timeperiod=3))
        df0['sd'] = sd = df0['slope30'].apply(lambda x: self.sd(x))


        df0['atrb'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=period)
        df0['atr'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=1) / ta.ATR(df0['high'], df0['low'], close,
                                                                                   timeperiod=period)
        df0['atr1'] = self.stand(ta.ATR(df0['high'], df0['low'], close, timeperiod=period))

        # 唐奇安线
        df0['tu'] = ta.MAX(df0['high'].shift(1), timeperiod=20)
        df0['td'] = ta.MIN(df0['low'].shift(1), timeperiod=20)

        # SAR 顶点
        sar = ta.SAR(df0["high"], df0["low"], acceleration=0.04, maximum=0.25)
        df0['sard'] = sard = close - sar
        df0['sarm'] = sard * sard.shift(1)
        df0['sarm'] = df0.apply(lambda row: self.turn(row['sarm'], row['sard'], 1), axis=1)

        # kdj顶点
        kdjK, kdjD = ta.STOCH(df0["high"], df0["low"], close,
                              fastk_period=4, slowk_period=3, slowk_matype=2, slowd_period=3,
                              slowd_matype=2)

        df0["kdj_d2"] = kdj_d2 = kdjK - kdjD
        df0["kdjm"] = kdj_d2 * kdj_d2.shift(1)
        df0["kdjm"] = df0.apply(lambda row: self.turn(row['kdjm'], row['kdj_d2'], 1), axis=1)

        df0["top"], df0["lower"] = ma + (scale - sd) * std, ma - (scale + sd) * std
        df0["p_l"] = close * (1 + self.shift)
        df0["p_h"] = close * (1 - self.shift)

        df1 = df0.apply(lambda row: self.point(row, ktype), axis=1)
        for key in self.pointColumns: df0[key] = df1[key]
        return df0

    #
    def saveStage(self, df2):
        self.currentVol, self.currentPositionType, self.preNode = 0, 0, None

        period, ini, unit = self.timePeriods, self.iniAmount / self.multiplier, self.multiplier
        self.records, status = [], 0

        self.batchid = ""
        for i in range(period, len(df2)):
            ma, close, p_l, p_h, top, lower, std, kdjm, sarm, pow, powm, atrb,sd,s1 = (
                df2.ix[i, key] for key in
                "ma,close,p_l,p_h,top,lower,std,kdjm,sarm,pow,powm,atrb,sd,slope".split(","))

            vol, pos = self.currentVol, self.currentPositionType
            mode, isBuy, isRun = 0, 0, False

            if self.preNode is not None:  mode = self.preNode['mode']

            sd2 = (self.scaleDiff2 - 1.2 * sd) * std / 2
            pd2 = (self.turnline - 0.25)
            if status == 0 and powm in [2, -2]:
                """
                   突变点处理 
                   将布林带策略切换为策略 
                """
                status = np.sign(powm)
                if vol > 0 and pos * status < 0:
                    if self.order(df2.iloc[i], -1, 4, ini, status):
                        isBuy, mode, isRun = 1, int(4 * status), True

                elif vol == 0:
                    isBuy, mode, isRun = 1, int(3 * status), True

            elif sarm * status < 0 or (abs(mode) in [3] and mode * sarm < 0):
                """
                    交叉点: 快慢线穿越处理
                    1、结束突变状态，变为布林带处理
                    2、根据节点状况开平新仓，状态为6 
                """
                # 结束macd状态
                if vol > 0:
                    pi,po = self.preNode['price'], self.preNode['pos']
                    if self.order(df2.iloc[i], -1, 3, ini, status):
                        isRun = False
                        # 平仓后再开
                        if pi!=0  and (close - pi) * po / pi > 0.015:
                             isBuy, mode, isRun = 1, -po, True

                status = 0

            elif status == 0:
                """
                    BULIN策略
                """
                if vol == 0:
                    # 突变状态开始
                    # 大于上线轨迹
                    if p_h >= top and powm == 0:
                        mode, isRun = -1, True

                    elif p_l <= lower and powm == 0:
                        mode, isRun = 1, True

                    elif (p_h + sd2) >= top and pow < pd2 and kdjm < 0:
                        mode, isRun = -2, True

                    elif (p_l - sd2) <= lower and pow < pd2 and kdjm > 0:
                        mode, isRun = 2, True

                    isBuy = 1
                    # 平仓

                else:
                    sign = pos
                    cond3 = sign * (close - ma)
                    #
                    if cond3 >= -sd2 and sign * kdjm < 0:
                        if ((p_h + sd2) >= top or (p_l - sd2) <= lower) and pow < pd2:
                            if self.order(df2.iloc[i], -1, 1, ini, status):
                                isBuy, mode, isRun = 1, 1, True
                        else:
                            mode, isRun = 2, True
                            isBuy = -1

                    elif cond3 >= 0 and pow < 0.25 and abs(s1)< 0.75:
                        mode, isRun = 0, True
                        isBuy = -1

                    else:
                        # 即时收益超出 atr 2倍
                        if self.preNode is not None:
                            pd = self.preNode['price']
                            if (sign * (pd - close) > 2 * atrb) and sign * kdjm < 0:
                                mode, isRun = 7, True
                                isBuy = -1

            if isRun:
                #print(self.uid, i, mode, isBuy)
                self.order(df2.iloc[i], isBuy, mode, ini, int(status))

        # 保存明细
        if len(self.records) > 0:
            self.Record.insertAll(self.records)

    def order(self, n0, isBuy, mode, vol, status=0):
        pN = self.preNode
        cv, uid, unit = self.currentVol, self.uid, self.multiplier
        ETF, aPrice = None, -1
        now = n0['createTime']

        type = 0

        # 查询匹配的期权当前价格
        if isBuy < 0:
            # 卖出
            ETF = self.Price.getETF(currentTime=now, code=pN['code'])
        else:
            # 新开仓
            type = 1
            ETF = self.Price.getETF(sign=np.sign(mode), price=n0['close'], currentTime=now)
            self.batchid = uuid.uuid1()

        if ETF is None:
            # print(pN['code'] if type == 0 else 'new', mode, now)
            ETF = {
                'code': 'none',
                'name': 'none',
                'ask': 0.05,
                'bid': 0.05,

            }
            # return False

        price = ETF["ask"] if isBuy > 0 else ETF["bid"]
        fee = vol * self.ratio

        #income = -isBuy * price * vol * unit + pN['amount'] if isBuy < 0 else -fee

        # 修改状态
        self.currentPositionType = np.sign(mode) if isBuy > 0 else pN['pos']

        income = 0
        if isBuy < 0:
            income = -isBuy * n0['close'] * self.currentPositionType + pN['ownerPrice']

        self.currentVol += isBuy * vol

        doc = {
            "code": ETF['code'],
            "name": ETF['name'],
            "createdate": now,
            "price": price,
            "vol": vol,
            "mode": mode,
            "isBuy": isBuy,
            "pos": self.currentPositionType,
            "ownerPrice": -isBuy * n0['close'] * self.currentPositionType,
            "fee": fee,
            "amount": -isBuy * price * vol * unit - fee,
            "pow": n0['pow'],
            "width": n0['bullwidth'],
            "income": income,
            "status": int(status),
            "batchid": self.batchid,
            "method": self.method,
            "uid": uid
        }

        self.records.append(doc)

        # 设置上一个记录
        if self.currentVol > 0:
            self.preNode = doc
        else:
            self.preNode = None

        return True

    def getP(self):
        codes = self.Rice.allCodes('ETF')
        print(codes)
        for code in codes['order_book_id'].values:
            if code.find('5100') > -1:
                print(code)


def main():
    actionMap = {
        "start": 1,  #
        "test": 0,
    }
    obj = train_etf50_kline()

    if actionMap["start"] == 1:
        obj.pool()

    if actionMap["test"] == 1:
        obj.getP()


if __name__ == '__main__':
    main()
