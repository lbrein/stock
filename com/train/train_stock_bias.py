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
from com.object.obj_entity import sh50_price_s, sh50_orderForm

# 选股
class train_etf50_kline(object):

    def __init__(self):
        self.Rice = interface_Rice()
        self.codes = "159901.XSHE,2159902.XSHE,2159915.XSHE,2510050.XSHG,2510900.XSHG, 2510300.XSHG,2519920.XSHG,\
                                                 2512160.XSHG,2512880.XSHG,2512800.XSHG,2510180.XSHG,2512660.XSHG".split(",")
        self.period = '30m'
        self.timePeriods = 60

        self.iniAmount = 250000
        self.ratio = 2

        self.startDate = public.getDate(diff=-130)  # 60天数据回测
        self.endDate = public.getDate(diff=0)

        self.Record = sh50_orderForm()
        self.Record.tablename = 'sh50_orderForm_train'
        self.Record.empty()
        self.Price = sh50_price_s()

    def pool(self):
        self.stag_etf(self.codes)

    def point(self, row):
        if np.isnan(row['ma60']): return 0

        close, vol, atr = row['close'], row['vol'], row['atr']
        p, j = 0, 1
        lists = self.timePeriodList

        # 上行标志
        for t in lists:
            ma, mam, mam2, mam3 = row['ma' + str(t)], row['mam' + str(t)], row['mam%s_2' % str(t)], row['mam%s_3' % str(t)]
            if (abs(mam2) ==2 and abs(mam3) == 1) or (mam2 == 0 and (vol > 2.0 or atr > 0.005)):
                p = int(j * np.sign(close - ma))

            j += 1
        return p

    def mam(self, row, p):
        if np.isnan(row['ma' + str(p)]): return 0
        return int(np.sign(row['close'] - row['ma' + str(p)]))

    # 分段布林策略
    def stag_etf(self, codes):
        self.codes = codes
        self.uid = uuid.uuid1()
        self.batchid = uuid.uuid1()

        res = self.Rice.kline(codes, period=self.period, start=self.startDate, end=self.endDate)
        df = res[codes[0]]
        print(df)
        # 计算统一特征
        df = self.add_stock_index(df, index_list=self.index_list)
        df['createTime'] = df.index
        df['mode'] = mode = df.apply(lambda row: self.point(row), axis=1)
        #df.iloc[[mode==mode.shift(1)], 'mode'] = 0

        file = self.Rice.basePath + '510050_XSHG_30m.csv'
        df.to_csv(file, index=1)

        self.saveStage(df)

    def add_stock_index(self, df, index_list=None):
        close = df["close"]
        if "MA" in index_list:
            for p in self.timePeriodList:
                df["ma" + str(p)] = ta.MA(close, timeperiod=p)
                df["mam" + str(p)] = mam = df.apply(lambda row: self.mam(row, p), axis=1)
                df["mam" + str(p)+"_2"] = ta.SUM(mam, timeperiod=2)
                df["mam" + str(p)+"_3"] = ta.SUM(mam, timeperiod=3)

        df['vol'] = df['volume'] / ta.MA(df['volume'], timeperiod=60)
        df['atr'] = ta.ATR(df['high'], df['low'], close, timeperiod=1)/close.shift(1)
        return df

    #
    def saveStage(self, df2):
        self.currentVol, self.currentPositionType, self.preNode = 0, 0, None

        period, ini, unit = self.timePeriods, self.iniAmount / self.multiplier, self.multiplier
        self.records = []

        for i in range(period, len(df2)):
            mode, close = (df2.ix[i, key] for key in "mode,close".split(","))
            preMode = df2.ix[i-1, 'mode']

            vol, pos = self.currentVol, self.currentPositionType
            isBuy, isRun = 0, False
            if mode == preMode: continue

            if vol > 0 and pos * mode < 0 and abs(mode) == 4:
                # 先平仓，再开仓
                self.order(df2.iloc[i], isBuy, mode, vol)
                # 再开仓
                isBuy, pos, isRun, vol = 1, mode, True, ini


            elif vol > 0 and pos * mode < 0 and abs(mode) < 4:
                # 部分减仓
                if vol >= (5 - abs(mode)):
                    isBuy, isRun, mode, vol = -1, True, mode, abs(mode) + vol - ini

            # 部分加仓
            elif ini > vol > 0 and pos * mode > 0 and abs(mode) < 4:
                # 部分减仓
                if vol < ini:
                     isBuy, isRun, mode, vol = 1, True, mode, abs(mode) if (ini-vol) > abs(mode) else ini-vol

            elif vol == 0 and abs(mode) == 4:
                isBuy, pos, isRun, vol = 1, mode, True, ini

            if isRun:
                # print(i, isBuy, pos, vol)
                self.order(df2.iloc[i], isBuy, mode, vol)

        # 保存明细
        if len(self.records) > 0:
            self.Record.insertAll(self.records)

    def order(self, n0, isBuy, mode, vol):
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

        if ETF is None:
            #print(pN['code'] if type == 0 else 'new', mode, now)
            ETF = {
                'code': 'none',
                'name': 'none',
                'power': n0['close'],
                'ask': 0,
                'bid': 0,
            }
            #return False

        price = ETF["ask"] if isBuy > 0 else ETF["bid"]
        fee = vol * self.ratio

        # income = -fee
        # if isBuy < 0:
        # income = (price - pN['price']) * vol * unit - fee

        # 修改状态
        self.currentPositionType = np.sign(mode) if isBuy > 0 else np.sign(-mode)
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
            # "income": income,
            "batchid": self.batchid,
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
        "stat": 0,
    }
    obj = train_etf50_kline()

    if actionMap["start"] == 1:
        obj.pool()

    if actionMap["test"] == 1:
        obj.getP()

    if actionMap["stat"] == 1:
        obj.stat()


if __name__ == '__main__':
    main()
