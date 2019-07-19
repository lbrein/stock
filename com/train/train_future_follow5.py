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
class train_future_follow5(object):
    """

    """

    csvList = ['SC', 'AP', 'SP']

    csvKey = '%s_14_%s_2_2_3.5_0.5'

    def __init__(self):
        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.topUse = False
        self.isEmptyUse = True    # 是否清空记录

        self.testDays = 100
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

        self.iniAmount = 500000  # 单边50万

        # self.iniAmount = 2000000 * 0.0025  # 单边50万
        self.stopLine = 2
        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.klineList = [ '3m']

        # 起始时间
        self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测
        self.endDate = public.getDate(diff=1)
        print(self.endDate)
        self.total_tablename = 'train_total_1'
        self.detail_tablename = 'train_future_1'
        self.method = 'fellowX'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method
        self.isAll = 0

    def iterCond(self):
        # 多重组合参数输出
        #for i in range(1):
        #    yield '%s_%s' % (str(self.bullLine), str(self.atrLine))
        #"""

        keys = ['atrLine', 'stdcTick', 'startTick']
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                self.__setattr__(keys[1], s1)
                for s2 in self.__getattribute__(keys[2] + 'List'):
                    self.__setattr__(keys[2], s2)
                    yield '%s_%s_%s' % (str(s0), str(s1), str(s2))
        #"""

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
            for kline in self.klineList:
                # try:
                kline = None
                self.start(codes, time0, None, Base, Rice)
                #pool.apply_async(self.start, (codes, time0, kline, Base, Rice))
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
        dfs = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=None, pre=10)
        #print(dfs[mCodes[0]])
        print('kline load:', mCodes, [len(dfs[m]) for m in mCodes])

        # 根据配置文件获取最佳交易手数对
        self.iniVolume = round(self.iniAmount / cs[0]["lastPrice"] / cs[0]["contract_multiplier"], 0)
        if self.iniVolume == 0: self.iniVolume = 1

        # 分参数执行
        docs = self.total(dfs, period=self.period)
        if docs is None or len(docs) == 0: return

        logger.info((self.codes, self.klineType, len(docs), " time:", time.time() - time0))
        self.Total.insertAll(docs)

    pointColumns = ['powm', 'price', 'diss', 'fall', 'falla']

    def point(self, row, df0, tick):
        ma, close,open, high, low, std, stdc, atr, atrc,kdjm,dd = \
            (row[key] for key in
             "ma,close,open,high,low,std,stdc,atr,atrc,kdjm,datetime".split(","))

        #print(dd, df0.index[0],  df0.loc[dd])

        lastRow = df0.loc[dd].shift(-1)
        ma_1, close_1, high_1, low_1, std_1 = \
            (lastRow[key] for key in
             "ma,close,high,low,std".split(","))

        open = lastRow['close']

        tt = str(public.parseTime(str(dd), style='%H:%M:%S'))

        kline = int(self.klineType.replace('m', ''))

        ss = '0' + str(3 * kline) if 3 * kline < 10 else str( 3 *kline)
        BL, AL, ST = self.bullLine, self.atrLine, self.startTick
        SD = self.stdcTick

        if '09:00:00' <= tt <= ('09:%s:00' % ss) or '21:00:00' <= tt <= ('09:%s:00' % ss):
             BL += ST

        sign = 1 if high > (ma + 2 * std) else -1 if low < (ma - 2 * std) else 0

        max = high if sign > 0 else low if sign < 0 else close
        diss = 0 if std == 0 else abs(max - ma) / std

        # 前一个diss
        max_1 = high_1 if sign > 0 else low_1 if sign < 0 else close_1
        diss_1 = 0 if std_1 == 0 else abs(max_1 - ma_1) / std_1

        fall = abs(max - close) / abs(max - open) if max!=open else 0
        falla = abs(max - close)

        # 超跌
        opt0 = diss < BL - 0.25 and fall > 0.5
        opt1 = diss > BL and fall > 0.33

        # 超长
        opt2 = diss > (BL+2) and fall > 1/(diss)
        opt10 = stdc > SD and abs(max - close) > 2 * tick

        powm, price = 0, close
        if opt2:
           powm = 4 * sign
           price = (max - sign * abs(max - open) * fall)

        elif opt10:
           powm = sign if opt0 else sign * 2 if opt1  else  0
           price = (max - sign * abs(max - open) * fall)

        if powm == 0 and opt10 and diss_1 > (BL - 0.5):
            if diss > (BL - 1) and sign * (close - open) < 0:
                powm = sign * 6

        columns = self.pointColumns
        return pd.Series([powm, price, diss, fall, falla], index=columns)

    def curMA(self, row, df0):
        e = row['datetime']
        s = public.str_date(public.getDate(diff=-1,start=e) + ' 21:00:00')
        df = df0[(df0['datetime'] >= s) & (df0['datetime'] <=e)]

        return (df['close'] * df['volume']).sum()/df['volume'].sum() if df['volume'].sum()!=0 else 0

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

        b0 = self.BS[self.code]
        #rate, mar, mul = (b0[c] for c in ['ratio', 'margin_rate', 'contract_multiplier'])

        close = df0["close"]
        df0["datetime"] = df0.index

        begin = public.str_date(public.getDate(diff=-1) + ' 21:00:00')
        print(begin)

        df0["curMa"] = df0.apply(lambda row: self.curMA(row, df0, str(begin)), axis=1)

        df0["ma"] = ma = ta.MA(close, timeperiod=period)
        df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)

        df0["ma7"] = ma = ta.MA(close, timeperiod=period/2)
        df0["slope"] = ta.LINEARREG_SLOPE(ma, timeperiod=5)

        df0['tick_size'] = 10 * b0['tick_size']
        df0["stdc"] = std / ta.MA(std, timeperiod=period)

        df0['atr'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=period)
        df0['atrr'] = df0['atr'] / ma * 10000
        df0['atrc'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=1) / df0['atr']

        #df0['vol'] = self.iniAmount / (self.stopLine * df0['atr'] * mul)
        #df0['amount'] = df0['vol'] * close * mul * mar

        # kdj顶点
        kdjK, kdjD = ta.STOCH(df0["high"], df0["low"], close,
                              fastk_period=5, slowk_period=3, slowk_matype=1, slowd_period=3,
                              slowd_matype=1)

        df0["kdj_d2"] = kdj_d2 = kdjK - kdjD
        df0["kdjm"] = kdj_d2 * kdj_d2.shift(1)
        df0["kdjm"] = df0.apply(lambda row: self.turn(row['kdjm'], row['kdj_d2'], 1), axis=1)

        # 循环 scale
        docs = []
        for conds in self.iterCond():
            uid = self.uidKey % ('_'.join(self.codes), str(period),  self.klineType, str(self.bullwidth), conds)

            df1 = df0.apply(lambda row: self.point(row, df0, b0['tick_size']), axis=1)

            #if self.code =='SC': print(self.code, self.startTick)

            for key in self.pointColumns:  df0[key] = df1[key]
            print(uid)
            if self.code in self.csvList:
                    #and uid== (self.csvKey % ('_'.join(self.codes), str(period)) + self.method):
                file = self.Rice.basePath + '%s_pre.csv' % (uid)
                print(uid, '---------------------------- to_cvs', file, df0.columns)
                df0.to_csv(file, index=0)

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
        obj = train_future_follow5()
        obj.Pool()


if __name__ == '__main__':
    main()
