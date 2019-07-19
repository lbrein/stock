# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ---  与tick 配套对比计算的 kline
    > 1m的 K线 模拟tick预测
    > 定期执行，用于筛选交易对
"""

from com.base.public import public, logger
from com.object.obj_entity import future_baseInfo
import talib as ta
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, Manager
from com.train.train_future_singleExpect import train_future_singleExpect
from com.base.stat_fun import fisher
import math


# 回归方法
class train_future_macd(train_future_singleExpect):
    """

    """
    iniAmount = 250000  # 单边50万
    csvList = [
        "14_2_15_3_2.0",
    ]

    def __init__(self):
        # 费率和滑点
        super().__init__()

        self.saveDetail = True  # 是否保存明细
        self.savePoint = False
        self.isSimTickUse = False  # 是否使用1分钟模拟tick测试，否则直接使用kline回测
        self.topUse = False
        self.isEmptyUse = True
        self.baseInfo = {}
        #self.codeLists = ['JM', 'SM', 'V', 'I', 'AP', 'J', 'RB', 'SC', 'MA', 'JD', 'CU', 'OI', 'AU']
        self.codeLists = ['SC', 'RB', 'FU', 'MA']

        # 可变参数
        self.periodList = [14]  # 窗体参数
        self.scaleList = [2]
        self.klineTypeList = ['15m']

        self.widthTimesPeriodList = [3]

        self.superlineList = [2.0]

        self.powline = 0.25
        self.turnlineList = [2]
        self.wdslineList = []
        self.turnline = 1.5
        self.paramList = ['superline']

        self.shiftScale = 0.527  # 滑点模拟系数
        self.scaleDiff2 = 0.8
        self.scaleDiff = 0
        self.processCount = 6

        self.testDays = 90
        self.adjustDates = []
        # testCOun

        self.testCount = 300
        self.testDays = 100

        # 起始时间
        self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测
        self.endDate = public.getDate(diff=0)

        self.total_tablename = 'train_total_1'
        self.detail_tablename = 'train_future_1'

        self.method = 'simTick' if self.isSimTickUse else 'quick'
        self.stage = 'jump'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method + "_" + self.stage

        # 切换全部计算
        self.isAll = 0

    def iterCond(self):
        # 多重组合参数输出
        #yield ''
        keys = self.paramList
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)
            yield '%s' % (str(s0))
            #for s1 in self.__getattribute__(keys[1] + 'List'):
                #self.__setattr__(keys[1], s1)

                # for s2 in self.__getattribute__(keys[2] + 'List'):
                #    self.__setattr__(keys[2], s2)
                #yield '%s_%s' % (str(s0), str(s1))

    def Pool(self):
        time0 = time.time()

        pool = Pool(processes=self.processCount)
        shareDict = Manager().list([])

        Base = future_baseInfo()
        # 交易量大的，按价格排序, 类型:turple,第二位为夜盘收盘时间
        lists = Base.all(vol=100)
        tops = self.tops()
        # 清空数据库
        self.switch()

        for rs in lists:
            # 检查时间匹配
            codes = [rs[0]]
            if self.topUse and codes not in tops: continue

            if self.isAll == 0 and codes[0] not in self.codeLists:  continue
            print(rs)

            for kt in self.klineTypeList:
                self.start(codes, time0, kt, shareDict)
                try:
                    #pool.apply_async(self.start, (codes, time0, kt, shareDict))
                    pass
                except Exception as e:
                    print(e)
                    continue
        pool.close()
        pool.join()

    pointColumns = ['mam', 'super', 'powm', 'yd', 'trend', 'datr']
    def point(self, row, ktype=15, period=15):
        wds, slope, open, close, high, low, top, lower, ma, ma5,std, mas,mam5,s5,kdj_d2,kdjm,interval,atr,maxp= \
            (row[key] for key in
             "wds,slope,open,close,high,low,top,lower,ma,ma5,std,mas,mam5,s5,kdj_d2,kdjm,interval,atr,maxp".split(
                 ","))
        sd = round(slope * 3, 0) / 10
        sd = np.sign(sd) * 0.5 if abs(sd) > 0.5 else sd

        # 超3倍反常值
        mam = (high - top) / std if high > top else (low-lower)/std if low < lower else 0
        # 宽度和sd矫正
        mam = mam - sd if mam != 0 else 0

        wstd = 4 * std * 100 / ma
        yd = 0.5 if close == open else  (high - close) / (high - open) if close > open else (close -low) / (open - low)

        # 差值5 或者 interval > 7
        opt0= interval > 6 and (s5 * mas > 0)
        opt1= interval > 10 and (slope * mas > 0)
        opt2= interval > 19 and (slope * mas > 0)

        trend = np.sign(mas) * 2 if opt2 else np.sign(mas) if (opt0 or opt1) else 0
        # kdj
        powm, super = 0, 0
        datr = np.sign(mas)  if np.sign(mas) * (maxp - close) > 1.8 * atr else 0

        if wds > 0.5 and wstd > 0.6:

            if trend ==0:
                # 超涨
                super = np.sign(mam) * 2 if abs(mam) > 1 else np.sign(mam) if abs(mam) > 0.25 else 0

            else:
                super = np.sign(mam) * 3 if abs(mam) > 2 else 0


        columns = self.pointColumns
        return pd.Series([mam, super, powm, yd, trend, datr], index=columns)

    def sd(self, x):
        s = round(x * 1.5, 0) / 10
        return np.sign(s) * 0.8 - 0.1 if abs(s) > 0.8 else s - 0.1

    def isout(self, row):
        close, top, lower = (row[key] for key in "close,top,lower".split(","))
        return  1 if close > top else -1 if close < lower else 0

    def isout0(self, row):
        close, ma = (row[key] for key in "close,ma".split(","))
        return 0 if np.isnan(ma) else 1 if close > ma else -1

    def isop(self, row):
        op, out = (row[key] for key in "op,out".split(","))
        return 0 if op >=0 else 1 if out > 0 else -1

    def mam(self, row):
        mas, mam = (row[key] for key in "mas,mam5".split(","))
        return 0 if mam >=0 else 1 if mas > 0 else -1

    def getInterval(self, row):
        index, mio, mao = (row[key] for key in "No,mio,mao".split(","))
        return index - max(mio, mao) + 1
        pass

    def getLastMax(self, row, high, low):
        no, mio, mao,mas = (row[key] for key in "No,mio,mao,mas".split(","))
        return high[mao:no].max() if mas > 0 else low[mio:no].min()

    def total(self, dfs, dfs2=None, period=60):
        # 计算参数
        df0 = dfs[self.mCodes[0]]
        df0["rel_price"] = close = df0["close"]
        df0["datetime"] = df0.index

        cc = close.reset_index(drop=True)
        df0["No"] = cc.index

        # bull
        df0["ma"] = ma = ta.SMA(close, timeperiod=period)
        df0["ma5"] = ma5 = ta.SMA(close, timeperiod=5)
        df0["s5"] = ta.LINEARREG_SLOPE(ma5, timeperiod=2)

        df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)
        df0["std5"] = std5 = ta.STDDEV(close, timeperiod=5, nbdev=1)

        df0["top"], df0["lower"] = ma+ 2 * std,  ma - 2*std
        df0["wds"] = self.stand(4 * std * 100 / ma, period= period)

        #df0['isout'] = isout = df0.apply(lambda row: self.isout(row), axis=1)
        #df0['isout3'] = ta.SUM(isout, timeperiod=3)

        # 趋势计算
        df0['mas'] = mas = ma5 - ma
        df0['mam5'] = mas * mas.shift(1)
        df0['mam5'] = om = df0.apply(lambda row: self.mam(row) , axis=1)

        df0['mio'], df0['mao'] =  ta.MINMAXINDEX(om, timeperiod=50)
        df0['interval'] = df0.apply(lambda row: self.getInterval(row), axis=1)

        # 计算止损
        df0['maxp'] = df0.apply(lambda row: self.getLastMax(row, df0['high'], df0['low']), axis=1)

        # 上下柜
        width = (4 * std / ma * 100).fillna(0)

        df0["wds"] = self.stand(width)

        # 短期价格波动参数值
        df0["slope"] = self.stand(ta.LINEARREG_SLOPE(close, timeperiod=period))

        # kdj顶点
        kdjK, kdjD = ta.STOCH(df0["high"], df0["low"], close,
                              fastk_period=5, slowk_period=3, slowk_matype=1, slowd_period=3,
                              slowd_matype=1)

        df0["kdj_d2"] = kdj_d2 = kdjK - kdjD
        df0["kdjm"] = kdj_d2 * kdj_d2.shift(1)
        df0["kdjm"] = df0.apply(lambda row: self.turn(row['kdjm'], row['kdj_d2'], 1), axis=1)

        df0['atr'] =ta.ATR(df0['high'], df0['low'], close, timeperiod=period)

        # 循环 scale
        docs = []
        for scale in self.scaleList:
            for conds in self.iterCond():
                ktype = self.klineType[:-1]
                uid = self.uidKey % (
                    '_'.join(self.codes), str(period), str(scale), ktype,
                    str(self.widthTimesPeriod) + '_' + conds)

                # 计算关键转折点
                df1 = df0.apply(lambda row: self.point(row, ktype, period), axis=1)
                for key in self.pointColumns:  df0[key] = df1[key]

                isCvs = False
                for key in self.csvList:
                    if uid.find(key) > -1:
                        #if self.codes[0] in ['MA','SC']:
                            isCvs = True
                            break

                if isCvs and self.isAll == 0:
                    cs = []
                    bans = 'p_l,p_h,rel_price,mio,mao,op,om'.split(',')
                    for c in df0.columns:
                        if c not in bans:
                            cs.append(c)

                    file = self.Rice.basePath + '%s_pre.csv' % (uid)
                    print(uid, '---------------------------- to_cvs', file)
                    df0.to_csv(file, index=0, columns=cs)

                # df0.fillna(0, inplace=True)
                df1 = None
                tot = None
                #tot = self.detect(df0, df1, period=period, uid=uid)
                if tot is not None and tot['amount'] != 0:
                    tot.update(
                        {
                            "scale": scale,
                            "code": self.codes[0],
                            "period": period,
                            "uid": uid,
                           # "shift": (p_l - p_h).mean(),
                            "createdate": public.getDatetime()
                        }
                    )
                    docs.append(tot)
        return docs

    # 买入后的最高/最低价
    def getMax(self, df0, s, e, mode):
        s = str(s)[:10]
        if mode > 0:
            return df0[(df0['datetime']>=s) & (df0['datetime']<e)].ix[:-1, 'close'].max()
        else:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'close'].min()

    # 核心策略部分
    def stageApply(self, df0, df1, period=15, uid=''):
        self.records = []
        """
            布林带策略：            
        """
        status, isOpen = 0, 0
        adjusts = [str(c[1])[:10]+' 09:15:00' for c in self.adjustDates]

        if self.codes[0]=='JD': print(adjusts)

        for i in range(period, len(df0)):
            isRun, isstop = False, 0
            ma, close, date,  sarm2, sarm5, powm, isout, atr, tu60,td60 = (
                df0.ix[i, key] for key in
                "ma,close,datetime,sarm2,sarm5,powm,isout,atr,tu60,td60".split(
                    ","))


            if isOpen!=0 and date in adjusts:
                self.adjustOrder(df0.iloc[i], date)
                """
                print('adjusts')
                optm = mas * kdjm < 0
                optm1 = mas * kdjm < 0

                opt01 = 1 if yd < 0.3 else 0
                opt02 = 1 if np.sign(mas) * (close - s5) < 0 else 0

                opt03= 1 if np.sign(mas) * (open - ma) / std > 1.5 else 0
                opt04 = 1 if np.sign(mas) * (close - ma) / std > 0.25 else 0

                opt06 =  1 if (-np.sign(mas) * (close - s5) > 3 * self.shift[0]) else 0
                opt07 = np.sign(mas) * (maxp - close) > 2 * atr
                """

            if isOpen == 0 :
                sign = np.sign(isout)
                tp = tu60 if sign > 0 else td60
                if abs(powm)==2:
                    isOpen, isRun = - 1 * np.sign(powm), True

                elif abs(isout)==3:
                    isOpen, isRun = 2 * np.sign(isout), True

                elif (abs(isout)==2 and sign * (close - tp) > 0):
                    isOpen, isRun = 3 * np.sign(isout), True

            elif self.preNode is not None:

                keeps = len(df0[df0['datetime'] > str(self.preNode[0]['createdate'])])

                if abs(isOpen) ==1 and powm * isOpen > 0:
                    if abs(powm) == 2:
                       if self.order(df0.iloc[i], None, 0, uid, df0, isstop=3):
                            isOpen, isRun = 4 * np.sign(powm), True
                    else:
                            isOpen, isRun, isstop = 0, True, 1

                elif abs(isOpen) ==2 and keeps > 4 and sarm5 * isOpen < 0:
                        isOpen, isRun, isstop = 0, True, 2

                else:
                    # 即时收益超出
                     Pd, s = self.preNode[0]['price'], self.preNode[0]['createdate']
                     mp = self.getMax(df0, s, date, isOpen)
                     mp = close if np.isnan(mp) else mp
                     if not np.isnan(mp) and (np.sign(isOpen) * (mp - close) > 2.0 * atr):
                          isOpen, isRun, isstop = 0, True, 6

            if isRun:
                # print('----- start', df0.ix[i,['datetime']], status, isOpen, isstop, isRun, powm)
                self.order(df0.iloc[i], None, isOpen, uid, df0, isstop=isstop)
        return self.records


def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = train_future_macd()
        obj.Pool()

if __name__ == '__main__':
    main()
