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
        "30_2.0_30_3_2.0_1.75",
        "15_2.0_15_3_2.0_2.5",
    ]

    def __init__(self):
        # 费率和滑点
        super().__init__()

        self.saveDetail = True  # 是否保存明细
        self.savePoint = False
        self.isSimTickUse = False  # 是否使用1分钟模拟tick测试，否则直接使用kline回测
        self.topUse = False
        self.isEmptyUse = False
        self.baseInfo = {}
        self.codeLists = ['JM', 'SM', 'V', 'I', 'AP', 'J', 'RB', 'SC', 'MA', 'JD', 'CU', 'OI']
        # self.codeLists = ['JM']

        # 可变参数
        self.periodList = [26]  # 窗体参数
        self.scaleList = [2.0]
        self.klineTypeList = ['15m']

        self.widthTimesPeriodList = [3]

        self.superlineList = [2.0]

        self.powline = 0.25
        self.turnlineList = [1.75, 2.5]
        self.turnline = 1.5
        self.paramList = ['superline', 'turnline']

        self.shiftScale = 0.527  # 滑点模拟系数
        self.scaleDiff2 = 0.8
        self.scaleDiff = 0
        self.processCount = 6

        self.testDays = 90
        self.adjustDates = []
        # 起始时间
        self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测
        self.endDate = public.getDate(diff=0)

        self.total_tablename = 'train_total_1'
        self.detail_tablename = 'train_future_1'

        self.method = 'simTick' if self.isSimTickUse else 'quick'
        self.stage = 'dema6'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method + "_" + self.stage

        # 切换全部计算
        self.isAll = 0

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
            print(rs)
            if self.isAll == 0 and codes[0] not in self.codeLists:  continue

            for kt in self.klineTypeList:
                #self.start(codes, time0, kt, shareDict)
                try:
                    pool.apply_async(self.start, (codes, time0, kt, shareDict))
                    pass
                except Exception as e:
                    print(e)
                    continue
        pool.close()
        pool.join()

    pointColumns = ['wdd', 'pow', 'powm', 'trend', 'isout']

    def point(self, row, ktype=15, period=15):
        wd, wd1, wds, s1, s2, high, low, open, close, top, lower, atr, atr1, tu, td, tu30, td30, sd = \
            (row[key] for key in
             "width,wd1,wdstd,slope,slope60,high,low,open,close,top,lower,atr,atr1,tu,td,tu30,td30,sd".split(
                 ","))

        TL, PL = self.turnline, self.powline
        # trend.

        if close > open:
            trend = 1 if high == open else ((close - open) / (high - open) + 0.5)
        else:
            trend = -1 if low == open else ((close - open) / (open - low) - 0.5)

        tr = (4 - atr * atr1) if atr > 2 else atr * atr1

        if tr < 0: tr = 0
        # pow
        pow = (wd1 + abs(s1 + s2 + trend * tr)) / 4

        sign = np.sign(s1) if s1 != 0 else np.sign(trend) if trend != 0 else np.sign(s2)

        wdd = wd / math.pow(float(ktype) * float(period), 0.5) * 15

        wdf = (wd / math.pow(float(ktype) * float(period), 0.5) * 15 + atr + atr1) / 3

        isout = -1 if close * (1 - 0.001) > (top + sd) else 1 if close * (1 + 0.001) < (lower - sd) else 0

        # powm
        opt1 = ((wdd < 2.5) and (close > tu or close < td))
        opt2 = pow > TL and ((wdd < 2.5) and (close > tu30 or close < td30))
        opt3 = (pow > PL or wdf > 1.0) and wdd > 0.5

        powm = 2 * sign if opt1 or opt2 else sign if opt3 else 0

        if (atr > 3.25) and (np.sign(trend) * isout < 0) and pow < 10:
            powm = -3 * np.sign(trend)

        columns = self.pointColumns
        return pd.Series([wdd, pow, int(powm), trend, int(isout)], index=columns)

    def sd(self, x):
        s = round(x * 1.5, 0) / 10
        return np.sign(s) * 0.8 - 0.1 if abs(s) > 0.8 else s - 0.1

    def total(self, dfs, dfs2=None, period=60):
        # 计算参数
        df0 = dfs[self.mCodes[0]]
        df0["rel_price"] = close = df0["close"]
        df0["datetime"] = df0.index

        s0 = self.shift[0]
        p_l = df0["p_l"] = (df0["close"] + s0)
        p_h = df0["p_h"] = (df0["close"] - s0)

        # bull
        df0["ma"] = ma = ta.MA(close, timeperiod=period)
        df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)
        # 上下柜
        df0["width"] = width = (4 * std / ma * 100).fillna(0)

        # kdj顶点
        kdjK, kdjD = ta.STOCH(df0["high"], df0["low"], close,
                              fastk_period=5, slowk_period=3, slowk_matype=1, slowd_period=3,
                              slowd_matype=1)

        df0["kdj_d2"] = kdj_d2 = kdjK - kdjD
        df0["kdjm"] = kdj_d2 * kdj_d2.shift(1)
        df0["kdjm"] = df0.apply(lambda row: self.turn(row['kdjm'], row['kdj_d2'], 1), axis=1)

        # SAR
        sar = ta.SAR(df0['high'], df0['low'], acceleration=0.03, maximum=0.2)

        df0['sard'] = sard = close - sar
        df0['sarm'] = sard * sard.shift(1)
        df0['sarm'] = df0.apply(lambda row: self.turn(row['sarm'], row['sard'], 1), axis=1)

        # ATR
        df0['atr'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=1) / ta.ATR(df0['high'], df0['low'], close,
                                                                                   timeperiod=period)

        df0['atr1'] = self.stand(ta.ATR(df0['high'], df0['low'], close, timeperiod=period))
        df0['atrb'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=period)

        # 唐奇安线
        df0['tu'] = ta.MAX(df0['high'].shift(1), timeperiod=60)
        df0['td'] = ta.MIN(df0['low'].shift(1), timeperiod=60)

        # 唐奇安线
        df0['tu30'] = ta.MAX(df0['high'].shift(1), timeperiod=30)
        df0['td30'] = ta.MIN(df0['low'].shift(1), timeperiod=30)


        # 循环 scale
        docs = []
        for scale in self.scaleList:
            for conds in self.iterCond():
                ktype = self.klineType[:-1]

                if self.isAll == 1:
                    b0 = self.baseInfo[self.mCodes[0]]
                    # 修正品种
                    bulltype = b0['bulltype']
                    # if bulltype == 0 and self.turnline > 2.0: continue
                    # if bulltype == 2 and (self.turnline < 2.0 or ktype == '5'): continue

                uid = self.uidKey % (
                    '_'.join(self.codes), str(period), str(scale), ktype,
                    str(self.widthTimesPeriod) + '_' + conds)

                df0["top"], df0["lower"] = df0['ma'] + (scale + df0['sd']) * df0['std'], df0['ma'] - (
                        scale - df0['sd']) * df0['std']

                # 计算关键转折点
                df1 = df0.apply(lambda row: self.point(row, ktype, period), axis=1)
                for key in self.pointColumns: df0[key] = df1[key]
                df0['isout'] = ta.SUM(df0['isout'], timeperiod=3)

                isCvs = False
                for key in self.csvList:
                    if uid.find(key) > -1:
                        isCvs = True
                        break

                if isCvs and self.isAll == 0:
                    cs = []
                    bans = 'ma,open,close,high,low,p_l,p_h,top,lower,std,delta,volume,sard,rel_price,volm,'.split(
                        ',')
                    for c in df0.columns:
                        if c not in bans:
                            cs.append(c)

                    file = self.Rice.basePath + '%s_pre.csv' % (uid)
                    print(uid, '---------------------------- to_cvs', file)
                    df0.to_csv(file, index=0, columns=cs)
                    # self.share.append(self.codes)

                # df0.fillna(0, inplace=True)
                df1 = None
                tot = None
                tot = self.detect(df0, df1, period=period, uid=uid)
                if tot is not None and tot['amount'] != 0:
                    tot.update(
                        {
                            "scale": scale,
                            "code": self.codes[0],
                            "period": period,
                            "uid": uid,
                            "shift": (p_l - p_h).mean(),
                            "std": df0['width'].mean(),
                            "createdate": public.getDatetime()
                        }
                    )
                    docs.append(tot)
        return docs

    # 核心策略部分
    def stageApply(self, df0, df1, period=15, uid=''):
        self.records = []
        """
            布林带策略：            
        """
        status, isOpen = 0, 0
        pd2 = self.turnline - 0.20
        b0 = self.baseInfo[self.mCodes[0]]
        # 修正品种
        bulltype = b0['bulltype']

        adjusts = [str(c[1])[:10]+' 09:15:00' for c in self.adjustDates]
        print(adjusts)

        for i in range(period, len(df0)):
            isRun, isstop = False, 0

            ma, close,date,p_l, p_h, top, lower, std, kdjm, sarm, pow, powm, sd, isout, atr, atr1, atrb, s1, s2, wdd = (
                df0.ix[i, key] for key in
                "ma,close,datetime,p_l,p_h,top,lower,std,kdjm,sarm,pow,powm,sd,isout,atr,atr1,atrb,slope,slope60,wdd".split(
                    ","))

            sd2 = (self.scaleDiff2 - 1.2 * sd) * std / 2  # std修正

            if isOpen!=0 and date in adjusts:
                  self.adjustOrder(df0.iloc[i], date)

            if status == 0 and powm in [2, -2]:
                """
                   突变点处理 
                   将布林带策略切换为策略 
                """
                status = int(np.sign(powm))

                if isOpen * status < 0:
                    if self.order(df0.iloc[i], None, 0, uid, df0, isstop=4):
                        isOpen = 0
                        if bulltype == 0 and abs(s2) < 1.25:
                            isOpen, isRun = int(4 * status), True

                elif isOpen == 0 and abs(s2) < 1.25:
                    isOpen, isRun = int(5 * status), True

            elif powm in [3, -3] and 4.0 < abs(s1) < 10 and isout != 0:
                # 触底反弹
                status = 0
                if isOpen != 0 and isOpen * powm < 0:
                    if self.order(df0.iloc[i], None, 0, uid, df0, isstop=6):
                        isOpen, isRun = int(2 * powm), True

                elif isOpen == 0:
                    isOpen, isRun = int(2 * powm), True

            elif status != 0 or abs(isOpen) in [4, 5]:
                """
                    交叉点: sarm结束点 
                    1、结束突变状态，变为布林带处理
                    2、根据节点状况开平新仓，状态为6 
                """

                if isOpen * sarm < 0:
                    if self.order(df0.iloc[i], None, 0, uid, df0, isstop=5):
                        isOpen = 0
                        if abs(s1) > 1.5:
                            isOpen, isRun = 3 * np.sign(sarm), True

                    status = 0

                elif status * sarm < 0:
                    status = 0

                elif isOpen != 0:
                    # 即时收益超出
                    if self.preNode is not None:
                        sign = np.sign(isOpen)
                        Pd = self.preNode[0]['price']
                        income = sign * (Pd - close)
                        if (income > 2 * atrb and sign * kdjm < 0) or (
                                income > (2 + 0.5) * atrb):
                            isOpen, isRun, isstop = 0, True, 7

            elif status == 0:
                """
                    布林带策略处理                 
                """
                cond1 = powm == 0 and abs(s1) < 0.75 and wdd > 0.5
                if isOpen == 0:
                    # 突变状态开始
                    # 大于上线轨迹

                    if p_h >= top and cond1:
                        isOpen, isRun = -1, True

                    elif p_l <= lower and cond1:
                        isOpen, isRun = 1, True

                    elif ((p_h + sd2) >= top or (isout < 0 and (p_h - sd2) >= ma)) and pow < pd2 and kdjm < 0:
                        isOpen, isRun = -2, True

                    elif ((p_l - sd2) <= lower or (isout > 0 and (p_l + sd2) <= ma)) and pow < pd2 and kdjm > 0:
                        isOpen, isRun = 2, True

                # 平仓
                else:
                    sign = np.sign(isOpen)
                    cond3 = (sign * ((p_h if isOpen > 0 else p_l) - ma))
                    #
                    if (cond3 >= - sd2 or sign * isout < 0) and sign * kdjm < 0:
                        if ((p_h + sd2) >= top or (p_l - sd2) <= lower) and pow < pd2:
                            if self.order(df0.iloc[i], None, 0, uid, df0, isstop=1):
                                # 再开盘
                                isOpen, isRun = -2 if (p_h + sd2) >= top else 2, True

                        else:
                            isOpen, isstop = 0, 1
                            isRun = True

                    elif cond3 >= 0 and powm == 0 and abs(s1) < 1:
                        isOpen, isstop = 0, 0
                        isRun = True

                    # 超过2倍 atr，止损
                    else:
                        # 即时收益超出
                        if self.preNode is not None:
                            Pd = self.preNode[0]['price']
                            income = sign * (Pd - close)
                            if (income > 2 * atrb and sign * kdjm < 0) or (
                                    income > (2 + 0.5) * atrb):
                                isOpen, isRun, isstop = 0, True, 7
                    # 周末尾盘

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
