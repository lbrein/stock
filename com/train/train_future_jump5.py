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
from multiprocessing import Pool, Manager
import statsmodels.api as sm  # 协整
import copy


# 回归方法
class train_future_singleExpect0(object):
    """

    """
    iniAmount = 250000  # 单边50万
    csvList = [
        "SM_30_2.0_5_5_0.025_0.5_0.1_quick_single",
        "JM_15_2.0_60_1_0.25_1.0_0.1_quick_single",
        "JD_15_2.0_60_5_0.025_0.5_1.25_1_quick_single",
        "RB_15_2.0_30_5_0.2_0.5_1.25_1_quick_single",
        "AP_15_2.0_15_1_0.2_0.5_0.1_quick_single_13"
    ]

    def __init__(self):
        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.isEmptyUse = False  # 是否清空记录

        self.codeLists = ['JM', 'SM', 'V', 'I', 'AP', 'J','RB', 'SC', 'MA', 'JD','CU' , 'OI']

        self.baseInfo = {}


        # k线时间
        # self.klineTypeList = ['5m']
        self.klineTypeList = ['15m', '30m']

        self.widthDeltaLineList = [0.025]
        self.widthDeltaLine = 0

        self.stopTimeLine = 5
        self.widthTimesPeriodList = [3]

        self.testDays = 90
        # 起始时间
        self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测
        self.endDate = public.getDate(diff=0)

        self.total_tablename = 'train_total_1'
        self.detail_tablename = 'train_future_1'
        self.totalMethod = 'single'
        self.method = 'simTick' if self.isSimTickUse else 'quick'
        self.stage = 'single15'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method + "_" + self.stage
        self.isAll = 0

    def iterCond(self):
        # 多重组合参数输出
        keys = ['widthDeltaLine', 'scaleDiff2', 'scaleDiff']
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                self.__setattr__(keys[1], s1)

                for s2 in self.__getattribute__(keys[2] + 'List'):
                    self.__setattr__(keys[2], s2)

                    yield '%s_%s_%s' % (str(s0), str(s1), str(s2))

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
            if codes[0] not in self.codeLists:  continue

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

    cindex = 0

    def start(self, codes, time0, kt, shareDict):
        print("子进程启动:", self.cindex, codes, time.time() - time0)

        self.klineType = kt
        # 主力合约
        self.codes = codes
        self.mCodes = mCodes = [n + '88' for n in codes]

        # 查询获得配置 - 费率和每手单量
        self.Base = future_baseInfo()
        for doc in self.Base.getInfo(codes):
            self.baseInfo[doc["code"] + '88'] = doc

        cs = [self.baseInfo[m] for m in self.mCodes]

        # 计算tick 导致的滑点
        sh = [self.baseInfo[d + '88']['tick_size'] for d in codes]
        self.shift = [sh[i] * self.shiftScale for i in range(len(sh))]

        # 子进程共享类
        self.Rice = tick_csv_Rice()
        self.Rice.setTimeArea(cs[0]["nightEnd"])
        self.Train = train_future()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename
        self.Train.tablename = self.detail_tablename

        # 查询获得N分钟K线
        dfs_l = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=60)

        # 按时间截取并调整
        # dfs= self.dateAdjust(codes, dfs, sh)
        print('kline load:', mCodes, [len(dfs[m]) for m in mCodes])

        # 根据配置文件获取最佳交易手数对
        self.iniVolume = round(self.iniAmount / cs[0]["lastPrice"] / cs[0]["contract_multiplier"], 0)

        # 分参数执行
        results = []
        for period in self.periodList:
            for wdp in self.widthTimesPeriodList:
                self.widthTimesPeriod = int(wdp)
                docs = self.total(dfs, dfs_l, period=period)
                if docs is None or len(docs) == 0:  continue
                logger.info((self.codes, period, self.klineType, len(docs), " time:", time.time() - time0))
                self.Total.insertAll(docs)

    def total(self, dfs, dfs2=None, period=60):
        # 计算参数
        df0 = dfs[self.mCodes[0]]
        df0["rel_price"] = close = df0["close"]
        df0["datetime"] = df0.index

        s0 = self.shift[0]
        p_l = df0["p_l"] = (df0["close"] + s0)
        p_h = df0["p_h"] = (df0["close"] - s0)

        if self.isSimTickUse:
            # 调用复合apply函数计算混合参数
            close2 = dfs2[self.mCodes[0]]["close"]
            df0_1 = df0.apply(lambda row: self.k_ma(row['datetime'], row['rel_price'], close2, period), axis=1)
            df0 = pd.concat([df0, df0_1], axis=1)

        else:
            df0["ma"] = ma = ta.MA(close, timeperiod=period)
            df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)
            # 上下柜
            # bullWidth
            df0["bullwidth"] = width = (4 * std / ma * 100).fillna(0)
            # 近三分钟width变动
            df0["widthDelta"] = wd1 = ta.MA(width - width.shift(1), timeperiod=self.widthTimesPeriod).fillna(0)
            df0["widthDelta2"] = wd2 = wd1 - wd1.shift(1)
            df0["wd2m"] = wd2 * wd2.shift(1)

            dif, dea, macd = ta.MACD(close, fastperiod=int(period / 3), slowperiod=period, signalperiod=9)
            df0["mastd"] = ta.STDDEV(macd, timeperiod=period, nbdev=1)
            df0["macdm"] = macd * macd.shift(1)
            df0["macd2d"] = macd - macd.shift(1)
            df0["macd2dm"] = (macd - macd.shift(1)) * (macd.shift(1) - macd.shift(2))

        # 相对波动
        df0['delta'] = (p_l - p_h) / df0['std']

        df1 = None
        # 循环 scale
        docs = []
        for scale in self.scaleList:
            for conds in self.iterCond():
                uid = self.uidKey % (
                    '_'.join(self.codes), str(period), str(scale), self.klineType[:-1],
                    str(self.widthTimesPeriod) + '_' + conds)

                self.stopTimeDiff = self.stopTimeLine * period * int(self.klineType[:-1])
                # 计算高低线值
                df0["top"], df0["lower"] = df0['ma'] + (scale - self.scaleDiff) * df0['std'], df0['ma'] - (
                        scale + self.scaleDiff) * df0['std']

                if uid in self.csvList:
                    file = self.Rice.basePath + '%s_%s_pre.csv' % ('_'.join(self.codes), self.klineType)
                    print(uid, '---------------------------- to_cvs', file, df0.columns)
                    df0.to_csv(file, index=0)

                # df0.fillna(0, inplace=True)
                # tot = None
                tot = self.detect(df0, df1, period=period, uid=uid)
                if tot is not None and tot['amount'] != 0:
                    tot.update(
                        {
                            "scale": scale,
                            "method": self.totalMethod,
                            "code": self.codes[0],
                            "period": period,
                            "uid": uid,
                            "shift": (p_l - p_h).mean(),
                            "createdate": public.getDatetime()
                        }
                    )
                    docs.append(tot)
        return docs

    def detect(self, df0, df1, period=15, uid=''):
        docs = self.stageApply(df0, df1, period=period, uid=uid)
        res = pd.DataFrame(docs)
        if len(res) > 0:
            if self.saveDetail:
                self.Train.insertAll(docs)

            diff = res[res['diff'] > 0]['diff'].mean()

            # 计算最大回测
            sum = res['income'].cumsum() + self.iniAmount
            inr = res['income'] / self.iniAmount
            # 计算夏普指数
            sha = (res['income'].sum() / self.iniAmount - 0.02 * self.testDays / 252) / inr.std()
            return {
                "count": int(len(docs) / 2),
                "amount": self.iniAmount,
                "price": res['rel_price'].mean(),
                "income": res["income"].sum(),
                "std": res['rel_std'].mean(),
                "delta": res['delta'].mean(),
                "maxdown": ((sum.shift(1) - sum) / sum.shift(1)).max(),
                "sharprate": sha,
                "timediff": int(0 if np.isnan(diff) else diff)
            }
        else:
            return None

    def ordercheck(self):
        pass


    # 核心策略部分
    def stageApply(self, df0, df1, period=15, uid=''):
        isOpen, isRise, preDate, prePrice = 0, 0, None, 0
        doc, docs = {}, []

        ma, p_l, p_h, top, lower, std, delta, width, wd1, wd2, wd2m, macd2d, macd2dm = (df0[key] for key in
                                          "ma,p_l,p_h,top,lower,std,delta,bullwidth,widthDelta,widthDelta2,wd2m,macd2d,macd2dm".split(
                                                                                            ","))

        sline, wline = self.stopTimeDiff, self.widthDeltaLine * width.mean() / 2.75

        for i in range(period, len(df0)):


        return docs

    batchId = None

    def calcIncome(self, n0, p0, df0):
        # 计算收益，最大/最小收益
        high = df0[(p0['createdate'] <= df0.index) & (df0.index <= n0['datetime'])]['high'].max()
        low = df0[(p0['createdate'] <= df0.index) & (df0.index <= n0['datetime'])]['low'].min()
        close = n0["close"]
        sign = p0["mode"] / abs(p0["mode"])

        # 收入
        income = sign * (close - p0["price"] - 2 * sign * self.shift[0]) * p0["vol"] - p0["fee"]
        # 最大收入
        highIncome =  sign * ((high if sign > 0 else low) - p0["price"] - 2 * sign * self.shift[0]) * p0["vol"] - p0["fee"]
        # 最大损失
        lowIncome = sign * ((high if sign < 0 else low) - p0["price"] - 2 * sign * self.shift[0]) * p0["vol"] - p0["fee"]

        return income, highIncome, lowIncome, high, low

    def order(self, n0, n1, mode, uid, df0, isstop=0):
        # baseInfo 配置文件，查询ratio 和 每手吨数
        b0 = self.baseInfo[self.mCodes[0]]
        if mode != 0:
            self.batchId = uuid.uuid1()
        # 交易量
        v0 = self.iniVolume * b0["contract_multiplier"]
        # 费率
        fee0 = (self.iniVolume * b0["ratio"]) if b0["ratio"] > 0.5 else ((b0["ratio"]) * n0["close"] * v0)

        doc = {
            "createdate": n0["datetime"],
            "code": self.codes[0],
            "price": n0["close"],
            "vol": self.preNode[0]["vol"] if self.preNode else v0,
            "mode": mode if not self.preNode else -self.preNode[0]["mode"],
            "isopen": 0 if mode == 0 else 1,
            "fee": fee0,
            "income": 0,
            "isstop": isstop,
            "rel_price": n0["rel_price"],
            "rel_std": n0["std"],
            "bullwidth": n0["bullwidth"],
            "widthDelta": n0["widthDelta"],
            "price_diff": n0["widthDelta2"],
            "shift": n0["wd2m"],
            "delta": n0["delta"],
            "batchid": self.batchId,
            'p_l': n0["p_l"],
            'p_h': n0["p_h"],
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
        obj = train_future_singleExpect0()
        obj.Pool()


if __name__ == '__main__':
    main()
