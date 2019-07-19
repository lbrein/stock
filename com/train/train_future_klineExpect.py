# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ---  与tick 配套对比计算的 kline
    > 1m的 K线 模拟tick预测
    > 定期执行，用于筛选交易对
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
class train_future_klineExpect(object):
    """

    """
    iniAmount = 250000  # 单边50万

    def __init__(self):
        # 费率和滑点
        self.saveDetail = False # 是否保存明细
        self.isSimTickUse = False  # 是否使用1分钟模拟tick测试，否则直接使用kline回测
        self.topUse = True
        self.isEmptyUse = False  # 是否情况记录
        self.isKeeping = False

        self.baseInfo = {}
        self.periodList = [15, 30, 60]  # 窗体参数
        self.scaleList = [2.0, 2.5]
        self.shiftScale = 0.527  # 滑点模拟系数
        self.deltaLine = 0.8

        # k线时间
        self.klineTypeList = ['5m', '15m', '30m', '60m']
        self.scaleDiff2List = [0.5, 0.75]
        self.widthDeltaLineList = [0.03, 0.06, 0.1, 0.2]
        self.stopTimeLineList = [3]
        self.widthDeltaLine = 0
        self.scaleDiffList = [0.1]

        self.testDays = 60
        # 起始时间
        self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测

        self.endDate = public.getDate()
        self.total_tablename = 'train_total_2'
        self.detail_tablename = 'train_future_2'
        self.method = 'simTick' if self.isSimTickUse else 'quick'
        self.stage = 'paird21'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method + "_" + self.stage

    def iterCond(self):
        # 多重组合参数输出
        keys = ['stopTimeLine', 'widthDeltaLine','scaleDiff2','scaleDiff']
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                self.__setattr__(keys[1], s1)

                for s2   in self.__getattribute__(keys[2] + 'List'):
                    self.__setattr__(keys[2], s2)

                    for s3 in self.__getattribute__(keys[3] + 'List'):
                        self.__setattr__(keys[3], s3)
                        yield '%s_%s_%s_%s' % (str(s0), str(s1), str(s2),str(s3))

    def empty(self):
        if self.isEmptyUse:
            Train = train_future()
            Total = train_total()
            Total.tablename = self.total_tablename
            Train.tablename = self.detail_tablename
            Train.empty()
            Total.empty()

    def tops(self, num=5):
        Total = train_total()
        Total.tablename = "train_total_2"
        return [m[0:2] for m in Total.last_top(num=num)], Total.exits()

    def Pool(self):
        time0 = time.time()

        pool = Pool(processes=4)
        shareDict = Manager().list([])
        self.empty()

        Base = future_baseInfo()
        # 交易量大的，按价格排序, 类型:turple,第二位为夜盘收盘时间
        lists = Base.all(vol=700)

        tops, exists = self.tops()

        for rs in list(itertools.combinations(lists, 2)):
            # 检查时间匹配
            if rs[0][1] != rs[1][1]:  continue
            codes = [rs[0][0], rs[1][0]]
            if self.topUse and codes not in tops:
                    continue

            if self.isKeeping and codes in exists:
                continue

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

        cs0, cs1 = self.baseInfo[mCodes[0]], self.baseInfo[mCodes[1]]

        if cs0["nightEnd"] != cs1["nightEnd"]: return None

        # 计算tick 导致的滑点
        sh = [self.baseInfo[d + '88']['tick_size'] for d in codes]
        self.shift = [sh[i] * self.shiftScale for i in range(len(sh))]

        # 子进程共享类
        self.Rice = tick_csv_Rice()
        self.Rice.setTimeArea(cs0["nightEnd"])
        self.Train = train_future()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename
        self.Train.tablename = self.detail_tablename
        self.share = shareDict

        # 查询获得N分钟K线
        dfs_l = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=60)


        # 获得1分钟K线作为滚动线
        if self.isSimTickUse:
            dfs = self.Rice.kline(mCodes, period='1m', start=self.startDate, end=self.endDate, pre=0)
        else:
            dfs = dfs_l

        # 按时间截取并调整
        # dfs= self.dateAdjust(codes, dfs, sh)
        print('kline load:', mCodes, len(dfs[mCodes[0]]), len(dfs[mCodes[1]]))

        # 根据配置文件获取最佳交易手数对
        self.iniVolume = round(self.iniAmount / cs0["lastPrice"] / cs0["contract_multiplier"], 0)

        # 分参数执行
        results = []
        for period in self.periodList:
            docs = self.total(dfs, dfs_l, period=period)

            if docs is None or len(docs) == 0:  continue
            logger.info((self.codes, period, self.klineType, len(docs), " time:", time.time() - time0))
            self.Total.insertAll(docs)
            # results.extend(docs)

    #  混合K线-计算平均值和标准差
    def k_ma(self, d, p, close, period):
        # 截
        df = close[close.index < d][-period - 5:]
        df = df.append(pd.Series([p], index=[d]))
        # 平均值
        ma = ta.MA(df, timeperiod=period)
        std = ta.STDDEV(df, timeperiod=period, nbdev=1)
        # bullWidth
        width = (4 * std / ma * 100).fillna(0)

        # 近width变动
        wd1 = ta.MA(width - width.shift(1), timeperiod=3).fillna(0)
        wd2 = wd1 - wd1.shift(1)
        wd2m = wd2 * wd2 * wd2.shift(1)
        columns = ['ma', 'std', 'bullwidth', 'widthDelta', 'widthDelta2', 'wd2m']
        return pd.Series([x.values[-1] for x in [ma, std, width, wd1, wd2, wd2m]], index=columns)


    preNode, batchId = None, {}

    def total(self, dfs, dfs2=None, period=60):
        # 计算参数
        df0, df1 = dfs[self.mCodes[0]], dfs[self.mCodes[1]]
        df0["rel_price"] = close = df0["close"] / df1["close"]
        df0["datetime"] = df0.index

        s0, s1 = self.shift[0], self.shift[1]
        p_l = df0["p_l"] = (df0["close"] + s0) / (df1["close"] - s1)
        p_h = df0["p_h"] = (df0["close"] - s0) / (df1["close"] + s1)

        if self.isSimTickUse:
            # 调用复合apply函数计算混合参数
            close2 = dfs2[self.mCodes[0]]["close"] / dfs2[self.mCodes[1]]["close"]
            df0_1 = df0.apply(lambda row: self.k_ma(row['datetime'], row['rel_price'], close2, period), axis=1)
            df0 = pd.concat([df0, df0_1], axis=1)

        else:

            df0["ma"] = ma = ta.MA(close, timeperiod=period)
            df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)
            # 上下柜
            # bullWidth
            df0["bullwidth"] = width = (4 * std / ma * 100).fillna(0)
            # 近三分钟width变动
            df0["widthDelta"] = wd1 = ta.MA(width - width.shift(1), timeperiod=3).fillna(0)
            df0["widthDelta2"] = wd2 = wd1 - wd1.shift(1)
            df0["wd2m"] = wd2 * wd2.shift(1)

            dif, dea, macd = ta.MACD(close, fastperiod=int(period / 3), slowperiod=period, signalperiod=5)
            df0["macd2d"] = macd - macd.shift(1)
            df0["macd2dm"] = (macd - macd.shift(1)) * (macd.shift(1) - macd.shift(2))

            df0["ma5"] = ma5 = ta.MA(close, timeperiod=5)
            df0["macd"] = macd5 = (ma5 - ma) / ma * 100
            df0["macdm"] = macd5 - macd5.shift(1)
            df0["macd2"] = macd5 * macd5.shift(1)

        # 相对波动
        df0['delta'] = (p_l - p_h) / df0['std']
        # 协整
        coint = sm.tsa.stattools.coint(df0["close"], df1["close"])
        # x相关性
        relative = per(df0["close"], df1["close"])

        # 循环 scale
        docs = []
        for scale in self.scaleList:
            for conds in self.iterCond():

                uid = self.uidKey % (
                    '_'.join(self.codes), str(period), str(scale), self.klineType[:-1], conds)

                self.stopTimeDiff = self.stopTimeLine * period * int(self.klineType[:-1])
                # 计算高低线值
                df0["top"], df0["lower"] = df0['ma'] + (scale-self.scaleDiff) * df0['std'], df0['ma'] - (scale+self.scaleDiff) * df0['std']

                # df0.fillna(0, inplace=True)
                tot = self.detect(df0, df1, period=period, uid=uid)
                if tot is not None and tot['amount'] != 0:
                    tot.update(
                        {
                            "scale": scale,
                            "code": self.codes[0],
                            "code1": self.codes[1],
                            "period": period,
                            "uid": uid,
                            "relative": relative,
                            "shift": (p_l - p_h).mean(),
                            "coint": 0 if np.isnan(coint[1]) else coint[1],
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
            return {
                "count": int(len(docs) / 4),
                "amount": self.iniAmount,
                "price": res['rel_price'].mean(),
                "income": res["income"].sum(),
                "std": res['rel_std'].mean(),
                "delta": res['delta'].mean(),
                "timediff": int(0 if np.isnan(diff) else diff)
            }
        else:
            return None

    # 核心策略部分
    def stageApply(self, df0, df1, period=15, uid=''):
        isOpen, isRise, preDate, prePrice = 0, 0, None, 0
        doc, doc1, docs = {}, {}, []
        sline, wline = self.stopTimeDiff, self.widthDeltaLine

        ma, p_l, p_h, top, lower, std, delta, width, wd1, wd2, wd2m, macd2d, macd2dm = (df0[key] for key in
                                                                                        "ma,p_l,p_h,top,lower,std,delta,bullwidth,widthDelta,widthDelta2,wd2m,macd2d,macd2dm".split(
                                                                                            ","))

        for i in range(period, len(df0)):

            isRun, isstop = False, 0
            #  开仓2
            if delta[i] > self.deltaLine or np.isnan(ma[i]): continue

            isRun, isstop = False, 0
            #  开仓2
            if delta[i] > self.deltaLine or np.isnan(ma[i]): continue

            cond1, cond2 = False, False
            if wline > 0:
                # 布林宽带变化率
                cond1 = (wd1[i] < wline) and (wd2[i] < (wline / 2))
                # 最大值
                cond2 = wd2m[i] < 0

            if isOpen == 0:
                # 突变状态开始
                # 大于上线轨迹
                if p_h[i] >= top[i] and cond1:
                    isOpen = -1
                    isRun = True

                elif p_l[i] <= lower[i] and cond1:
                    isOpen = 1
                    isRun = True


                elif self.isEverOut(p_h, top, std, 1, i) and not cond1 and (
                        (macd2dm[i] < 0 and macd2d[i] > 0) or cond2):
                    isOpen = -2
                    isRun = True

                elif self.isEverOut(p_l, lower, std, -1, i) and not cond1 and (
                        (macd2dm[i] < 0 and macd2d[i] < 0) or cond2):
                    isOpen = 2
                    isRun = True
            # 平仓
            else:
                sign, dline = isOpen / abs(isOpen), - self.scaleDiff2 * std[i] / 2

                cond3 = (sign * ((p_h[i] if isOpen > 0 else p_l[i]) - ma[i]))
                #
                if cond3 >= -dline and not cond1 and (cond2 or (macd2dm[i] < 0 and sign * macd2d[i] < 0)):
                    isOpen, isstop = 0, 2
                    isRun = True

                elif cond3 >= 0 and cond1:
                    isOpen = 0
                    isRun = True


                # 超时止损
                elif sline > 0 and self.preNode is not None:
                    tdiff = self.Rice.timeDiff(str(self.preNode[0]['createdate']), str(df0.index[i]), quick=sline)
                    if tdiff > sline and cond3 >= -dline and cond2:
                        isOpen, isstop = 0, 1
                        isRun = True

            if isRun:
                doc, doc1 = self.order(df0.iloc[i], df1.iloc[i], isOpen, uid, isstop=isstop)
                if doc1 is not None:
                    docs.append(doc)
                    docs.append(doc1)
        return docs

    batchId = None

    def adjustOrder(self, n0, date):

        preCode = ''
        for c in self.adjustDates:
            if c[1] == date:
                preCode = c[2]
                break

        s = str(date).split(" ")[0]
        df1 = self.Rice.kline([preCode], period='1d', start=public.getDate(diff=-2, start=s), end=s, pre=0)
        oldP = df1[preCode]['open'].values[-1]
        newP = n0['open']

        # 调仓卖出
        doc = copy.deepcopy(self.preNode[0])
        sign = np.sign(self.preNode[0]['mode'])
        pp = self.preNode[0]['price']

        doc['price'] = oldP
        doc['isopen'] = 0
        doc['mode'] = -doc['mode']
        doc['isstop'] = 6
        doc['createdate'] = date
        doc['income'] = sign * (oldP - pp - 2 * sign * self.shift[0]) * doc["vol"] - doc["fee"]

        self.records.append(doc)

        doc1 = copy.deepcopy(self.preNode[0])
        doc1['createdate'] = date
        doc1['mode'] = int(6 * sign)
        doc1['price'] = newP
        doc1['isopen'] = 1
        doc1['batchid'] = self.batchId = uuid.uuid1()

        self.records.append(doc1)
        self.preNode = [doc1]

    def order(self, n0, n1, mode, uid, isstop=0):
        # baseInfo 配置文件，查询ratio 和 每手吨数

        b0, b1 = self.baseInfo[self.mCodes[0]], self.baseInfo[self.mCodes[1]]
        if mode != 0:
            self.batchId = uuid.uuid1()

        # 交易量
        v0 = self.iniVolume * b0["contract_multiplier"]
        v1 = int(v0 * n0["close"] / n1["close"] / b1["contract_multiplier"]) * b1["contract_multiplier"]

        # 费率
        fee0 = (self.iniVolume * b0["ratio"]) if b0["ratio"] > 0.5 else ((b0["ratio"]) * n0["close"] * v0)
        fee1 = (v1 / b1["contract_multiplier"] * b1["ratio"]) if b1["ratio"] > 0.5 else (b1["ratio"] * n1["close"] * v1)

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
            "bullwidth": n0["width"],
            #"widthDelta": n0["widthDelta"],
            "delta": n0["rsi"],
            "batchid": self.batchId,
            'p_l': n0["p_l"],
            'p_h': n0["p_h"],
            "diff": 0 if mode!=0 else self.Rice.timeDiff(str(self.preNode[0]['createdate']), str(n0["datetime"])),
            "uid": uid
        }

        doc1 = {}
        doc1.update(doc)
        doc1.update({
            "code": self.codes[1],
            "price": n1["close"],
            "vol": self.preNode[1]["vol"] if self.preNode else v1,
            "mode": -mode if not self.preNode else -self.preNode[1]["mode"],
            "fee": fee1,
        })

        if mode == 0 and self.preNode:
            p0, p1 = self.preNode[0], self.preNode[1]
            sign = p0["mode"] / abs(p0["mode"])

            doc["income"] = sign * (n0["close"] - p0["price"] - 2 * sign * self.shift[0]) * p0["vol"] - p0[
                "fee"]
            doc1["income"] = - sign * (n1["close"] - p1["price"] + 2 * sign * self.shift[1]) * p1["vol"] - p1[
                "fee"]
            doc["diff"] = int(public.timeDiff(str(n0['datetime']), str(p0['createdate'])) / 60)
            self.preNode = None

        else:
            doc["income"] = -doc["fee"]
            doc1["income"] = -doc1["fee"]
            self.preNode = (doc, doc1)
        return doc, doc1


    def calcIncome(self, n0, p0, df0):
        # 计算收益，最大/最小收益
        high = df0[(p0['createdate'] <= df0.index) & (df0.index <= n0['datetime']) ]['high'].max()
        low = df0[(p0['createdate'] <= df0.index) & (df0.index <= n0['datetime'])]['low'].min()
        close = n0["close"]
        sign = p0["mode"] / abs(p0["mode"])

        # 收入
        income = sign * (close - p0["price"] - 2 * sign * self.shift[0]) * p0["vol"] - p0["fee"]
        # 最大收入
        highIncome =  sign * (high if sign > 0 else low - p0["price"] - 2 * sign * self.shift[0]) * p0["vol"] - p0["fee"]
        # 最大损失
        lowIncome = sign * (high if sign < 0 else low - p0["price"] - 2 * sign * self.shift[0]) * p0["vol"] - p0["fee"]

        return income, highIncome, lowIncome

def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = train_future_klineExpect()
        obj.Pool()


if __name__ == '__main__':
    main()
