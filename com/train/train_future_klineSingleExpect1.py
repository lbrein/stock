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
from com.model.train_future_2_tick_new import model_future_quickTickTest
import itertools
import time
import uuid
from multiprocessing import Pool, Manager
import statsmodels.api as sm  # 协整
import seaborn as sns


# 回归方法
class train_future_singleExpect(object):
    """

    """
    iniAmount = 500000  # 单边50万

    def __init__(self):
        # 费率和滑点
        self.baseInfo = {}
        self.saveDetail = False  # 是否保存明细
        self.periodList = [15, 30, 60]  # 窗体参数
        self.scaleList = [1.8, 2.0, 2.5]
        self.shiftScale = 0.527  # 滑点模拟系数
        # k线时间
        self.klineTypeList = ['5m', '15m']
              # 起始时间
        self.startDate = public.getDate(diff=-60)  # 60天数据回测

        self.endDate = public.getDate()
        self.total_tablename = 'train_total'
        self.method = 'simTick'
        self.stage = 'single'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method + "_" + self.stage

    def top(self, mode=0):
        if mode == 0:
            Total = train_total()
            Total.tablename = self.total_tablename
            rs = Total.last_top(10, maxsames=10,  minRate = 0.02)
            return [l[0] for l in rs]
        elif mode==1:
            Base = future_baseInfo()
            return [ n[0] for n in  Base.all(vol=300)]

    def Pool(self):
        time0 = time.time()

        pool = Pool(processes=4)
        shareDict = Manager().list([])

        Base = future_baseInfo()
        # 交易量大的，按价格排序, 类型:turple,第二位为夜盘收盘时间
        lists = Base.all(vol=300)
        print(len(lists), lists)

        for rs in lists:
            # 检查时间匹配
            codes = [rs[0]]
            for kt in self.klineTypeList:
                #self.start(codes, time0, kt, shareDict)
                try:
                    pool.apply_async(self.start, (codes, time0,  kt , shareDict))
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

        cs0 = self.baseInfo[mCodes[0]]

        # 计算tick 导致的滑点
        sh = [self.baseInfo[d + '88']['tick_size'] for d in codes]
        self.shift = [sh[i] * self.shiftScale for i in range(len(sh))]

        # 子进程共享类
        self.Rice = tick_csv_Rice()
        self.Rice.setTimeArea(cs0["nightEnd"])
        self.Train = train_future()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename

        # 查询获得N分钟K线
        dfs_l = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=60)
        # 获得1分钟K线作为滚动线
        dfs = self.Rice.kline(mCodes, period='1m', start=self.startDate, end=self.endDate, pre=0)

        # 按时间截取并调整
        # dfs= self.dateAdjust(codes, dfs, sh)
        print('kline load:', mCodes, len(dfs[mCodes[0]]))

        # 根据配置文件获取最佳交易手数对
        self.iniVolume = round(self.iniAmount / cs0["lastPrice"] / cs0["contract_multiplier"], 0)

        # 分参数执行
        results = []
        for period in self.periodList:
            docs = self.total(dfs, dfs_l, period=period)

            if docs is None or len(docs) == 0:  continue
            logger.info((self.codes, period, self.klineType, len(docs), " time:", time.time() - time0))
            results.extend(docs)

            # 结果写入共享阈
        self.Total.insertAll(results)

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
        df0 = dfs[self.mCodes[0]]
        df0["rel_price"] = close = df0["close"]
        df0["datetime"] = df0.index

        s0 = self.shift[0]
        p_l = df0["p_l"] = (df0["close"] + s0)
        p_h = df0["p_h"] = (df0["close"] - s0)

        close2 = dfs2[self.mCodes[0]]["close"]

        # 双线模拟计算k线数据的tick行为
        df0['ma'] = ma = df0.apply(lambda row: self.k_ma(row['datetime'], row['rel_price'], close2, period, 0), axis=1)
        df0['std'] = std = df0.apply(lambda row: self.k_ma(row['datetime'], row['rel_price'], close2, period, 1),
                                     axis=1)

        # bullWidth
        df0["bullwidth"] = width = (4 * std / ma * 100).fillna(0)

        # 近三分钟width变动
        df0["widthDelta"] = ta.MA(width - width.shift(1), timeperiod=3).fillna(0)
        df0["delta"] = (p_l - p_h) / std

        # 循环 scale
        docs = []
        for scale in self.scaleList:
            uid = self.uidKey % (
                "_".join(self.codes), str(period), str(scale), self.klineType[:-1], str(self.shiftScale))

            df0["top"], df0["lower"] = ma + scale * std, ma - scale * std

            df0.fillna(0, inplace=True)
            tot = self.detect(df0, period=period, uid=uid)

            if tot is not None and tot['amount'] != 0:
                tot.update(
                    {
                        "scale": scale,
                        "code": self.codes[0],
                        "period": period,
                        "uid": uid,
                        "shift": (p_l - p_h).mean(),
                        "method":self.method,
                        "stage":self.stage,
                        "createdate": public.getDatetime(),

                    }
                )
                docs.append(tot)

        return docs

    # 核心策略部分
    def detect(self, df0, period=15, uid=''):
        isOpen, preDate, prePrice = 0, None, 0
        doc, docs =  {}, []
        # df0.reset_index(drop=True)
        ma, p_l, p_h, top, lower = (df0[key] for key in "ma,p_l,p_h,top,lower".split(","))

        i = period
        while True:
            isRun = False
            # if np.isnan(ma[i]): continue

            # 开仓
            if isOpen == 0:
                # 大于上线轨迹
         # skip i
                if p_h[i] >= top[i]:
                    isOpen = -1
                    isRun = True

                elif p_l[i] <= lower[i]:
                    isOpen = 1
                    isRun = True
            # 平仓
            else:
                # 回归ma则平仓  或  超过24分钟 或到收盘时间 强制平仓
                if (isOpen * ((p_h[i] if isOpen == 1 else p_l[i]) - ma[i])) >= 0:
                    isOpen = 0
                    isRun = True
                # 止损

            if isRun:
                doc = self.order(df0.iloc[i],  isOpen, uid)
                if doc is not None:
                    docs.append(doc)

            i += 1
            if i >= len(df0):
                break

        res = pd.DataFrame(docs)

        if len(res) > 0:
            if self.saveDetail:
                print(self.Train.tablename, len(res), 'save')
                self.Train.insertAll(docs)

            diff = res[res['diff'] > 0]['diff'].mean()
            return {
                "count": int(len(docs) / 2),
                "amount": (doc["price"] * doc["vol"]) if doc is not None else 0,
                "price": doc["price"] ,
                "income": res["income"].sum(),
                "std": res['rel_std'].mean(),
                "delta": res['delta'].mean(),
                "timediff": int(0 if np.isnan(diff) else diff)
            }
        else:
            return None

    batchId = None

    def order(self, n0,  mode, uid):
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
            "vol": self.preNode["vol"] if self.preNode else v0,
            "mode": mode if not self.preNode else -self.preNode["mode"],
            "isopen": 0 if mode == 0 else 1,
            "fee": fee0,
            "income": 0,
            "rel_price": n0["rel_price"],
            "rel_std": n0["std"],
            "bullwidth": n0["bullwidth"],
            "widthDelta": n0["widthDelta"],
            "delta": n0["delta"],
            "batchid": self.batchId,
            "atr": n0["atr"],
            'p_l': n0["p_l"],
            'p_h': n0["p_h"],
            "diff": 0,
            "uid": uid
        }

        if mode == 0 and self.preNode:
            p0 = self.preNode
            doc["income"] = p0["mode"] * (n0["close"] - p0["price"] - 2 * p0["mode"] * self.shift[0]) * p0["vol"] - p0[
                "fee"]
            doc["diff"] = int(public.timeDiff(str(n0['datetime']), str(p0['createdate'])) / 60)
            self.preNode = None

        else:
            doc["income"] = -doc["fee"]
            self.preNode = doc
        return doc


def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = train_future_singleExpect()
        obj.Pool()


if __name__ == '__main__':
    main()
