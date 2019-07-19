# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ---
    > 与 tick 配套对比计算的 kline
    > 单一商品期货品种
    > 多条件
"""

from com.base.stat_fun import per, fisher
from com.base.public import public, logger
import pandas as pd
import talib as ta
import numpy as np
from com.object.obj_entity import train_future, train_total, future_baseInfo

import time
import uuid
from multiprocessing import Pool, Manager
import statsmodels.api as sm  # 协整
import seaborn as sns
from com.model.train_future_klineSingleExpect import train_future_single_bull

# 回归方法
class train_future_single_multiCond(train_future_single_bull):
    """

    """
    iniAmount = 500000  # 单边50万

    def __init__(self):
        # 费率和滑点
        self.baseInfo = {}
        self.saveDetail = False  # 是否保存明细
        self.periodList = [15, 30, 45, 60]  # 窗体参数

        self.scaleList = [1.5, 1.8, 2.0, 2.5, 2.8]
        self.shiftScale = 0.527  # 滑点模拟系数

        # k线时间
        self.klineTypeList = ['5m', '15m']
        self.uidKey = "%s_%s_%s_%s_kline_%s"
        # 起始时间
        self.startDate = public.getDate(diff=-30)  # 60天数据回测

        self.endDate = public.getDate()
        self.total_tablename = 'train_total'

    def Pool(self):
        time0 = time.time()

        pool = Pool(processes=4)
        shareDict = Manager().list([])

        Base = future_baseInfo()
        # 交易量大的，按价格排序, 类型:turple,第二位为夜盘收盘时间
        lists = Base.all(vol=300)
        print(len(lists), lists)

        #for rs in list(itertools.combinations(lists, 1)):
        for rs in lists:
            # 检查时间匹配
            #if rs[0][1] != rs[1][1]: continue
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
                        "createdate": public.getDatetime()
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


def main():
    obj = train_future_single_multiCond()
    obj.Pool()

if __name__ == '__main__':
    main()
