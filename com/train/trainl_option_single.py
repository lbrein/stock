# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ----- 期货训练模型 - 单
"""

from com.base.stat_fun import per, fisher
from com.base.public import public, logger
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
import numpy as np
from com.object.obj_entity import  train_future, train_total, future_baseInfo
import uuid
from com.data.interface_Rice import interface_Rice
import itertools
import time
from multiprocessing import Pool, Manager

# 回归方法
class model_future_train_rice(object):
    """

    """
    def __init__(self):
        # 费率和滑点
        self.iniAmount = 100000
        self.baseInfo = {}
        self.saveDetail = True

        self.stopLine = 0
        self.klinePeriodList = ['5m', '15m', '30m']
        self.periodList = [15, 30, 45, 60]
        self.scaleList=[2.0, 2.5]
        self.widthDeltaLineList = [-1, 0.05]

        self.deltaLine = 0.5
        self.shiftRatio = 0.527
        self.ratio = 0.00003

        self.startDate = public.getDate(diff=-60)  # 60天数据回测
        self.endDate = public.getDate()
        self.total_tablename = 'train_total_2'
        self.detail_tablename = 'train_future_3'
        self.uidKey = '%s_%s_%s_%s_%s_ETF_' + str(self.shiftRatio)

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

        # 交易量大的，按价格排序, 类型:turple,第二位为夜盘收盘时间
        pool = Pool(processes=4)
        shareDict = Manager().list([])

        #Rice = interface_Rice()
        #rs = Rice.allCodes(type='ETF')
        #print(rs[rs["order_book_id"].str.find('510050')>-1])


        for rs in ['510050.XSHG']:
            for klinePeriod in self.klinePeriodList:
                #self.start([rs], klinePeriod , time0, shareDict)
                try:
                    pool.apply_async(self.start, ([rs], klinePeriod, time0, shareDict))
                    pass
                except Exception as e:
                    print(e)
                    continue

        pool.close()
        pool.join()

    def start(self, codes, klinePeriod, time0, shareDict):
        print("子进程启动:", codes, time.time()-time0)

        self.Rice = interface_Rice()
        self.Train = train_future()
        self.Base = future_baseInfo()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename
        self.Train.tablename = self.detail_tablename

        # 主力合约
        self.mCodes = codes
        self.klinePeriod = klinePeriod

        # 查询获得分钟K线
        dfs = self.Rice.kline(codes, period= klinePeriod, start=self.startDate, end=self.endDate)

        print("kline load", self.mCodes)
        # 查询获得配置 - 费率和每手单量
        self.shift = 0.001 * self.shiftRatio

        # 分参数执行
        res = []
        for period in self.periodList:
            for scale in self.scaleList:
                for deltaline in self.widthDeltaLineList:
                    self.widthDeltaLine = deltaline
                    doc = self.total(dfs[codes[0]], scale=scale, period=period)

                    if doc is None: continue
                    doc.update({
                        "code": codes[0],
                    #    "code1": codes[1],
                    })
                    logger.info((doc,  " time:", time.time()-time0))
                    res.append(doc)
                # 结果写入共享阈
                #shareDict.append(doc)

        self.Total.insertAll(res)

    def total(self, df0, scale=1, period=60):
        uid = self.uidKey % (self.mCodes[0], str(period), str(scale), self.klinePeriod, str(self.widthDeltaLine))

        df0["rel_price"] = close = df0["close"]
        df0["datetime"] = df0.index
        p_l = close + self.shift
        p_h = close - self.shift

        num = len(df0)

        ma = ta.MA(close[0:num], timeperiod=period)
        df0["rel_std"] = sd = ta.STDDEV(close, timeperiod=period, nbdev=1)

        # 上下柜
        top, lower = ma + scale * sd, ma - scale * sd
        #self.shift[0] = sd.values[-1] / close.values[-1]

        # bullWidth
        df0["bullwidth"] = width =  4 * sd / sd.mean()
        df0["delta"] =  delta = (self.shift /sd).fillna(0)
        # 近三分钟width变动
        df0["widthdelta"] = widthdelta=ta.MA(width - width.shift(1), timeperiod = 3).fillna(0)
        #df0["widthdelta"] = widthdelta = (width - width.shift(1)).fillna(0)
        widthdelta2 = widthdelta - widthdelta.shift(1)
        w2 = widthdelta2 * widthdelta2.shift(1)

        isOpen, preDate, prePrice = 0, None, 0
        doc, docs =  {}, []

        self.index = uuid.uuid1()
        for i in range(num):
            isRun = False
            c_datetime = df0.index[i]
            if i < period and np.isnan(ma[i]): continue

            # 开仓
            if isOpen == 0:
                # 开市前分钟不再开仓
                if delta[i] > self.deltaLine : continue

                if self.widthDeltaLine != -1:
                    cond1 =  (widthdelta[i] > self.widthDeltaLine) or ( widthdelta2[i] > (self.widthDeltaLine / 2))
                else:
                    cond1 = False

                # 大于上线轨迹
                if ((p_l[i] > top[i]) and not cond1)  :
                    isOpen = -1
                    isRun = True

                elif ((p_h[i] < lower[i]) and not cond1) :
                    isOpen = 1
                    isRun = True


                elif cond1 and w2[i] < 0 and (p_h[i] > top[i]) :
                     isOpen = -2
                     isRun = True

                elif (cond1 and w2[i] < 0) and (p_l[i] < lower[i]):
                     isOpen = 2
                     isRun = True

            # 平仓
            else:
               # 回归ma则平仓  或  超过24分钟 或到收盘时间 强制平仓
               if  (isOpen * ((p_h[i] if isOpen>0 else p_l[i]) - ma[i])) >= 0:
                     isOpen = 0
                     isRun = True
               # 止损

               #elif ((isOpen==2) and (p_l[i] < top[i])) or ((isOpen==-2) and (p_h[i] > lower[i])):
               #elif (abs(isOpen)==2) and (w2[i] < 0) :
               #      isOpen = 0
               #      isRun = True

            if isRun:
                doc = self.order(df0.iloc[i], isOpen)
                if doc:
                    doc["uid"] =uid
                    docs.append(doc)


        res = pd.DataFrame(docs)

        if self.saveDetail:
            print(abs(widthdelta).mean(), sd.mean())
            self.Train.insertAll(docs)

        if len(res) > 0:
            return {
                "scale": scale,
                "period": period,
                "count": len(res),
                "amount": (doc["price"] * doc["vol"]) if doc is not None else 0 ,
                "income": res["income"].sum(),
                "uid": uid,
                #"delta":(self.shift/sd).mean(),
                "shift": widthdelta.mean() ,
                "std": sd.mean(),
                "createdate": public.getDatetime()
            }
        else:
            return None

    preNode, index = None, None
    def order(self, n0,  mode):
        # baseInfo 配置文件，查询ratio 和 每手吨数
        #b0 = self.baseInfo[self.mCodes[0]]
        # 交易量
        v0 = round(self.iniAmount / n0["close"], 0)

        # 费率
        fee0 = self.ratio * n0["close"] * v0

        doc = {
            "createdate": n0["datetime"],
            "code": self.mCodes[0],
            "price": n0["close"],
            "vol": v0,
            "mode":  mode if not self.preNode else -self.preNode["mode"],
            "isopen": 0 if mode == 0 else 1,
            "fee": fee0,
            "income": 0,
            "rel_price": n0["rel_price"],
            "rel_std": n0["rel_std"],
            "batchid": self.index,
            "bullwidth": n0["bullwidth"],
            "delta": 0,
            "widthDelta": n0["widthdelta"],
        }

        if mode == 0 and self.preNode:
            p0 = self.preNode
            sign = p0["mode"]/abs(p0["mode"])
            doc["income"] = sign * (n0["close"] - p0["price"] - sign * 2 * self.shift) * p0["vol"] - p0["fee"]
            self.preNode = None
            self.index = uuid.uuid1()

        else:
            doc["income"] = -doc["fee"]
            self.preNode = doc

        return doc

def main():
    obj = model_future_train_rice()
    #obj.draw()
    obj.Pool()

if __name__ == '__main__':
    main()

