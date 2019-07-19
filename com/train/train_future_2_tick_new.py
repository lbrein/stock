# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

      期货多币对 多进程对冲交易程序

      使用tick进行回测

"""

from com.base.public import public, logger
import pandas as pd
import talib as ta
from com.object.obj_entity import train_future, future_baseInfo
from com.data.interface_Rice import tick_csv_Rice
import time
from multiprocessing import Pool, Manager
import uuid
import os
import profile
import re


# 回归方法
class model_future_quickTickTest(object):
    bullPeriod = 10
    iniVolume = 10
    iniAmount = 500000  # 单边100万
    future_Map = [
        # ["AG", "AU", 15, 2, 5],  # 交易商品1，交易商品2，bullPeriod, stdAmount,iniVolume
        ["SR", "I", 30, 2, 5],
        ["RU", "BU", 15, 2, 5],
        ["J", "I", 15, 2, 5],
        ["OI", "P", 30, 2, 5],
        ["MA", "ZC", 15, 2, 5],
        ["B", "M", 15, 2, 5],
        # ["L", "C", 15, 2, 5]  ,
    ]
    action = [('isdeal', '_1'),
              ('isnew', '_2')
              ]
    #
    cvsParam = {
        'basePath': 'E:/stock/csv/eight/' if os.name == 'nt' else "/root/home/stock/csv/",
        'cvs_start': '2018-08-01 09:00:00',
        'cvs_end': '2018-08-31 15:00:00',
        'dd': ["1901"]}

    def __init__(self):
        self.Record = None
        self.Rice = None
        self.mCodes = None
        self.Base = None
        self.config = {}

        # 参数
        self.periods = [30, 60]  # 窗口宽度
        self.scales = [ 2.0, 2.5]  # 标准差倍数
        self.klinePeriods = [1]  # K线类型
        self.stopLine = 0  # 止损幅度
        self.stopMinute = 0  # 止损时间
        self.shiftScale = 2.0  # 价差与 tick_size 比率
        self.deltaLimits = [0.8]  # 价差/标准差 限制

        aa = self.action[0]
        self.tickType = aa[0]  # tick 过滤类型: none: 不过滤 isnew-分钟 isdea-5秒线
        #
        self.tablename = 'train_future%s' % aa[1]  # 结果存储表
        self.uidKey = "%s_%s_%s_%s_%s_aa".replace('aa', aa[0])  # uid格式

    def getExistCodes(self):
        Record = train_future()
        Record.tablename = self.tablename
        long = len(self.uidKey)
        return Record.exitCodes(long=long - 2)

    def pool(self):
        pool = Pool(processes=4)
        shareDict = Manager().list([])

        Rice = tick_csv_Rice()
        Rice.setParam(self.cvsParam)
        ticks = Rice.exitTicks()
        print(ticks)

        for rc in ticks:
            #     if rc in exists:
            #         continue
            map = rc.split("_")
            for kp in self.klinePeriods:
                #self.start(map, kp, shareDict)
                try:
                    pool.apply_async(self.start, (map, kp, shareDict))
                    pass
                except Exception as e:
                    print(e)
                    continue
        # break

        pool.close()
        pool.join()

    def start(self, map, kPeriod, shareDict):
        # 每个进程单独初始化对象
        self.time0 = time.time()
        self.Record = train_future()
        # 数据库存储
        self.Record.tablename = self.tablename

        self.Rice = tick_csv_Rice()
        self.Rice.setParam(self.cvsParam)
        self.Rice.shareDict = self.shareDict = shareDict
        self.Rice.kPeriod = self.kPeriod = kPeriod

        self.Base = future_baseInfo()
        self.timeArea = [("09:00:00", "11:30:00"), ("13:30:00", "15:00:00")]
        self.codes = map[0:2]

        # 查询基础配置
        for doc in self.Base.getInfo(self.codes):
            self.config[doc["code"]] = doc

        c0, c1 = self.config[self.codes[0]], self.config[self.codes[1]]
        # 设置交易对的收盘时间,用于计算收盘时间并计算止损
        self.Rice.setTimeArea(doc["nightEnd"])

        # 根据配置文件获取最佳交易手数对
        self.iniVolume = round(self.iniAmount / c0["lastPrice"] / c0["contract_multiplier"], 0)

        # 设置shift区间
        [t, c] = [[self.config[d][key] for d in self.codes] for key in ["tick_size", "lastPrice"]]
        self.shift = ((c[0] + t[0]) / (c[1] - t[1]) - (c[0] - t[0]) / (c[1] + t[1])) / self.shiftScale

        logger.info(("model_future_detect start: %s " % "_".join(self.codes), self.uidKey, self.Rice.TimeArea))

        # 启动进程
        ticks = self.Rice.getTicks(self.codes)
        # tick类型选择
        if self.tickType != 'none':
            ticks = ticks[ticks[self.tickType] == 1]

        ticks["p_l"] = ticks["a1"] / ticks["n_b1"]
        ticks["p_h"] = ticks["b1"] / ticks["n_a1"]

        indexR, records = 0, []
        # 及时K线
        for index, row in ticks.iterrows():
            #print(row["datetime"])
            res = self.onTick(row, isdelay=self.tickType != 'isnew')

            if len(res) > 0:
                records.extend(res)

            #  阶段性存入到数据库，并清空
            if len(records) > 50:
                self.Record.insertAll(records)
                indexR += len(records)
                print(public.getDatetime(), " insert into record:", self.codes, self.kPeriod, indexR)
                records = []

        # 将剩下的写入数据库
        if len(records) > 0:
            self.Record.insertAll(records)

    indexI = 0
    isOpen, isNotStop, preNode, batchId, cacheParam = {}, {}, {}, {}, {}

    def onTick(self, tick, isdelay=False):
        # 获取最新K线
        now = tick["datetime"]
        if isdelay:
            now = re.sub(':00$', ':01', tick['datetime'])

        dfs = self.Rice.getCvsKline(self.codes, now=now, isNew=int(tick["isnew"]))
        # 多重计算
        res = []
        for p in self.periods:
            for s in self.scales:
                for de in self.deltaLimits:
                    self.stdAmount = s
                    self.bullPeriod = p
                    self.deltaLimit = de

                    # 单元uid
                    self.uid = self.uidKey % (
                        "_".join(self.codes), str(self.bullPeriod), str(s), str(de), str(self.kPeriod))
                    # 计算参数
                    if self.uid not in self.isOpen.keys():
                        self.isOpen[self.uid] = 0  # 操作状态 0-平 1-买开 -1:卖开
                        self.preNode[self.uid] = None
                        self.batchId[self.uid] = uuid.uuid1()
                        self.isNotStop[self.uid] = (1, 0)  # 是否止损反向盘 1-否 -1-是
                        self.cacheParam[self.uid] = {}

                    # 计算参数
                    param = self.paramCalc(dfs, tick)
                    # 比较
                    docs = self.orderCheck(tick, param)
                    if docs:
                        res.extend(docs)
        return res

    def cache(self, key, cur):
        cache = self.cacheParam[self.uid]
        if key in cache.keys() and cur['isnew'] == 0:
            return cache[key]
        return None

    def cache_set(self, key, df):
        self.cacheParam[self.uid][key] = df
        return df

    # 计算布林参数
    def paramCalc(self, dfs, cur):
        # 分钟线
        # 周期和标准差倍数
        period, scale = self.bullPeriod, self.stdAmount
        size = period + 5

        df0, df1 = dfs[self.codes[0]][-size:], dfs[self.codes[1]][-size:]
        # df0, df1 = dfs[self.codes[0]], dfs[self.codes[1]]
        # 计算相关参数
        # 每分钟计算一次
        close = self.cache('close', cur)

        if close is None:
            close = self.cache_set('close', df0["close"] / df1["close"])

        # 添加最新
        close = close.append(pd.Series(cur["close"]))
        #
        ma = ta.MA(close, timeperiod=period).fillna(0)
        sd = ta.STDDEV(close, timeperiod=period, nbdev=1)
        top, lower = ma + scale * sd, ma - scale * sd

        # 计算其他参数值，采用一分钟计算一次
        (width, widthDelta) = (self.cache(key, cur) for key in ["width", "widthDelta"])
        if width is None:
            # 布林宽度
            width = self.cacheParam[self.uid]["width"] = (2 * scale * sd) / ma * 100
            # 布林变化
            widthDelta = self.cacheParam[self.uid]["widthDelta"] = ta.MA(width - width.shift(1), timeperiod=3)
            # 真实波幅
            # min, max = ta.MINMAX(close, timeperiod=period)
            # atr = self.cacheParam[self.uid]["atr"] = ta.WMA((max.dropna() - min.dropna()), timeperiod=5)

        return {
            "ma": ma.values[-1],
            "top": top.values[-1],
            "lower": lower.values[-1],
            "std": sd.values[-1],
            "delta": (cur["p_l"] - cur["p_h"]) / sd.values[-1],
            "shift": (cur["p_l"] - cur["p_h"]) / self.shift,
            "width": width.values[-1],
            "widthDelta": widthDelta.values[-1],
            # "atr": atr.values[-1],
            "datetime": cur["datetime"]
        }

    # 检查交易条件
    def orderCheck(self, cur, param):

        isRun = False
        # 开仓
        if self.isOpen[self.uid] == 0:
            """
            # 休市前5分钟不再开仓
            isNear = False
            for tt in ["09:30:00", "13:00:00", "21:00:00"]:
                dim =  public.timeDiff(c_date+" "+tt, c_datetime)/60
                if  dim < 0 and dim > -30:
                    isNear = True
            """
            # 过滤delta高
            if param["delta"] <= self.deltaLimit:
                # 大于上线轨迹
                # if cur["p_h"] > param["top"] and self.isStop(-1):
                if cur["p_h"] > param["top"]:
                    self.isOpen[self.uid] = -1
                    isRun = True



                # 低于下轨线
                elif cur["p_l"] < param["lower"]:
                    self.isOpen[self.uid] = 1
                    isRun = True

        # 平仓
        else:
            # 回归ma则平仓  或 超过24分钟 或到收盘时间 强制平仓
            io = self.isOpen[self.uid]
            ph, pl, pm = cur["p_h"], cur["p_l"], param["ma"]

            # 强制平仓反买之后再平仓（采用macd和布林中线)

            # 自然平仓
            if io * ((ph if io == 1 else pl) - pm) >= 0:
                # 过滤delta高的
                if param["delta"] < 1.5 * self.deltaLimit:
                    self.isNotStop[self.uid] = (1, io)
                    self.isOpen[self.uid] = 0
                    isRun = True

            # 止损，平掉已有开仓，
            elif self.preNode[self.uid] and self.stopMinute > 0:
                p = self.preNode[self.uid][0]
                timeDiff = self.Rice.timeDiff(p["createdate"], cur['datetime'])
                # 超时止损
                if timeDiff > self.stopMinute:
                    self.isNotStop[self.uid] = (-1, io)  # 强制平仓
                    self.isOpen[self.uid] = 0
                    isRun = True

            """
               if  (io * (param["close"] - p["rel_price"])) < 0:
                   if param["delta"] < (1.5 * self.deltaLimit):
                       self.isNotStop[self.uid] = (-1, io) # 强制平仓
                       print(self.uid, "stop", self.isNotStop[self.uid])
                       self.isOpen[self.uid] = 0
                       isRun = True

            
                       # 开反方向盘 防止持续开盘
                       self.isOpen[self.uid] = -pre
                       res1 = self.order(cur, self.isOpen[self.uid], param)
                       for doc in res1:
                            res.append(doc)
                       return res
            """

        self.indexI += 1
        if self.indexI % 20000 == 0:
            print(self.indexI, self.uid, self.isOpen[self.uid], param, time.time() - self.time0)
            print('')

        if isRun:
            res = self.order(cur, self.isOpen[self.uid], param)
            return res

        return None

    # 检查是否止损
    def isStop(self, mode):
        ss = self.isNotStop[self.uid]
        if ss[0] == -1 and ss[1] == mode:
            return False
        else:
            self.isNotStop[self.uid] = (1, 0)
            return True

    def timeDiff(self, preDate, curDate=None):
        if curDate is None:
            curDate = public.getDateTime()

    # 下单
    def order(self, cur, mode, param):
        # 使用uuid作为批次号
        if self.isOpen[self.uid] == mode and mode != 0:
            self.batchId[self.uid] = uuid.uuid1()

        # print(self.uid, mode, self.batchId[self.uid])

        # future_baseInfo 参数值
        b0, b1 = self.config[self.codes[0]], self.config[self.codes[1]]

        # 每次交易量
        v0 = self.iniVolume * b0["contract_multiplier"]
        v1 = round(v0 * cur["close"] / b1["contract_multiplier"], 0) * b1["contract_multiplier"]

        # 开仓 1/ 平仓 -1
        status = 0 if mode == 0 else 1

        # 买 / 卖 ,  若mode=0. 则按持仓反向操作
        isBuy = mode if not self.preNode[self.uid] else -self.preNode[self.uid][0]["mode"]

        # 费率
        fee0 = (self.iniVolume * b0["ratio"]) if b0["ratio"] > 0.5 else (
                b0["ratio"] * v0 * (cur["a1"] if isBuy == 1 else cur["b1"]))
        fee1 = (v1 / b1["contract_multiplier"] * b1["ratio"]) if b1["ratio"] > 0.5 else \
            (b1["ratio"] * v1 * (cur["n_a1"] if isBuy == -1 else cur["n_b1"]))

        doc = {
            "createdate": cur["datetime"],
            "code": self.codes[0],
            "price": cur["a1"] if isBuy == 1 else cur["b1"],
            "vol": v0,
            "mode": isBuy,
            "isopen": status,
            "isstop": self.isNotStop[self.uid][0],
            "fee": fee0,
            "income": 0,
            "rel_price": cur["p_l"] if isBuy == 1 else cur["p_h"],  # 实际交易价格(比价)
            "rel_std": param["std"],  # 标准差
            "batchid": self.batchId[self.uid],
            "uid": self.uid,
            "price_diff": isBuy * ((cur["p_l"] if isBuy == 1 else cur["p_h"]) - cur['close']),  #
            "delta": param["delta"],  # 价差/标准差
            "shift": param["shift"],  # 价差/tick_size
            "widthDelta": param["widthDelta"],
            "bullwidth": param["width"],  # 布林带宽度
            # "atr": param["atr"],  # 真实波幅
        }

        doc1 = {}
        doc1.update(doc)
        doc1.update({
            "code": self.codes[1],
            "price": cur["n_a1"] if isBuy == -1 else cur["n_b1"],
            "vol": v1,
            "mode": -isBuy,
            "fee": fee1,
        })

        #  计算income
        if mode == 0 and self.preNode[self.uid]:
            p0, p1 = self.preNode[self.uid][0], self.preNode[self.uid][1]
            doc["income"] = p0["mode"] * (doc["price"] - p0["price"]) * p0["vol"] - p0["fee"]
            doc1["income"] = p1["mode"] * (doc1["price"] - p1["price"]) * p1["vol"] - p1["fee"]
            self.preNode[self.uid] = None
        else:
            doc["income"] = -doc["fee"]
            doc1["income"] = -doc1["fee"]
            self.preNode[self.uid] = (doc, doc1)

        return [doc, doc1]


def main():
    action = {
        "kline": 1,
        #  "tick": 1,
    }

    if action["kline"] == 1:
        obj = model_future_quickTickTest()
        obj.pool()


if __name__ == '__main__':
    main()
