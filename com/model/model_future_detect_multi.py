# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

      期货多币对 多进程对冲交易程序
      多个币对按收盘时间统一为多个进程

"""

from com.base.public import public, logger
import pandas as pd
import talib as ta
from com.object.obj_entity import future_orderForm, future_baseInfo, train_total
from com.data.interface_Rice import interface_Rice
from multiprocessing import Pool, Manager
import uuid
import numpy as np

#from com.data.interface_pyctp import interface_pyctp


# from com.data.interface_vnpy import  interface_vnpy
"""
A_MA_60_2.5_30 1
{'code': 'A1901', 'mode': 1, 'isopen': 1, 'batchid': '0b56439c-bbd2-11e8-af4b-00163e450c45', 'price': 3712.0, 'vol': 70.0, 'fee': 14.0, 'rel_price': 1.1173992156982422}
"""
# 期货混合交易模型
class model_future_detect_multi(object):

    def __init__(self):
        # 统一参数
        self.config = {}  # 配置参数字典表
        self.deltaLimit = 1.00  # 浮动/标准差
        self.iniAmount = 250000  # 单边50万
        self.stopLine = 0  # 止损线，标准差倍数
        #
        self.topUse = True  # 启用定时统计列表
        self.sameLimit = 3  # 同一code允许出现次数
        self.topNumbers = 30  # 最大新币对 (监控币对还包括历史10和未平仓对)
        self.minRate = 0.035  # 最低收益率
        self.topTableName = 'train_total_0'
        self.methodName = 'mul'  # 策略名称
        #
        self.future_Map = []
        self.noneUsed = []  # 不再继续的商品对

    def pool(self):
        pool = Pool(processes=4)
        shareDict = Manager().list([])

        # 初始化codes列表
        if self.topUse:  self.filterCodes()

        for codes in self.seperate():
            #self.start(codes, shareDict)
            try:
                pool.apply_async(self.start, (codes, shareDict))
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

        # 同时统计未平仓的对，加入监测，待平仓后不再开仓

    def filterCodes(self):
        # 查询每周统计表
        Top = train_total()
        Top.tablename = self.topTableName
        self.future_Map = Top.last_top(num=self.topNumbers, maxsames=self.sameLimit, minRate=self.minRate)
        num0 = len(self.future_Map)

        Result = future_orderForm()

        # 添加top10
        codesList = [[c[0], c[1]] for c in self.future_Map]
        for map in Result.topCodes():
            if not map[0:2] in codesList:
                self.future_Map.append(map)
        num1 = len(self.future_Map) - num0

        # 添加未平仓，并不再开仓
        codesList = [[c[0], c[1]] for c in self.future_Map]
        for map in Result.currentOpenCodes(method=self.methodName):
            print(map)
            if not map[0:2] in codesList:
                self.future_Map.append(map)
                self.noneUsed.append(map[0:2])

        logger.info(('总监控币对：', len(self.future_Map), ' 新列表:', num0, ' top10:', num1))
        logger.info(([n for n in self.future_Map]))
        logger.info(('暂停币对:', len(self.noneUsed), [n for n in self.noneUsed]))

    # 按结束时间分割为不同进程
    def seperate(self):
        Base = future_baseInfo()
        maps, codes = {}, []

        # 合并元素
        for n in self.future_Map:
            for i in [0, 1]:
                if not n[i] in codes:
                    codes.append(n[i])

        # 按结束时间分组分进程
        for doc in Base.getInfo(codes):
            key = doc['nightEnd']
            if key not in maps.keys():
                maps[key] = []
            maps[key].append(doc['code'])

        for key in maps.keys():
            yield maps[key]

    def start(self, full_codes, shareDict):
        # 每个进程单独初始化对象
        self.Record = future_orderForm()
        self.Rice = interface_Rice()
        self.Base = future_baseInfo()
        #self.CTP = interface_pyctp()

        self.shareDict = shareDict
        self.full_codes = full_codes

        # 子进程中共享变量
        self.isOpen, self.batchId, self.preNode, self.isStop = {}, {}, {}, {}

        # 配置文件
        for doc in self.Base.getInfo(full_codes):
            self.config[doc["code"]] = doc

        # 设置交易对的收盘时间
        self.Rice.setTimeArea(doc["nightEnd"])

        # 查询主力代码，并存入字典
        full_mCodes = self.Rice.getMain(full_codes)
        self.mCodeMap = {}
        for i in range(len(full_codes)):
            self.mCodeMap[full_codes[i]] = full_mCodes[i]

        # 初始化订单状态和batchID
        self.kTypeMap = {}
        openMap = self.Record.getOpenMap(self.methodName, full_codes)

        for map in self.future_Map:
            if map[0] not in full_codes or map[1] not in full_codes : continue
            self.iniNode(map, openMap)

            # 按k线时间框类型初始化 kTypeMap dict,
            ktype = int(map[4])

            if not ktype in self.kTypeMap.keys():
                self.kTypeMap[ktype] = []

            for i in [0, 1]:
                if not self.mCodeMap[map[i]] in self.kTypeMap[ktype]:
                    self.kTypeMap[ktype].append(self.mCodeMap[map[i]])

        # 子进程启动
        logger.info(("model_future_detect start: %s " % ",".join(full_mCodes), self.Rice.TimeArea))
        self.Rice.startTick(full_mCodes, callback=self.onTick)

    # 初始化节点
    def iniNode(self, map, openMap):
        key, uid = "_".join(map[0:2]), "_".join(map)
        if key in openMap:
            self.isOpen[uid] = openMap[key][0]["mode"]
            self.batchId[uid] = openMap[key][0]["batchid"]
            print(uid, self.isOpen[uid])
            # 添加配置文件
            self.preNode[uid] = openMap[key]
        else:
            self.isOpen[uid] = 0
            self.batchId[uid] = 0  # 后续生成uuid
            self.preNode[uid] = []

    # 及时K线
    tmp = 0
    def onTick(self, tick):
        # 计算参数
        for map in self.future_Map:
            if map[0] not in self.full_codes or map[1] not in self.full_codes: continue

            self.uid = '_'.join([str(n) for n in map])  # uid

            self.codes = map[0:2]  # 代码
            self.bullPeriod = int(map[2])  # 布林带窗口大小
            self.stdAmount = float(map[3])  # 标准差倍数
            self.ktype = int(map[4])
            self.mCodes = [self.mCodeMap[c] for c in self.codes]  # 当前主力合约代码

            # 按时长读取k线数据
            dfs = self.Rice.getKline(self.kTypeMap[self.ktype], ktype=self.ktype, uid=self.ktype)

            # tick 数据
            param = self.paramCalc(dfs, tick)
            if param is not None:
                # 比较
                self.orderCheck(tick, param)

    # 计算布林参数
    def paramCalc(self, dfs, cur):
        if len(dfs) == 0:
            return None

        c0, c1 = cur[self.mCodes[0]], cur[self.mCodes[1]]
        for key in ["asks", "bids", "ask_vols", "bid_vols"]:
            if (c0[key][0] * c1[key][0] == 0): return None

        # 周期和标准差倍数
        period, scale = self.bullPeriod, self.stdAmount
        size = period + 5

        df0, df1 = dfs[self.mCodes[0]][-size:-1], dfs[self.mCodes[1]][-size:-1]

        # 计算相关参数
        close = df0["close"] / df1["close"]

        # 添加最新
        close = close.append(pd.Series(c0["last"] / c1["last"]))

        ma = ta.MA(close, timeperiod=period)
        std = ta.STDDEV(close, timeperiod=period, nbdev=1)
        top, lower = ma + scale * std, ma - scale * std
        #
        width = (top - lower) / ma * 100
        #
        #widthDelta =ta.MA(width - width.shift(1), timeperiod=3)

        return {
            "ma": ma.values[-1],
            "top": top.values[-1],
            "lower": lower.values[-1],
            "width": width.values[-1],
            "std":  std.values[-1],
            "close": c0["last"] / c1["last"],
            #"widthdelta": widthDelta.values[-1], #布林带变化
            "p_l": c0["asks"][0] / c1["bids"][0],  # 买入铜，卖出铅价格
            "p_h": c0["bids"][0] / c1["asks"][0],  # 买入铅，卖出铜价格
            "delta": (c0["asks"][0] / c1["bids"][0] - c0["bids"][0] / c1["asks"][0]) /  std.values[-1],
        }


    def orderCheck(self, cur, param):
        isRun = False
        # 开仓

        if self.isOpen[self.uid] == 0:

            #if param["delta"] > self.deltaLimit: return None

            # 已关闭的交易对只平仓， 不再开仓
            if self.codes in self.noneUsed: return None

            # 大于上线轨迹
            if param["p_h"] > param["top"]:
                self.isOpen[self.uid] = -1
                isRun = True

            # 低于下轨线
            elif param["p_l"] < param["lower"]:
                self.isOpen[self.uid] = 1
                isRun = True

        # 平仓
        else:
            # 回归ma则平仓  或  超过24分钟 或到收盘时间 强制平仓
            io = self.isOpen[self.uid]
            if io * ((param["p_h"] if io == 1 else param["p_l"]) - param["ma"]) >= 0:

                self.isOpen[self.uid] = 0
                isRun = True

        # 分钟线
        self.tmp += 1
        if self.tmp % 1000 == 1:
            #print(self.codes, cur, param)
            pass

        if isRun:
            self.order(cur, self.isOpen[self.uid], param)
            logger.info([" purchase record: ", self.codes, self.isOpen[self.uid]])


    def order(self, cur, mode, param):
        # 当前tick对
        n0, n1 = cur[self.mCodes[0]], cur[self.mCodes[1]]

        # future_baseInfo 参数值
        b0, b1 = self.config[self.codes[0]], self.config[self.codes[1]]

        times0, times1 = b0["contract_multiplier"], b1["contract_multiplier"]
        # 每次交易量
        v0 = (round(self.iniAmount / n0["last"] / times0, 0) * times0) if not self.preNode[self.uid] else \
        self.preNode[self.uid][0]["vol"]
        v1 = round(self.iniAmount / n1["last"] / times1, 0) * times1 if not self.preNode[self.uid] else \
        self.preNode[self.uid][1]["vol"]

        # 开仓 1/ 平仓 -1
        status = 0 if mode == 0 else 1

        # 买 / 卖 ,  若mode=0. 则按持仓反向操作
        isBuy = mode if not self.preNode[self.uid] else -self.preNode[self.uid][0]["mode"]

        # 费率
        fee0 = (v0 / b0["contract_multiplier"] * b0["ratio"]) if b0["ratio"] > 0.5 else (
                b0["ratio"] * v0 * (n0["asks"][0] if isBuy == 1 else n0["bids"][0]))
        fee1 = (v1 / b1["contract_multiplier"] * b1["ratio"]) if b1["ratio"] > 0.5 else \
            (b1["ratio"] * v1 * (n1["asks"][0] if isBuy == -1 else n1["bids"][0]))

        now = public.getDatetime()
        # 使用uuid作为批次号
        if mode != 0:
            self.batchId[self.uid] = uuid.uuid1()

        doc = {
            "createdate": now,
            "code": n0["code"],
            "price": n0["asks"][0] if isBuy == 1 else n0["bids"][0],
            "vol": v0,
            "hands": v0 / b0["contract_multiplier"],
            "mode": isBuy,
            "isopen": status,
            "fee": fee0,
            "income": 0,
            "rel_price": param["p_l"] if isBuy == 1 else param["p_h"],
            "rel_std": param["std"],
            "batchid": self.batchId[self.uid],
            "method": self.methodName,
            "uid": self.uid
        }

        doc1 = {}
        doc1.update(doc)
        doc1.update({
            "code": n1["code"],
            "price": n1["asks"][0] if isBuy == -1 else n1["bids"][0],
            "vol": v1,
            "hands": v1 / b1["contract_multiplier"],
            "mode": -isBuy,
            "fee": fee1,
        })

        if mode == 0 and self.preNode:
            p0, p1 = self.preNode[self.uid][0], self.preNode[self.uid][1]
            doc["income"] = p0["mode"] * (doc["price"] - p0["price"]) * p0["vol"] - p0["fee"]
            doc1["income"] = p1["mode"] * (doc1["price"] - p1["price"]) * p1["vol"] - p1["fee"]
            self.preNode[self.uid] = None

        else:
            doc["income"] = -doc["fee"]
            doc1["income"] = -doc1["fee"]
            self.preNode[self.uid] = (doc, doc1)

        # 发送订单
        self.send([doc, doc1])

    def send(self, res):
        # 存入数据库
        self.Record.insertAll(res)
        # 通过CTP下单
        try:
            pass
            #self.CTP.sendOrder(res)
        except:
            print('ctp error')

    def sendCallBack(self):
        pass


def main():
    obj = model_future_detect_multi()
    obj.pool()


if __name__ == '__main__':
    main()
