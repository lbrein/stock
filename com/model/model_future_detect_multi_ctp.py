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
#from multiprocessing.managers import BaseManager

import uuid
import numpy as np
from com.ctp.interface_pyctp import interface_pyctp, BaseInfo, ProcessMap
import time
import re
import traceback
import copy

# 期货混合交易模型
class model_future_detect_multi(object):

    def __init__(self):
        # 统一参数
        self.config = {}  # 配置参数字典表
        self.deltaLimit = 0.8  # 浮动/标准差
        self.iniAmount = 250000  # 单边50万
        self.stopTimeLine = 5  # 止损时间线
        self.batchNum = 2
        self.banCodeList = [] # 暂时无权限操作的code

        self.isWorking = False
        self.ctpuser = 'simnow1'
        #
        self.topUse = True  # 启用定时统计列表
        self.isCTPUse = False  # 是否启用CTP接口（并区分
        self.isAutoAlterPosition = True # 到期是否自动调仓
        self.isDeadCheck = True # 是否启用自动检查并交易沉淀品种

        self.topFilter = ' count>10 '
        self.sameLimit = 3  # 同一code允许出现次数
        self.topNumbers = 30  # 最大新币对 (监控币对还包括历史10和未平仓对)
        self.minRate = -0.03 # 最低收益率

        self.orderFormTest = 'future_orderForm_test'
        self.topTableName = 'train_total_s2'
        self.methodName = 'mul'  # 策略名称
        #
        self.future_Map = []
        self.noneUsed = []  # 不再继续的商品对

    def pool(self):
        pool = Pool(processes=6)
        shareDict = Manager().list([])
        CTP = None
        # 初始化codes列表
        if self.topUse:
             self.filterCodes()

        for codes in self.seperate():
            #if "" in codes or "AP" not in codes: continue
            #self.start(codes, shareDict, CTP)
            try:
                pool.apply_async(self.start, (codes, shareDict, CTP))
                time.sleep(3)
                pass
            except Exception as e:
                print('error', e)
                continue

        pool.close()
        pool.join()

        # 同时统计未平仓的对，加入监测，待平仓后不再开仓
    def filterCodes(self):
        # 查询每周统计表
        Top = train_total()
        Top.tablename = self.topTableName
        self.future_Map = Top.last_top(num=self.topNumbers, filter=self.topFilter, maxsames=self.sameLimit,
                                       minRate=self.minRate)
        num0 = len(self.future_Map)
        Record = future_orderForm()
        if not self.isWorking:  Record.tablename = self.orderFormTest

        # 添加top10
        codesList = [[c[0], c[1]] for c in self.future_Map]
        for map in Record.topCodes(method=self.methodName, toptable=self.topTableName,batchNum=self.batchNum):
            if not map[0:2] in codesList:
                self.future_Map.append(map)

        num1 = len(self.future_Map) - num0

        # 添加未平仓，并不再开仓
        codesList = [[c[0], c[1]] for c in self.future_Map]
        for map in Record.currentOpenCodes(method=self.methodName, batchNum=self.batchNum):
            if not map[0:2] in codesList and map[0]:
                self.future_Map.append(map[:-1])
                self.noneUsed.append(map[0:2])

        # 开盘时调仓处理
        if self.isCTPUse and self.isAutoAlterPosition:
            codes = self.combinCode()
            try:
                CTP = interface_pyctp(use=True, userkey=self.ctpuser)
                state, orders = CTP.alterPosi(codes)
                if len(orders)- state >0:
                    logger.info(('自动调仓完成', len(orders)- state))
                    Record.insertAll(orders)
            except:
                pass

        logger.info(('总监控币对：', len(self.future_Map), ' 新列表:', num0, ' top10:', num1, ' 未平仓:', len(self.noneUsed)))
        logger.info(([n for n in self.future_Map]))
        logger.info(('暂停币对:', len(self.noneUsed), [n for n in self.noneUsed]))

    def combinCode(self):
        codes = []
        for n in self.future_Map:
            for i in [0, 1]:
                # 暂时禁止交易的品种FU
                if not n[i] in codes and n[i] not in self.banCodeList:
                    codes.append(n[i])
        return codes

    # 按结束时间分割为不同进程
    def seperate(self):
        Base = future_baseInfo()
        maps, codes = {}, []
        dd = str(public.getTime(style='%H:%M:%S'))
        # 按结束时间分组分进程
        for doc in Base.getInfo(self.combinCode()):
            key = doc['nightEnd']
            #
            if '03:00:00' < key < dd: continue

            if key not in maps.keys():
                maps[key] = []
            maps[key].append(doc['code'])

        for key in maps.keys():
            yield maps[key]

    def start(self, full_codes, shareDict, CTP):
        self.shareDict = shareDict
        self.Record = future_orderForm()
        if not self.isWorking:
            self.Record.tablename = self.orderFormTest

        self.Rice = interface_Rice()
        # 基础配置信息类
        self.baseInfo = BaseInfo(full_codes, self.Rice)
        # ctp 接口类
        self.CTP = CTP if CTP is not None else interface_pyctp(use=self.isCTPUse,userkey=self.ctpuser)
        self.CTP.baseInfo = self.baseInfo
        # 进程控制类
        self.procMap = ProcessMap()

        # 设置交易对的收盘时间
        self.Rice.setTimeArea(self.baseInfo.nightEnd)
        # 按k线类型分拆组，进行K线数据调用
        self.groupByKline(full_codes)

        # 初始化节点
        try:
            self.iniNode(full_codes)
        except Exception as e:
            print('iniNode error', e)

        #return
        # 子进程启动
        full_mCodes = self.baseInfo.mainCodes #主力合约
        logger.info(("model_future_detect start: %s " % ",".join(full_mCodes), self.Rice.TimeArea))
        self.Rice.startTick(full_mCodes, callback=self.onTick)

    # 按K线类型分组
    def groupByKline(self, full_codes):
        self.used_future_Map, self.kTypeMap  = [], {}
        for map in self.future_Map:
            if map[0] not in full_codes: continue
            self.used_future_Map.append(map)

            # 按k线时间框类型初始化 kTypeMap dict,
            ktype = int(map[4])
            if ktype not in self.kTypeMap.keys():
                self.kTypeMap[ktype] = []

            for i in [0, 1]:
                mCode = self.baseInfo.mCode(map[i])
                if mCode not in self.kTypeMap[ktype]:
                    self.kTypeMap[ktype].append(mCode)

    # 初始化节点
    def iniNode(self, full_codes):

        openMap = self.Record.getOpenMap(self.methodName, full_codes)
        # 非CTP方式
        if not self.isCTPUse:
            for map in self.used_future_Map:

                key, uid = "_".join(map[0:2]), "_".join(map)
                self.procMap.new(uid)
                if key in openMap:
                    self.procMap.setIni(uid, openMap[key])
                    print(uid, key, self.procMap.isOpen)
            return True

        # 先检查有记录
        ods = []
        for map in self.used_future_Map:
            if  map[0] in self.banCodeList or map[1] in self.banCodeList: continue

            key, uid = "_".join(map[0:2]), "_".join(map)
            self.procMap.new(uid)  # 初始化uid

            if key in openMap:
                docs = self.CTP.iniPosition(map[0:2], openMap[key])
                # 无position的
                if docs is None:
                    print('--- 0 unmatch ---', key)
                    for d in openMap[key]: ods.append(str(d['id']))
                    openMap.pop(key)
                else:
                    print('--1', uid, 'mode', docs[0]["mode"], docs[0]["hands"], docs[1]["hands"])
                    # 满足position,则设置初始状态
                    self.procMap.setIni(uid, docs)  # 有记录的并有匹配position的

        if len(ods) > 0:  # 废弃不满足条件的商品对
            self.Record.disuse(ods)
            pass

        # 无交易记录 ，恢复出交易的
        res = []
        for map in self.used_future_Map:
            key, uid = "_".join(map[0:2]), "_".join(map)
            if key in openMap.keys(): continue

            docs = self.CTP.iniPosition(map[0:2], [])
            if docs is not None:
                # 补充交易记录
                print('--2', uid, 'mode', docs[0]["mode"], docs[0]["hands"], docs[1]["hands"])
                batchid = uuid.uuid1()
                for d in docs:
                    d.update({
                        'batchid': batchid,
                        'uid': uid,
                        'method': self.methodName,
                        'status': 6
                    })
                    res.append(d)
                self.procMap.setIni(uid, docs)  # 有记录的并有匹配position的

        if len(res)> 0: # 添加记录
            # 测试检查
            self.Record.insertAll(res)

        # 处理僵尸
        if self.isDeadCheck:
            self.dead(full_codes)

    # 处理僵尸商品对，直接卖出
    def dead(self, full_codes):
        t, res = self.CTP.orderDead(full_codes)
        if res is not None and t < len(res):
            logger.info(("dead deal success :", len(res), t))
            self.Record.insertAll([d for d in res if d['status']==6])

    # Tick 响应
    def onTick(self, tick):
        # 计算参数
        for map in self.used_future_Map:
            if map[0] in self.banCodeList: continue

            self.uid = self.procMap.setUid(map) # uid
            self.mCodes = [self.baseInfo.mCode(c) for c in self.procMap.codes]  # 当前主力合约代码
            # 挂起项目
            if self.procMap.isOpen == -9 : continue

            kline = self.procMap.kline
            # 按时长读取k线数据
            dfs = self.Rice.getKline(self.kTypeMap[kline], ktype=kline, key=kline)
            try:
                # 计算指标
                param = self.paramCalc(dfs, tick)
                if param is not None and not np.isnan(param["ma"]):
                    # 执行策略，并下单
                    self.orderCheck(tick, param)
            except Exception as e:
                print(traceback.format_exc())

    # 计算布林参数
    def paramCalc(self, dfs, cur):
        # 分钟线
        if dfs is None or len(dfs) == 0: return None

        c0, c1 = cur[self.mCodes[0]], cur[self.mCodes[1]]
        for key in ["asks", "bids", "ask_vols", "bid_vols"]:
            if (c0[key][0] * c1[key][0] == 0): return None

        # 周期和标准差倍数
        period, scale = self.procMap.period, self.procMap.scale
        size =  period + 10

        # 去掉当前的即时k线
        df0, df1 = dfs[self.mCodes[0]][-size:-1], dfs[self.mCodes[1]][-size:-1]

        # 计算相关参数
        close = (df0["close"] / df1["close"])
        # nan 处理
        close = close.fillna(close[close.notnull()].mean())

        # 添加最新
        close = close.append(pd.Series(c0["last"] / c1["last"]))

        ma = ta.MA(close, timeperiod=period)
        sd = ta.STDDEV(close, timeperiod=period, nbdev=1)
        top, lower = ma + scale * sd, ma - scale * sd
        #
        width = (top - lower) / ma * 100
        #

        widthDelta = ta.MA(width - width.shift(1), timeperiod=3)
        # 即时K线取2点中值

        wd2 = widthDelta - widthDelta.shift(1)
        wd2m = widthDelta * widthDelta.shift(1)

        return {
            "ma": ma.values[-1],
            "top": top.values[-1],
            "lower": lower.values[-1],
            "width": width.values[-1],
            "std": sd.values[-1],
            "close": c0["last"] / c1["last"],
            "widthdelta": widthDelta.values[-1],  # 布林带变化
            "wd2": wd2.values[-1],  # 布林带二阶变化率
            "wd2m": wd2m.values[-1],

            "p_l": c0["asks"][0] / c1["bids"][0],  # 买入铜，卖出铅价格
            "p_h": c0["bids"][0] / c1["asks"][0],  # 买入铅，卖出铜价格
            "delta": (c0["asks"][0] / c1["bids"][0] - c0["bids"][0] / c1["asks"][0]) / sd.values[-1],
            "interval":0,
        }

    itemCount = 0
    def debugT(self,str, n=1000):
        self.itemCount += 1
        if self.itemCount % n ==0:
            print(self.itemCount, str)

    def orderCheck(self, cur, param):
        isOpen, isRun, isstop = self.procMap.isOpen, False, 0
        wline, sd2 = self.procMap.widthline, self.procMap.scaleDiff2

        if param["delta"] > self.deltaLimit: return None
        # 开仓
        cond1, cond2 = False, False
        if wline > 0:
            # 布林宽带变化率
            cond1 = (param['widthdelta'] < wline and param['wd2'] < (wline / 2))
            # 拐点
            cond2 = param['wd2m'] < 0

        if isOpen == 0:
            # 已关闭的交易对只平仓， 不再开仓
            if self.procMap.codes in self.noneUsed: return None
            # 大于上线轨迹
            if (param["p_h"] > param["top"]) and cond1:
                isOpen, isRun = -1, True

                # 低于下轨线
            elif (param["p_l"] < param["lower"]) and cond1:
                isOpen, isRun = 1, True


            elif ((param["p_h"] + sd2 * param['std']/2) > param["top"]) and not cond1 and cond2:
                isOpen = -2
                isRun = True

            elif ((param["p_l"] - sd2 * param['std']/2) < param["lower"]) and not cond1 and cond2:
                isOpen = 2
                isRun = True

        # 平仓
        else:

            stopMinutes = self.stopTimeLine * self.procMap.period * self.procMap.kline
            preNode = self.procMap.preNode

            cond3 = (isOpen * ((param['p_h'] if isOpen > 0 else param['p_l']) - param['ma'])) >= 0
            #
            cond4 = (isOpen * (
                    ((param['p_h'] + sd2 / 2 * param['std']) if isOpen > 0 else (
                            param['p_l'] - sd2 / 2 * param['std'])) - param['ma'])) >= 0

            # 回归ma则平仓  或  超过24分钟 或到收盘时间 强制平仓
            if cond4 and not cond1 and cond2:
                isOpen, isRun, isstop = 0, True, 2

            elif cond3 and cond1:
                isOpen, isRun = 0, True

            # 止损
            elif stopMinutes > 0 and preNode is not None:
                tdiff = self.Rice.timeDiff(preNode[0]['createdate'], quick=stopMinutes)
                if tdiff > stopMinutes and cond4 and cond2:
                    isOpen, isstop = 0, 1
                    isRun = True

        #self.debugT((self.uid,isOpen,isRun))

        if isRun:
            self.order(cur, isOpen, param, isstop=isstop)

    def order(self, cur, mode, param, isstop=0):
        # 当前tick对
        n0, n1 = cur[self.mCodes[0]], cur[self.mCodes[1]]

        # future_baseInfo 参数值
        b0, b1 = (self.baseInfo.doc(self.procMap.codes[i]) for i in [0, 1])
        times0, times1 = b0["contract_multiplier"], b1["contract_multiplier"]

        preNode = self.procMap.preNode
        # 每次交易量
        v0 = (round(self.iniAmount / n0["last"] / times0, 0) * times0) if not preNode else preNode[0]["hands"] * times0
        v1 = round(self.iniAmount / n1["last"] / times1, 0) * times1 if not preNode else preNode[1]["hands"] * times1

        # 开仓 1/ 平仓 -1
        isOpen = 0 if mode == 0 else 1

        # 买 / 卖 ,  若mode=0. 则按持仓方向平仓操作
        isBuy = -preNode[0]["mode"] if (mode == 0 and preNode is not None) else mode

        # 费率
        fee0 = (v0 / times0 * b0["ratio"]) if b0["ratio"] > 0.5 else (
                b0["ratio"] * v0 * (n0["asks"][0] if isBuy == 1 else n0["bids"][0]))
        fee1 = (v1 / times1 * b1["ratio"]) if b1["ratio"] > 0.5 else \
            (b1["ratio"] * v1 * (n1["asks"][0] if isBuy == -1 else n1["bids"][0]))

        now = public.getDatetime()
        # 使用uuid作为批次号
        if mode != 0: self.procMap.batchid = uuid.uuid1()

        doc = {
            "createdate": now,
            "code": n0["code"],
            "symbol": self.baseInfo.ctpCode(n0["code"]),
            "price": n0["asks"][0] if isBuy == 1 else n0["bids"][0],
            "vol": v0,
            "hands": v0 / times0,
            "ini_hands": v0 / times0,  # 提交单数
            "ini_price": n0["asks"][0] if isBuy == 1 else n0["bids"][0],  # 提交价格
            "mode": isBuy,
            "isopen": isOpen,
            "isstop":isstop,
            "fee": fee0,
            "income": 0,
            "rel_price": param["p_l"] if isBuy == 1 else param["p_h"],
            "rel_std": param["std"],
            "batchid": self.procMap.batchid,
            "delta": param["delta"],
            "bullwidth": param["width"],
            "widthdelta": param["widthdelta"],
            "status": 0,  # 定单状态
            "method": self.methodName,
            "uid": self.uid
        }

        doc1 = copy.deepcopy(doc)
        doc1.update({
            "code": n1["code"],
            "symbol": self.baseInfo.ctpCode(n1["code"]),
            "price": n1["asks"][0] if isBuy == -1 else n1["bids"][0],
            "vol": v1,
            "ini_price": n1["asks"][0] if isBuy == -1 else n1["bids"][0],
            "hands": v1 / times1,
            "ini_hands": v1 / times1,
            "mode": -isBuy,
            "fee": fee1,
        })

        if mode == 0 and preNode is not None:
            p0, p1 = preNode[0], preNode[1]
            doc["income"] = round(p0["mode"] * (doc["price"] - p0["price"]) * p0["vol"] - doc["fee"], 2)
            doc1["income"] = round(p1["mode"] * (doc1["price"] - p1["price"]) * p1["vol"] - doc1["fee"], 2)
            doc["interval"] = doc1["interval"] = self.Rice.timeDiff(p0['createdate'])
        else:
            doc["income"] = -doc["fee"]
            doc1["income"] = -doc1["fee"]

        # 下单并记录
        self.record([doc, doc1], mode)

    def record(self, orders, mode):
        state = 0
        isOpen = 0 if mode == 0 else 1
        # 发送订单并记录到数据库
        if self.isCTPUse and self.CTP is not None:
            # 检查订单条件
            state = self.CTP.checkPosition(orders, isOpen)
            if state == 0:
                # 发送订单
                try:
                    self.CTP.sendOrder(orders)
                    # 检查全部执行结果
                    state, orders = self.CTP.checkResult()
                except Exception as e:
                    print(traceback.format_exc())
                    state = 2

        if state == 0:
            logger.info(["---- purchase record:-- ", state, self.uid, mode])
            # 保存 到进程变量
            self.procMap.preNode = orders if (mode != 0) else None
            self.procMap.isOpen = mode
            # 交易成功
            self.Record.insertAll(orders)

        elif state < 0:
             print('----账户持仓检查不符合 ----:', state, self.uid, mode, orders)
             time.sleep(3)

        elif state == 1:
             logger.info(('---- 配对只成交一单--:', state, self.uid, mode, orders))
             self.procMap.isOpen = -9 # 标记为挂起项目
             # 执行反向操作
             s, res = self.CTP.forceOrder(orders)
             if s==0:
                 logger.info(('--配对单成交反向操作成功----:', res))
                 orders.extend(res)   # 添加反向操作记录

             self.Record.insertAll(orders)
             time.sleep(3)
        else:
             print('---- 交易不成功 ----:', state, self.uid, mode, orders)
             time.sleep(3)

def main():
    obj = model_future_detect_multi()
    obj.pool()


if __name__ == '__main__':
    main()
