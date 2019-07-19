# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

      期货单品种交易

"""

from com.base.public import public, logger
import pandas as pd
import talib as ta
from com.object.obj_entity import future_orderForm, future_baseInfo, train_total, future_status
from com.data.interface_Rice import interface_Rice
from multiprocessing import Pool, Manager
from multiprocessing.managers import BaseManager

import uuid
import numpy as np
from com.ctp.interface_pyctp import interface_pyctp, BaseInfo, ProcessMap
import time
import traceback
import math
import copy
from com.object.mon_entity import mon_tick

# 回归方法
class MyManager(BaseManager):
      pass

MyManager.register('interface_Rice', interface_Rice)
MyManager.register('future_baseInfo', future_baseInfo)
MyManager.register('interface_pyctp', interface_pyctp)

def Manager2():
    m = MyManager()
    m.start()
    return m

# 期货混合交易模型
class model_future_ctp(object):

    def __init__(self):
        # 统一参数
        self.iniAmount = 150000  # 单边50万

        self.isWorking = False  # 是否正式运行
        self.isAutoAlterPosition = True
        self.banCodeList = []  # 暂时无权限操作的code
        self.isTickSave = True
        self.batchNum = 1  # 批处理的交易数量
        self.topUse = True  # 启用定时统计列表
        self.timeInterval = 0.5 # 间隔处理时间

        self.indexCodeList = [] # 指数期货

        self.isCTPUse = False  # 是否启用CTP接口（并区分
        self.isDeadCheck = False  # 是否启用自动检查并交易沉淀品种
        self.topFilter = """(count>10)"""
        self.ctpuser = 'simnow1'
        self.tickIntList = ['kdjm', 'sarm', 'isout', 'isout3', 'isout5']
        self.tickInterval = 10  # 间隔处理时间

        self.sameLimit = 1  # 同一code允许出现次数
        self.topNumbers = 30  # 最大新币对 (监控币对还包括历史10和未平仓对)
        self.bestTopNumbers = 10
        self.minRate = 0.02  # 最低收益率
        self.orderFormTest = 'future_orderForm_test'
        self.topTableName = 'train_total_dema5'

        self.methodName = 'single'  # 策略名称
        self.relativeMethods = ['mZhao', 'mZhao55']
        self.scaleDiff2 = 0.8
        self.powline = 0.25

        self.widthTimesPeriod = 3
        self.sourceType = 'combin'
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

        pid = 0
        for codes in self.seperate():
            print('pool send:', pid, len(codes))
            #self.start(codes, shareDict, CTP)
            try:
                pool.apply_async(self.start, (codes, shareDict, CTP))
                time.sleep(3)
                pid += 1
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

    def setAlterIncome(self, orders, openDocs):
        for doc in orders:
            c = doc['name']
            if doc['isopen'] == 0 and c in openDocs:
                p0 = openDocs[c][0]
                sign = np.sign(p0['mode'])
                doc["income"] = round(sign * (doc["price"] - p0["price"]) * p0["vol"] - doc["fee"], 2)
                doc["interval"] = self.Rice.timeDiff(p0['createdate'])  # 间隔时间

        return orders

    # 调仓处理
    def alterPosition(self, Record, openCodes):
        # 开盘时CTP调仓处理
        Rice = interface_Rice()
        if self.isCTPUse and self.isAutoAlterPosition:
            openDocs = Record.getOpenMap(self.methodName, codes=[], batchNum=self.batchNum)
            try:
                CTP = interface_pyctp(use=True, userkey=self.ctpuser)
                state, orders = CTP.alterPosi(openDocs)

                if len(orders) - state > 0:
                    logger.info(('自动调仓完成', len(orders) - state))
                    orders = self.setAlterIncome(orders, openCodes)
                    Record.insertAll(orders)

            except:
                logger.info(('alterPosition error:', self.methodName))

        elif self.isAutoAlterPosition:
            # 非CTP调仓
            if openCodes is not None:
                openDocs = Record.getOpenMap(self.methodName, codes=[], batchNum=self.batchNum)
                orders, d0, d1 = [], {}, {}

                for key in openDocs:
                    # 查询最新主力代码
                    doc, pCode = openDocs[key][0], openDocs[key][0]['code']
                    mCode = Rice.getMain([key])[0]

                    if mCode!= pCode:
                        print('start:', pCode, mCode)

                        # 查询最新价格，调仓前和调仓后
                        s = Rice.snap([pCode, mCode])
                        if s is None: continue
                        price0 = s[pCode]['bids'][0] if doc['mode']>0 else s[pCode]['asks'][0]
                        price1 = s[mCode]['bids'][0] if doc['mode']<0 else s[mCode]['asks'][0]

                        d0 = copy.deepcopy(doc)
                        d0['mode'] = -doc['mode']
                        d0['isopen'] = 0
                        d0['price'] = d0['rel_price'] = price0
                        sign = np.sign(-d0["mode"])
                        d0["income"] = round(sign * (price0 - doc["price"]) * d0["vol"] - doc["fee"], 2)
                        del d0['id']

                        # 按原状态买入
                        d1 = copy.deepcopy(doc)
                        d1['code'] = mCode
                        d1['status'] = d0['status'] = 4

                        d0['method'] = d1['method'] = self.methodName
                        d1['rel_price'] = d1['price'] = price1
                        d0['createdate'] = d1['createdate'] = public.getDatetime()
                        d1['batchid'] = uuid.uuid1()
                        del d1['id']

                        orders.append(d0)
                        orders.append(d1)
                        logger.info(('自动调仓完成：', doc['code'], mCode, 4))

                if len(orders)>0:
                       Record.insertAll(orders)
                       pass

    def combinCode(self):
        codes = []
        for n in self.future_Map:
            if not n[0] in codes and n[0] not in self.banCodeList:
                codes.append(n[0])
        return codes

    # 按结束时间分割为不同进程
    def seperate(self):
        Base = future_baseInfo()
        maps, codes = {}, []
        dd = str(public.getTime(style='%H:%M:%S'))

        # 按结束时间分组分进程
        for doc in Base.getInfo(self.combinCode()):
            key = key0 = doc['nightEnd']
            if '03:00:00' < key0 < dd: continue
            if '08:30:00' < dd < '15:00:00':
                key = '15:00:00' if '09:00:00' < key0 < '15:30:00' else '23:30:00' if '15:30:00' < key0 < '23:35:00' else '02:30:00'
                pass

            if key not in maps.keys():
                maps[key] = []

            maps[key].append(doc['code'])

        pc = 11
        for key in maps.keys():
            if len(maps[key]) < pc+1:
                yield maps[key]
            else:
                l, t = len(maps[key]), len(maps[key]) // pc
                for i in range(t+1):
                    s, e = i * pc, l if (i + 1) * pc > l else (i + 1) * pc
                    if s < e:
                        yield maps[key][s:e]


    def start(self, full_codes, Rice = None, CTP=None):
        # print(full_codes)
        self.Record = future_orderForm()
        self.PStatus = future_status()

        if not self.isWorking: self.Record.tablename = self.orderFormTest

        if self.isTickSave:
            self.Tick = mon_tick()

        self.Rice = interface_Rice() if Rice is None else Rice

        # 基础配置信息类
        self.baseInfo = BaseInfo(full_codes, self.Rice)

        # ctp 接口类
        self.CTP = interface_pyctp(use=self.isCTPUse, userkey=self.ctpuser) if CTP is None else CTP
        self.CTP.baseInfo = self.baseInfo

        # 进程控制类
        self.procMap = ProcessMap()

        # 趋势预测
        self.trendMap = self.Record.trendMap(self.relativeMethods)

        # 指数期货
        self.indexList = [c[0] for c in self.indexCodeList]

        # 设置交易对的收盘时间
        self.Rice.setTimeArea(self.baseInfo.nightEnd)

        if len(self.indexCodeList) > 0:
             self.Rice.setIndexList(self.indexCodeList)

        # 按k线类型分拆组，进行K线数据调用
        self.groupByKline(full_codes)

        # 初始化节点
        self.iniNode(full_codes)

        # 子进程启动
        full_mCodes = self.baseInfo.mainCodes  # 主力合约

        logger.info(("model_future_detect start: %s " % ",".join(full_codes), self.Rice.TimeArea))

        self.Rice.startTick(full_mCodes, kmap=self.kTypeMap, timesleep=self.timeInterval,  source=self.sourceType, callback=self.onTick)

    # 按K线类型分组
    def groupByKline(self, full_codes):
        self.used_future_Map, self.kTypeMap = [], {}

        for map in self.future_Map:
            if map[0] not in full_codes: continue
            self.used_future_Map.append(map)

            # 按k线时间框类型初始化 kTypeMap dict,
            ktype = int(map[3])
            # 字典
            self.kTypeMap[map[0]] = ktype
            # 分组
            if ktype not in self.kTypeMap.keys():
                self.kTypeMap[ktype] = []

            mCode = self.baseInfo.mCode(map[0])
            if mCode not in self.kTypeMap[ktype]:
                self.kTypeMap[ktype].append(mCode)

    # 初始化节点
    def iniNode(self, full_codes):
        openMap = self.Record.getOpenMap(self.methodName, codes=full_codes, batchNum=self.batchNum)
        statusMap = self.PStatus.getStatus(self.topTableName)

        # 非CTP方式
        for map in self.used_future_Map:
            #print(map)
            key, uid = map[0], "_".join(map)
            self.procMap.new(uid)  # 初始化进程参数类

            # 初始化品种状态
            if key in openMap:
                found = True
                # CTP方式检查
                if self.isCTPUse:
                    # 检查持仓满足
                    found = self.CTP.checkPosition(openMap[key], 0, reverse=-1, refresh=False) == 0
                    #if key =='IH': print(openMap[key], found, res)

                if found:
                    status = statusMap[key] if key in statusMap else 0
                    self.procMap.setIni(uid, openMap[key], status=status)
                    logger.info((uid, key, self.procMap.isOpen, self.procMap.status))


    pointColumns = ['wdd', 'pow', 'powm', 'trend', 'isout']
    def orderCheck(self, cur, param):
        pass

    def paramCalc(self, dfs, cur):
        pass

    def sd(self, x):
        s = round(x * 1.6, 0) / 10
        return np.sign(s) * 0.8 - 0.1 if abs(s) > 0.8 else s - 0.1

    def stand(self, ser):
        ser = ser.fillna(0)
        return ser / ser.abs().mean()

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    def apart(self, PS, ktype):
        apart = math.pow((int(time.time()) % (60 * ktype)) * 1.0 / (60 * ktype), 0.5)
        return PS * apart + PS.shift(1) * (1 - apart)

    itemCount = 0
    cvsMap, cvsCodes = {}, []
    def debugT(self, str, n=1000, param=None):
        self.itemCount += 1
        if str != '' and self.itemCount % n == 0:
            logger.info((self.itemCount, str))

        #print(self.isTickSave)

        if not self.isTickSave: return

        code, method = self.procMap.codes[0], self.procMap.currentUid.split("_")[-1]
        #print(self.csvCodes)
        if param is not None and (code in self.cvsCodes or self.cvsCodes == []):
            # param = param.to_dict()
            param['code'] = code
            param['method'] = method
            param['isopen'] = self.procMap.isOpen
            param['status'] = self.procMap.status

            for key in self.tickIntList:
                if key in param: param[key] = int(param[key])

            # 初始化
            if code not in self.cvsMap:
                self.cvsMap[code] = [param]
                self.cvsMap['c_' + code] = 1

            else:
                self.cvsMap['c_' + code] += 1
                c, t = self.cvsMap[code], self.cvsMap['c_' + code]

                if len(c) > 2:
                    if self.isTickSave:
                        self.Tick.col.insert_many(self.cvsMap[code])

                    self.cvsMap[code] = []

                elif t % self.tickInterval == 0:
                    #print(param)
                    self.cvsMap[code].append(param)

    def debugR(self, param=None):

        if not self.isTickSave: return
        code, method = self.procMap.codes[0], self.procMap.currentUid.split("_")[-1]
        if param is not None:
            param['code'] = code
            param['method'] = method

            for key in self.tickIntList:
                if key in param: param[key] = int(param[key])

            #print(param)
            try:
                self.Tick.col.insert_one(param)
            except Exception as e:
                logger.error((traceback.format_exc()))

    def procTemp(self, param):
        for key in ['atr', 'powm']:
            if key in param:
                self.procMap.__setattr__(key, param[key])

    # 买入后的最高/最低价
    def getMax(self, df0, s, e, mode):
        s = str(s)[:10]
        if mode > 0:
            return df0[(df0['datetime']>=s) & (df0['datetime']<e)].ix[:, 'close'].max()
        else:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:,'close'].min()

    def order(self, cur, mode, param, isstop=0):
        # 当前tick对
        n0 = cur[self.mCodes[0]]

        # future_baseInfo 参数值
        b0 = self.baseInfo.doc(self.procMap.codes[0])
        times0 = b0["contract_multiplier"]

        preNode = self.procMap.preNode
        # 每次交易量
        v0 = (round(self.iniAmount / n0["last"] / times0, 0) * times0) if not preNode else preNode[0]["hands"] * times0
        # 开仓 1/ 平仓 -1
        isOpen = 0 if mode == 0 else 1

        # 买 / 卖 ,  若mode=0. 则按持仓方向平仓操作
        isBuy = -preNode[0]["mode"] if (mode == 0 and preNode is not None) else mode
        # 费率
        fee0 = (v0 / times0 * b0["ratio"]) if b0["ratio"] > 0.5 else (
                b0["ratio"] * v0 * (n0["asks"][0] if isBuy == 1 else n0["bids"][0]))

        now = public.getDatetime()
        # 使用uuid作为批次号
        if mode != 0:
            self.procMap.batchid = uuid.uuid1()

        price = n0["asks"][0] if isBuy == 1 else n0["bids"][0]
        if price == 0:
            price = n0['close']

        doc = {
            "createdate": now,
            "code": n0["code"],
            "symbol": self.baseInfo.ctpCode(n0["code"]),
            "price": price,
            "vol": v0,
            "hands": v0 / times0,
            "ini_hands": v0 / times0,  # 提交单数
            "ini_price": n0["asks"][0] if isBuy == 1 else n0["bids"][0],  # 提交价格
            "mode": isBuy,
            "isopen": isOpen,
            "isstop": isstop,
            "fee": fee0,
            "income": 0,
            "rel_price": param["p_l"] if isBuy == 1 else param["p_h"],
            "rel_std": param["std"],
            "batchid": self.procMap.batchid,
            "delta": param["pow"] if 'pow' in param else 0,
            "bullwidth": param["width"],
            "widthdelta": param["wd1"],
            # "macd": param["macd"],
            "status": 0,  # 定单P执行CT返回状态
            "pstatus": int(self.procMap.status),  # 策略状态：0-bull 1,-1: trend状态
            "method": self.methodName,
            "uid": self.uid
        }

        # 下单并记录
        return self.record([doc], mode)

    def setIncome(self, orders, mode):
        doc = orders[0]
        preNode = self.procMap.preNode
        if mode == 0 and preNode is not None:
            p0 = preNode[0]
            sign = np.sign(p0["mode"])
            doc["income"] = round(sign * (doc["price"] - p0["price"]) * p0["vol"] - doc["fee"], 2)
            #doc["interval"] = int(self.Rice.timeDiff(p0['createdate']))  # 间隔时间
        else:
            doc["income"] = -doc["fee"]
        return orders

    # 设置过程状态
    def setPStatus(self, status):
        """
            更改进程状态，并写入数据库，供重新启动程序时调用
        """
        self.procMap.status = status
        method = self.topTableName
        self.PStatus.setStatus('_'.join(self.procMap.codes), method, status)
        logger.info(("setPStatus", self.procMap.codes, method, status))


    def record(self, orders, mode):
        state = 0
        isOpen = 0 if mode == 0 else 1
        # 发送订单并记录到数据库
        if self.isCTPUse and self.CTP is not None and orders[0]['vol']>0:
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

        # 检查结果写入数据库
        if state == 0:
            # 重新计算实际收入
            self.setIncome(orders, mode)
            logger.info([" purchase record: ", state, self.uid, mode, orders[0]])

            # 保存/清空 前一次操作文件 到进程变量
            self.procMap.preNode = orders if (mode != 0) else None
            self.procMap.isOpen = mode
            # 交易成功
            self.Record.insertAll(orders)
            return True

        else:
            if state < 0:
                logger.info(('----账户持仓检查不符合 ----:', state, self.uid, mode))

            elif state == 4:
                # 等待状态
                logger.info(('----检查通过，交易提交中 挂起 ----:', state, self.uid, mode))
                self.banCodeList.append(self.procMap.codes[0])
            else:
                logger.info(('----检查通过，交易不成功 ----:', state, self.uid, mode))
                self.banCodeList.append(self.procMap.codes[0])
            time.sleep(3)
            return False

def main():
    pass
    #addOrder(docs)

if __name__ == '__main__':
    main()
