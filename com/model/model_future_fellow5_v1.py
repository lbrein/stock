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
from multiprocessing import Pool

import numpy as np
from com.ctp.interface_pyctp import interface_pyctp, BaseInfo, ProcessMap
import time
import traceback
import math
import copy
from com.object.mon_entity import mon_tick
from com.model.model_future_ctp import model_future_ctp
import uuid


# 期货混合交易模型
class model_future_fellow5_v1(model_future_ctp):

    def __init__(self):
        # 统一参数
        super().__init__()

        self.iniAmount = 500000  # 单边50万

        self.isAutoAlterPosition = True
        self.isTest = False
        self.isTickSave = False

        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]

        self.banCodeList = []  # 暂时无权限操作的code
        self.isCTPUse = False  # 是否启用CTP接口（并区分
        self.tickIntList = ['powm', 'kdjm', 'mode', 'isopen', 'isstop']

        self.cvsCodes = ['SC', 'IH']
        self.tickInterval = 1
        self.klinePeriod = 14
        self.atrLine = 2
        self.bullLine = 3.5

        self.uidStyle = '%s_14_2.0_%s_3.5_2_%s'
        self.future_Map = []
        self.batchNum = 1
        self.ctpuser = 'simnow1'
        self.methodName = 'fellow'

    def pool(self):
        pool = Pool(processes=6)
        pid = 0
        for codes in self.seperate():
            print('pool send:', pid, len(codes), codes)
            #if 'SC' not in codes: continue
            #self.start(codes)
            try:
                pool.apply_async(self.start, (codes,))
                time.sleep(1)
                pid += 1
                # break
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

    # 按结束时间分割为不同进程
    def seperate(self):
        Base = future_baseInfo()
        codes = Base.getUsedMap(hasIndex=True, isquick=True)
        dd = str(public.getTime(style='%H:%M:%S'))
        maps = {}
        # 按结束时间分组分进程
        for doc in Base.getInfo(codes):
            if doc['code'] in self.banCodeList: continue

            uid = self.uidStyle % (doc['code'], doc['quickkline'][:-1], self.methodName)
            self.future_Map.append(uid.split("_"))
            key = doc['nightEnd']
            if '03:00:00' < key < dd: continue

            if key not in maps.keys():
                maps[key] = []

            maps[key].append(doc['code'])

        for key in maps.keys():
            if len(maps[key]) < 10:
                yield maps[key]
            else:
                l, t = len(maps[key]), len(maps[key]) // 10
                for i in range(t):
                    s, e = i * 10, l if (i + 1) * 10 > l else (i + 1) * 10
                    yield maps[key][s:e]

    def start(self, full_codes):
        # print(full_codes)
        self.Record = future_orderForm()
        if not self.isCTPUse:
            self.Record.tablename = 'future_orderForm_1'

        self.time0 = time.time()

        if self.isTickSave:
            self.Tick = mon_tick()

        self.Rice = interface_Rice()
        # 基础配置信息类
        self.baseInfo = BaseInfo(full_codes, self.Rice)

        # ctp 接口类
        # self.CTP =  interface_pyctp(use=self.isCTPUse, baseInfo=self.baseInfo, userkey=self.ctpuser)

        # 进程控制类
        self.procMap = ProcessMap()

        # 设置交易对的收盘时间
        self.Rice.setTimeArea(self.baseInfo.nightEnd)

        self.indexList = []
        if len(self.indexCodeList) > 0:
            self.Rice.setIndexList(self.indexCodeList)
            self.indexList = [c[0] for c in self.indexCodeList]

        # 按k线类型分拆组，进行K线数据调用
        self.groupByKline(full_codes)
        # 初始化节点
        self.iniNode(full_codes)
        # return
        # 子进程启动
        full_mCodes = self.baseInfo.mainCodes  # 主力合约
        logger.info(("%s start: %s" % (self.__class__.__name__, ",".join(full_mCodes)), self.Rice.TimeArea))

        self.Rice.startTick(full_mCodes, kmap=self.kTypeMap, source='combin', callback=self.onTick)

    # 按K线类型分组
    def groupByKline(self, full_codes):
        self.used_future_Map, self.kTypeMap = [], {}
        for map in self.future_Map:
            code = map[0]
            if map[0] not in full_codes: continue
            self.used_future_Map.append(map)

            # 按k线时间框类型初始化 kTypeMap dict,
            ktype = int(map[3])
            # 字典
            self.kTypeMap[code] = ktype
            # 分组
            if ktype not in self.kTypeMap.keys():
                self.kTypeMap[ktype] = []

            mCode = self.baseInfo.mCode(code)
            if mCode not in self.kTypeMap[ktype]:
                self.kTypeMap[ktype].append(mCode)

    # 初始化节点
    def iniNode(self, full_codes):
        openMap = self.Record.getOpenMap(self.methodName, codes=full_codes, batchNum=1)

        # 非CTP方式
        for map in self.used_future_Map:
            key, uid = map[0], "_".join(map)
            self.procMap.new(uid)  # 初始化进程参数类

            # 初始化品种状态
            if key in openMap:
                self.procMap.setIni(uid, openMap[key])
                logger.info((uid, key, self.procMap.isOpen, self.procMap.status))

    # Tick 响应
    def onTick(self, tick):
        # 计算参数
        # 当前时间
        tt = str(public.getTime(style='%H:%M:%S'))
        for map in self.used_future_Map:
            if map[0] in self.banCodeList: continue

            # 股指期货时间过滤
            if map[0] in self.indexList and '08:59:00' < tt < '09:30:00': continue
            if map[0] not in self.indexList and '13:00:00' < tt < '13:30:00': continue

            self.uid = self.procMap.setUid(map, num=self.batchNum)  # uid
            self.mCodes = [self.baseInfo.mCode(c) for c in self.procMap.codes]  # 当前主力合约代码
            kline = self.procMap.kline

            # 按时长读取k线数据
            dfs = self.Rice.getKline(self.kTypeMap[kline], ktype=kline, key=str(kline) + 'm', num=1)

            # 检查间隔时间
            try:
                # 计算指标
                param = self.paramCalc(dfs, tick)
                #if map[0] =='SC' and self.itemCount % 20 ==0:
                #    print(tick[self.mCodes[0]])

                if param is not None and not np.isnan(param["ma"]):
                    # mongodb Record
                    self.debugT('', param=param)

                    # 执行策略，并下单
                    self.orderCheck(tick, param)

            except Exception as e:
                print(traceback.format_exc())

    def apart(self, PS, ktype):
        apart = math.pow((int(time.time()) % (60 * ktype)) * 1.0 / (60 * ktype), 0.5)
        return PS * apart + PS.shift(1) * (1 - apart)

    # 买入后的最高/最低价
    def getMax(self, df0, s, e, mode):
        if mode > 0:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'close'].max()
        else:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'close'].min()

    def setLastClose(self, close):
        tt = int(time.time()) // 5
        rec = self.procMap.get('Min5')
        if tt != rec:
            self.procMap.set('Min5', tt)
            ss = self.procMap.get('Min5Record')
            if ss is None:
                ss = []
                ss.append(close)
            else:
                ss.append(close)
                if len(ss) > 3: ss.pop(0)
            self.procMap.set('Min5Record', ss)

    def getLastClose(self):
        ss = self.procMap.get('Min5Record')
        if ss is None: return 0, 0
        if len(ss) < 3: return ss[-1], 0
        if len(ss) == 3: return ss[-1], ss[0]

    pointColumns = ['powm', 'diss', 'fall']
    def point(self, row, row_1, tsize):
        ma, close, open, high, low, std, stdc, atr, atrc, dd = \
            (row[key] for key in
             "ma,close,open,high,low,std,stdc,atr,atrc,datetime".split(","))

        open = row_1['close']
        tt = str(public.getTime(style='%H:%M:%S'))
        kline = self.procMap.kline

        BL = 4.0 - float(kline)/10

        #
        apart = (60 * kline) - (int(time.time()) % (60 * kline))
        if '09:00:00' <= tt < '09:15:00' or '21:00:00' <= tt < '21:10:00' or '14:45:00' <= tt < '15:00:01':
            BL += 5.0

        if apart > 10:
            BL += 0.5

        sign = 1 if high > (ma + 2 * std) else -1 if low < (ma - 2 * std) else 0

        # 计算长度、回撤率
        max = high if sign > 0 else low if sign < 0 else close
        diss = 0 if std == 0 else abs(max - ma) / std
        fall = abs(max - close) / abs(max - open) if max!=open else 0

        # 条件
        opt0 = (diss > BL) and atrc > self.atrLine
        opt1 = (diss > (BL - 0.25)) and (atrc > (self.atrLine + 1.5))

        # 超长
        opt2 = diss > (BL + 1)

        # 超回收
        opt3 = diss > (BL - 0.35) and (fall > 0.5)

        opt8 = (abs(max - close) > 3 * tsize)
        opt9 = (atr / ma) * 10000 > 8 and std > 3 * tsize

        opt10 = stdc > 1.25
        powm = 0
        if opt8 and opt9:
            if opt10:
               powm = sign if opt0 else sign * 2 if opt1 else sign * 4 if opt3 else 0

            elif opt2:
                powm = sign * 3

        # 设置整点数据
        if int(time.time()) % 5 == 0:
            self.setLastClose(close)

        return powm, diss , fall

    # 计算布林参数
    klinecolumns = ['high', 'open', 'volume', 'close', 'low']
    def paramCalc(self, dfs, cur):
        if len(dfs) == 0: return None
        period = self.klinePeriod
        c0 = cur[self.mCodes[0]]
        c0['close'] = c0['last']

        # 去掉当前的即时k线
        df0 = copy.deepcopy(dfs[self.mCodes[0]].iloc[-35:])
        # print(len(df0))
        # 计算相关参数
        columns = self.klinecolumns

        #if self.procMap.codes[0]=='SC':
        #    print(df0)

        # 添加即时K线
        df0.loc[public.getDatetime()] = pd.Series([c0[key] for key in columns], index=columns)
        close = df0["close"]
        df0["datetime"] = df0.index

        df0["ma"] = ma = ta.MA(close, timeperiod=period)
        df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)

        df0["stdc"] = std / ta.MA(std, timeperiod=period)

        df0['atr'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=period)
        df0['atrc'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=1) / df0['atr']
        df0["slope"] = ta.LINEARREG_SLOPE(ma, timeperiod=5)

        # kdj顶点
        kdjK, kdjD = ta.STOCH(df0["high"], df0["low"], close,
                              fastk_period=9, slowk_period=3, slowk_matype=1, slowd_period=3,
                              slowd_matype=1)

        df0["kdj_d2"] = kdj_d2 = kdjK - kdjD
        df0["kdjm"] = kdj_d2 * kdj_d2.shift(1)
        df0["kdjm"] = df0.apply(lambda row: self.turn(row['kdjm'], row['kdj_d2'], 1), axis=1)

        b0 = self.baseInfo.doc(self.procMap.codes[0])

        self.df0 = df0
        param_1 = copy.deepcopy(df0.iloc[-2]).to_dict()
        param = copy.deepcopy(df0.iloc[-1]).to_dict()

        powm, diss, fall  = self.point(param, param_1, b0['tick_size'])

        # 记录最近5/15秒点的数据
        param.update(c0)
        param.update({
            "powm": powm,
            "diss": diss,
            "fall":fall,
            #"last5":last5,
            #"last15": last15,
            "now": c0['now'],
            "p_l": c0["asks"][0],
            "p_h": c0["bids"][0],
        })
        return param

    pub = 0
    itemCount = 0

    def orderCheck(self, cur, param):
        isOpen, isRun, isstop = self.procMap.isOpen, False, 0

        powm, atr, date, kdjm, close = (param[key] for key in "powm,atr,datetime,kdjm,close".split(","))

        self.preNode = self.procMap.preNode

        if isOpen == 0:
            if powm != 0:
                isRun, isOpen = True, int(-powm)

        elif isOpen != 0 and self.preNode is not None:
            # 止盈止损
            kline = self.procMap.kline
            preP, s = self.preNode[0]['price'], self.preNode[0]['createdate']
            #
            apart = (60 * kline) - (int(time.time()) % (60 * kline))
            # 保持时间
            keeps = len(self.df0[self.df0['datetime'] > str(s)])

            if isOpen * kdjm < 0 and apart < 10 and keeps > 3:
                isRun, isOpen, isstop = True, 0, 1

            # 反转
            elif isOpen * powm > 0:
                if self.order(cur, 0, param, isstop=3):
                    isRun, isOpen = True, int(-powm)

            else:
                # 止损
                mp = self.getMax(self.df0, s, date, isOpen)
                sign = np.sign(isOpen)
                mp = close if np.isnan(mp) else mp

                if sign * (preP - close) > 1.5 * atr:
                    isRun, isOpen, isstop = True, 0, 4

                elif not np.isnan(mp) and (sign * (mp - close) > 3 * atr):
                    isOpen, isRun, isstop = 0, True, 5

        if isRun:
            # logger.info((self.procMap.codes[0], param))
            self.order(cur, isOpen, param, isstop=isstop)


    def order(self, cur, mode, param, isstop=0):
        # 当前tick对
        n0 = cur[self.mCodes[0]]

        # future_baseInfo 参数值
        b0 = self.baseInfo.doc(self.procMap.codes[0])
        times0 = b0["contract_multiplier"]

        preNode = self.procMap.preNode

        # 每次交易量
        code = self.procMap.codes[0]
        if preNode is not None:
            v0 = preNode[0]["hands"]
        else:
            v0 = self.iniAmount / times0 / n0['close']
            v0 = int(v0 + 0.2) if v0 > 0.8 else 1
            if v0 == 0: v0 = 1

        # 开仓 1/ 平仓 -
        isOpen = 0 if mode == 0 else 1

        # 买 / 卖 ,  若mode=0. 则按持仓方向平仓操作
        isBuy = -preNode[0]["mode"] if (mode == 0 and preNode is not None) else mode

        # 费率
        fee0 = (v0 * b0["ratio"]) if b0["ratio"] > 0.5 else (
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
            "name": code,
            "symbol": self.baseInfo.ctpCode(n0["code"]),
            "price": price,
            "vol": v0 * times0,
            "hands": v0,
            "ini_hands": v0,  # 提交单数
            "ini_price": price,  # 提交价格
            "mode": int(isBuy),
            "isopen": isOpen,
            "isstop": isstop,
            "fee": fee0,
            "income": 0.0,
            "rel_price": price,
            "delta": param['diss'],
            "rel_std": param['std'],
            "widthDelta": param['atr'],
            # 'stop_price': 0 if mode == 0 else dp,
            "batchid": self.procMap.batchid,
            "status": 0,  # 定单P执行CT返回状态
            "method": self.methodName,
            "uid": self.uid
        }

        if self.isTickSave:
            monR = copy.deepcopy(param)
            monR.update(doc)
            monR.update({"type": "record"})
            self.debugR(monR)

        # 下单并记录
        if not self.isTest:
            return self.record([doc], mode)

def test():
    pass


def main():
    obj = model_future_fellow5_v1()
    obj.pool()


if __name__ == '__main__':
    main()
