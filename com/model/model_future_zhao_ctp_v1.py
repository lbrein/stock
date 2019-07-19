# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

      期货单品种交易

"""

from com.base.public import public, logger
import pandas as pd
import talib as ta

from com.object.obj_entity import future_orderForm, train_total, future_baseInfo
from multiprocessing import Pool
from com.ctp.interface_pyctp import ProcessMap
from com.data.interface_Rice import interface_Rice
from com.model.model_future_ctp import model_future_ctp
import uuid
import numpy as np
import traceback
import copy
import time

# 期货混合交易模型
class model_future_zhao_v1(model_future_ctp):

    def __init__(self):
        # 统一参数
        self.isWorking = True # 是否正式运行
        self.isAutoAlterPosition = True
        self.isTest = False
        #self.isCTPUse = False  # 是否启用CTP接口（并区分

        self.isTickSave = False
        self.topUse = False  # 启用定时统计列表
        self.volumeCalcType = 0  # 交易量计算类别

        self.banCodeList = ['PB', 'SN']  # 暂时不操作的code（不列交易量低的)
        self.longCodeList = ['CU', 'A', 'B', 'CS', 'JD', 'AP', 'L', 'RM', 'EG', 'M', 'ZC', 'RU', 'SC']  # 只做多仓的list
        self.shortCodeList = ['ZN', 'I', 'RB', 'FG', 'V', 'SR', 'AU']  # 只做空仓的list

        self.oneCodeList = ['SC', 'IH', 'IF', 'IC']  # 最低为1手单的
        self.indexCodeList = []  # 指数期货对应K线指数表
        self.tickIntList = ['mam', 'interval', 'out_s', 'trend', 'isout', 'isout3', 'isout5','mode', 'isopen', 'isstop', 'status']
        self.tickInterval = 100  # 间隔处理时间
        self.klinecolumns = ['high', 'open', 'volume', 'close', 'low']
        self.tmpStopList = [('SC', -1)]

        self.ctpuser = 'zhao'
        self.methodName = 'mZhao'  # 策略名称

        self.relativeMethods = ['mZhao', 'mZhao55']

        self.iniAmount = 3080000  # 初始总金额
        self.stopLine = 0.0025

        self.sourceType = 'snap'
        self.timeInterval = 0.5

        self.tangStartPeriod0 = 18 - 1
        self.tangStartPeriod1 = 27 - 1
        self.tangDropPeriod = 40 - 1
        self.tangStartPeriod55 = 34 - 1
        self.atrPeriod = 21

        self.uidStyle = '%s_40_2.0_1_0_%s'

        self.batchNum = 1  # 批处理的交易数量
        self.topNumbers = 30  # 最大新币对 (监控币对还包括历史10和未平仓对)
        self.minVol = 100
        self.orderFormTest = 'future_orderForm_test'
        self.topTableName = 'train_total_zhao'
        self.widthTimesPeriod = 3
        self.future_Map = []
        self.noneUsed = []  # 不再继续的商品对

        self.Record = None

    def iniWorking(self):
        if self.isWorking:
            self.orderFormTest = 'future_orderForm'
            self.isCTPUse = True
            self.ctpuser = 'zhao'

        else:
            self.orderFormTest = 'future_orderForm' # 临时
            self.isCTPUse = False
            self.ctpuser = 'simnow1'

    def pool(self):
        pool = Pool(processes=6)
        self.iniWorking()
        # 初始化codes列表
        self.filterCodes()
        #share = Manager2()

        CTP = None
        Rice = None

        pid = 0
        for codes in self.seperate():
            #if 'JD' not in codes: continue
            #print(codes)
            print('pool send:', pid, len(codes), codes)
            #self.start(codes, Rice, CTP)
            try:
                pool.apply_async(self.start, (codes, Rice, CTP))
                time.sleep(3)
                pid += 1
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()


    # 同时统计未平仓的对，加入监测，待平仓后不再开仓
    def filterCodes(self):
        # 查询每周统计表
        if self.topUse:
            Top = train_total()
            Top.tablename = self.topTableName
            self.future_Map = Top.last_top1(num=self.topNumbers, method=self.methodName, minVol=self.minVol,
                                            ban=self.banCodeList)
        else:
            Base = future_baseInfo()

            codes = Base.getUsedMap(hasIndex=len(self.indexCodeList) > 0)
            for code in codes:
                if not code in self.banCodeList:
                    uid = self.uidStyle % (code, self.methodName)
                   #  if code in ['IC']: print(uid)
                    self.future_Map.append(uid.split("_"))

        num0 = len(self.future_Map)
        Record = future_orderForm()
        if not self.isWorking: Record.tablename = self.orderFormTest

        # 添加未平仓，并不再开仓
        codesList = [c[0] for c in self.future_Map]
        # 当前开仓清单
        openCodes = Record.currentOpenCodes(method=self.methodName, batchNum=self.batchNum)
        if openCodes is not None:
            for map in openCodes:
                if not map[0] in codesList and not map[0] in self.banCodeList:
                    self.future_Map.append(map[:-3])
                    self.noneUsed.append(map[0])

        # 调仓处理
        dd = str(public.getTime(style='%H:%M:%S'))
        if "10:15:00" < dd < "11:00:00":
            self.alterPosition(Record, openCodes)

        logger.info(('总监控币对：', len(self.future_Map), ' 新列表:', num0, ' 未平仓:', len(self.noneUsed)))
        logger.info(([n[0] for n in self.future_Map]))
        logger.info(('暂停币对:', len(self.noneUsed), [n for n in self.noneUsed]))

    # Tick 响应
    def onTick(self, tick):
        # 计算参数
        ttt = str(public.getTime(style='%H:%M:%S'))
        opt0 = ("10:15:00" <= ttt < "10:30:00") or ("13:00:00" < ttt < "13:30:00")
        opt1 = ("09:00:00" < ttt < "09:30:00")
        #print(opt0, opt1)
        for map in self.used_future_Map:
            if map[0] in self.banCodeList: continue

            # 上午休息时间，排除指数期货运行时间
            opt2 = len(self.indexList) > 0 and map[0] in self.indexList
            if (opt2 and opt1) or (not opt2 and opt0): continue

            self.uid = self.procMap.setUid(map, num=self.batchNum)  # uid
            self.mCodes = [self.baseInfo.mCode(c) for c in self.procMap.codes]  # 当前主力合约代码

            kline = self.procMap.kline
            # 按时长读取k线数据
            dfs = self.Rice.getKline(self.kTypeMap[kline], ktype=kline, key='1d', num=120)

            # 检查间隔时间
            try:
                # 计算指标
                param = self.paramCalc(dfs, tick)

                if param is not None and not np.isnan(param["ma20"]):
                    # mongodb Record
                    self.debugT('', param=param)
                    # 执行策略，并下单
                    self.orderCheck(tick, param)
                    pass
            except Exception as e:
                print(traceback.format_exc())

    def isout0(self, row):
        close, ma20, ma55 = (row[key] for key in "close,ma20,ma55".split(","))
        return 0 if np.isnan(ma55) else 1 if close > ma55  else -1

    def isout(self, row, pos):
        close, ma20, ma55, trend = (row[key] for key in "close,ma20,ma55,trend".split(","))
        pt = pos if pos!=0 else trend if trend!=0 else 1 if ma20 < ma55 else -1
        return 1 if close > ma55 and pt > 0 else -1 if close < ma55 and pt < 0 else 0

    def k_fix(self, row, mode):
        close, open, high, low = (row[key] for key in ['close', 'open', 'high', 'low'])
        if open == 0: return close
        lim, rate = 0.003, 0.075

        d0 = abs(close - open) / open
        if mode == 1:
            if close > open and open != 0:
                trend = 1 if high == open else abs(close - open) / (high - open)
                opt = (d0 < lim and trend < rate ) or (d0 > lim and trend < rate * 2)
                return close if opt else high
            else:
                return high

        elif mode == -1:
            if close < open and open != 0:
                trend = 1 if low == open else abs(open - close) / (open - low)
                opt = (d0 < lim and trend < rate) or (d0 > lim and trend < rate * 2)
                return close if opt else low
            else:
                return low

    def isMam(self, row):
        #
        mad, mac, mac2, mac3 = (row[key] for key in "mad,mac,mac2,mac3".split(","))
        opt = np.isnan(mad) or mad > 0 or mac == 0
        # 为零时同向偏转
        opt1 = (mad ==0 and (mac * mac2 > 0 or (mac2==0 and mac * mac3) > 0))
        return 0 if (opt or opt1) else 1 if mac > 0 else -1

    # 计算布林参数
    def paramCalc(self, dfs, cur):
        # 分钟线
        if len(dfs) == 0: return None

        c0 = cur[self.mCodes[0]]
        c0['close'] = c0['last']

        # 去掉当前的即时k线
        df0 = copy.deepcopy(dfs[self.mCodes[0]])

        # 添加即时K线
        df0.loc[public.getDatetime()] = pd.Series([c0[key] for key in self.klinecolumns], index=self.klinecolumns)

        close = df0["close"]

        df0["datetime"] = df0.index

        # MA指标
        df0["ma10"] = ma10 = ta.MA(close, timeperiod=10)
        df0["ma20"] = ma20 = ta.MA(close, timeperiod=20)
        df0["ma55"] = ta.MA(close, timeperiod=55)

        # 21日平均 ATR
        df0['atrb'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=21)

        # 修正不正常K线
        df0['high'] = df0.apply(lambda row: self.k_fix(row, 1), axis=1)
        df0['low'] = df0.apply(lambda row: self.k_fix(row, -1), axis=1)

        # 计算ma10-ma20 穿越线间距
        df0['mac'] = mac = ma10 - ma20
        df0['mac2'] = mac.shift(2)
        df0['mac3'] = mac.shift(3)

        # isPoint0
        df0['mad'] = mac * mac.shift(1)
        df0['mam'] = mam = df0.apply(lambda row: self.isMam(row), axis=1)

        minidx, maxidx = ta.MINMAXINDEX(mam, timeperiod=75)
        df0['interval'] = abs(minidx - maxidx)

        # 唐奇安线18日
        df0['tu_s'] = ta.MAX(df0['high'].shift(1), timeperiod=self.tangStartPeriod0)
        df0['td_s'] = ta.MIN(df0['low'].shift(1), timeperiod=self.tangStartPeriod0)

        # 唐奇安线27日
        df0['tu_s1'] = ta.MAX(df0['high'].shift(1), timeperiod=self.tangStartPeriod1)
        df0['td_s1'] = ta.MIN(df0['low'].shift(1), timeperiod=self.tangStartPeriod1)

        # 40日低点
        ld = close[close.notnull()]
        dp = self.tangDropPeriod if len(ld) > self.tangDropPeriod else len(ld) - 1

        df0['tu_d'] = ta.MAX(df0['high'].shift(1), timeperiod=dp)
        df0['td_d'] = ta.MIN(df0['low'].shift(1), timeperiod=dp)

        # 唐奇安线34日
        df0['tu_34'] = ta.MAX(df0['high'].shift(1), timeperiod=self.tangStartPeriod55)
        df0['td_34'] = ta.MIN(df0['low'].shift(1), timeperiod=self.tangStartPeriod55)

        # df0['tu_e'] = ta.MAX(df0['high'].shift(1), timeperiod=self.tangEndPeriod)
        # df0['td_e'] = ta.MIN(df0['low'].shift(1), timeperiod=self.tangEndPeriod)

        # 计算穿越值
        """    
        fp, fd = 27, 5
        code = self.procMap.codes[0]
        posTrend = 0 if code not in self.trendMap else self.trendMap[code]['trend']

        out = df0.apply(lambda row: self.isout0(row), axis=1)
        df0['out_s'] = ta.SUM(out, timeperiod=fp)
        df0['trend'] = df0['out_s'].apply(lambda x: -1 if x > fd else 1 if x < -fd else 0)

        df0['isout'] = isout = df0.apply(lambda row: self.isout(row, posTrend), axis=1)
        df0['isout3'] = ta.SUM(isout, timeperiod=3)
        df0['isout5'] = ta.SUM(isout, timeperiod=5)
        """
        df0['isout3'] = 0
        df0['isout5'] = 0

        # 计算关键转折点
        # df0.fillna(0, inplace=True)
        param = copy.deepcopy(df0.iloc[-1]).to_dict()

        self.df0 = df0
        param.update({
            "p_l": c0["asks"][0],
            "p_h": c0["bids"][0]
        })
        return param

    def getTrend(self, code):
        return 1 if code in self.longCodeList else -1 if code in self.shortCodeList else 0

    def getStatus(self, methods, code):
        if self.Record is None:
            self.Record = future_orderForm()

        posMode = self.Record.openMode(methods, code)
        lastStop = self.Record.lastStop(methods, code)
        trend = self.getTrend(code)

        status = -1
        if posMode[1] == 0:
            if lastStop[0] == 6 or (lastStop[0] == 0 and trend == 0):
                status = 0

            elif lastStop[0] in [3, 5] or (lastStop[0] == 0 and trend != 0):
                status = 1

            elif lastStop[0] == 2:
                status = 2

        elif posMode[1] == 1:
            if posMode[0].find('55') > -1:
                status = 5

            elif lastStop[0] == 3:
                status = 3.5

            else:
                status = 3

        elif posMode[1] == 2:
            status = 4

        return status

    def orderCheck(self, cur, param):
        isOpen, isRun, isstop = self.procMap.isOpen, False, 0

        close, date, ma10, ma20, ma55, atrb, tu_s, td_s, tu_s1, td_s1, tu_d, td_d, interval = (param[key] for key in
                                                                                               "close,datetime,ma10,ma20,ma55,atrb,tu_s,td_s,tu_s1,td_s1,tu_d,td_d,interval".split(
                                                                                                   ","))
        self.preNode = self.procMap.preNode
        code = self.procMap.codes[0]
        if np.isnan(tu_d): return None

        if isOpen == 0:
            """
                开盘策略 
                1、ma10>ma20 并且 > 18日均线 
                2、
            """
            opt1 = (ma10 > ma20 and close > tu_s) or (ma10 < ma20 and close < td_s)
            opt2 = not (np.isnan(td_s1)) and ((close < td_s1) or (close > tu_s1))

            if opt1 and interval > 12:
                isOpen, isRun = np.sign(ma10 - ma20), True

            elif opt2:
                isOpen, isRun = -2 if close < td_s1 else 2, True

            if isRun:
                lastStop = self.Record.lastStop(self.relativeMethods, code)

                # 止损按指定方向开仓，（3、6）任意方向
                opt3 = lastStop[0] == 2  #反向平仓不开仓只开系统2
                opt4 = lastStop[0] in (0, 3, 5) and ((isOpen > 0 and code in self.shortCodeList) or (isOpen < 0 and code in self.longCodeList))
                # 状态5
                posMode = self.Record.openMode(self.relativeMethods, code)
                opt5 = posMode[1] == 1 and (posMode[0].find('55') > -1)

                # 临时特殊补充过滤：
                if opt3 or opt4 or opt5:
                    isOpen, isRun = 0, False

        elif isOpen != 0:
            """
                交叉点: sarm结束点 
                1、结束突变状态，变为布林带处理
                2、根据节点状况开平新仓，状态为6 
            """
            if self.preNode is not None:

                opt1 = (ma10 > ma20 and close > tu_s and isOpen < 0) or (ma10 < ma20 and close < td_s and isOpen > 0)
                opt2 = (close < td_s1 and isOpen > 0) or (close > tu_s1 and isOpen < 0)
                # 持仓时间
                keepDays = len(self.df0[self.df0['datetime'] > str(self.preNode[0]['createdate'])])

                if (keepDays > 13) and ((opt1 and interval > 12) or opt2):
                       # 反向平仓先有系统2，先平系统2，再延时平系统1
                       if self.Record.openMode(self.relativeMethods, code)[1]==1:
                          lastStop = self.Record.lastStop(self.relativeMethods, code)

                          if lastStop[0]!=3:
                             isOpen, isRun, isstop = 0, True, 2

                          else:

                             # 隔日检查反向平仓1
                             d, t = public.parseTime(str(lastStop[1]), style='%Y-%m-%d'), public.parseTime(str(lastStop[1]), style='%H:%M:%S')
                             if "15:01:00" < t <= "23:59:59":  d = public.getDate(1, start=d)
                             if public.getDatetime() > (d +' 15:01:00'):
                                 isOpen, isRun, isstop = 0, True, 2

                else:
                    # stop_price 止损
                    stop_price = td_d if isOpen > 0 else tu_d
                    opt5 = not np.isnan(stop_price) and (isOpen * (close - stop_price)) < 0
                    # 开仓止损线止损
                    if opt5:
                        isOpen, isRun, isstop = 0, True, 5

        if isRun:
            logger.info(('orderChecked', code, isOpen))
            self.order(cur, isOpen, param, isstop=isstop)

    def order(self, cur, mode, param, isstop=0):
        # 当前tick对
        n0 = cur[self.mCodes[0]]

        # future_baseInfo 参数值
        b0 = self.baseInfo.doc(self.procMap.codes[0])
        times0 = b0["contract_multiplier"]

        preNode = self.procMap.preNode

        # 每次交易量
        dp = param['td_d'] if mode > 0 else param['tu_d']
        if np.isnan(dp) or dp == n0['close']:
            dp = n0['close'] * (1 - 0.2)

        code = self.procMap.codes[0]
        if preNode is not None:
            v0 = preNode[0]["hands"]
        else:
            if code in self.oneCodeList:
                v0 = 1
            elif self.volumeCalcType == 0:
                v0 = (self.iniAmount * self.stopLine / abs(n0['close'] - dp) / b0["contract_multiplier"])

            elif self.volumeCalcType == 1:
                dd = 3 * param['atrb']
                v0 = (self.iniAmount * self.stopLine / dd / b0["contract_multiplier"])
            else:
                v0 = 0

            v0 = int(v0 + 0.2) if v0 > 0.8 else 1

        # 开仓 1/ 平仓 -
        isOpen = 0 if mode == 0 else 1

        # 买 / 卖 ,  若mode=0. 则按持仓方向平仓操作
        isBuy = -preNode[0]["mode"] if (mode == 0 and preNode is not None) else mode
        # 费率
        fee0 = (v0 * b0["ratio"] * 1.1) if b0["ratio"] > 0.5 else (
                b0["ratio"] * 1.1 * v0 * times0 * (n0["asks"][0] if isBuy == 1 else n0["bids"][0]))

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
            "ini_price": n0["asks"][0] if isBuy == 1 else n0["bids"][0],  # 提交价格
            "mode": int(isBuy),
            "isopen": isOpen,
            "isstop": isstop,
            "fee": fee0,
            "income": 0,
            "rel_price": param["p_l"] if isBuy == 1 else param["p_h"],
            'stop_price': 0 if mode == 0 else dp,
            "batchid": self.procMap.batchid,
            "status": 0,  # 定单P执行CT返回状态
            "method": self.methodName,
            "uid": self.uid,
            "memo": ''

        }
        """
        monR = copy.deepcopy(param)
        monR.update(doc)
        monR.update({"type": "record"})
        self.debugR(monR)
        """

        # 下单并记录
        if not self.isTest:
            return self.record([doc], mode)

def test():
    obj = model_future_zhao_v1()
    obj.procMap  = ProcessMap()
    obj.Rice = interface_Rice()
    Rec = future_orderForm()
    openMap = Rec.getOpenMap('zhao55', codes=['NI'], batchNum=1)
    # print(openMap)
    key, uid = 'NI', 'NI_40_2.0_1_0_zhao55'
    obj.procMap.setIni(uid, openMap[key], status=0)

    doc = {'createdate': '2019-05-24 14:39:27', 'code': 'NI1907', 'name': 'NI', 'symbol': 'ni1907',
           'price': 99250.0, 'vol': 8.0, 'hands': 8, 'ini_hands': 8.0, 'ini_price': 99010.0, 'mode': 5,
           'isopen': 0, 'isstop': 3, 'fee': 48.0, 'income': 0, 'rel_price': 99010.0, 'stop_price': 0,
           'batchid': '54093cf8-7139-11e9-82c1-1c1b0d16fcc2',
           'status': 6, 'method': 'zhao55', 'uid': 'NI_40_2.0_1_0_zhao55',
           'session': 1466737504, 'front': 11, 'direction': b'0', 'orderID': '2137451'}
    obj.setIncome([doc], 0)
    Rec.insert(doc)

def main():
    obj = model_future_zhao_v1()
    obj.pool()

if __name__ == '__main__':
    main()
    #test()