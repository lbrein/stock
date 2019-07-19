# -*- coding: utf-8 -*-

""" 
Created on  2018-05-03 
@author: lbrein
      
        sh50 期权价格记录和监控

"""

from com.base.public import public, config_ini, logger
import numpy as np
import talib as ta
from com.data.interface_Rice import interface_Rice
from com.object.obj_entity import sh50_orderForm
from com.data.data_interface import sinaInterface, s_webSocket
import copy
import pandas as pd
from com.model.model_sh50_MA import model_sh50_ma
import math

class model_sh50_bull(model_sh50_ma):
    def __init__(self):
        # 新浪接口
        super().__init__()

        self.codes = ['510050.XSHG']

        self.iniAmount = 40000
        self.multiplier = 10000
        self.currentVol = 0
        self.currentPositionType = 0
        self.scale = 2.0
        self.ratio = 2
        self.shift = 0.002
        self.scaleDiff2 = 1.0
        self.isParamSave = True

        self.ws = s_webSocket("ws://%s:8090/etf50boll" % config_ini.get("ws.host", session="db"))
        self.titleList = ['createdate', 'code', 'name', 'isBuy', 'vol',  'price', 'ownerPrice']
        self.table_head = ""
        self.table_end = ""
        self.method = 'bull4'
        self.viewItemList = ['createTime', 'close', 'pow', 'top', 'low', 'sarm', 'kdjm', 'powm', 'isopen']

    def start(self):
        self.uid ="510050-XSHG_15_15_0.5_2.0"
        map = self.uid.split('_')


        self.timePeriods = int(map[1])
        self.klineType = int(map[2])
        self.period = str(self.klineType)+'m'
        self.powline = float(map[3])
        self.ktypeMap = {self.codes[0]: int(map[2])}
        self.turnline = float(map[4])
        self.tickParam, self.status = None, 0
        self.iniNode()

        logger.info(('etf50 - bull_trend stage start', self.uid))
        self.Rice.startTick(self.codes, callback=self.onTick, source='kline', kPeriod=self.period, kmap =self.ktypeMap)

        # 初始化节点

    def paramCalc(self, df0, t0):
        # 替换即时k线
        df = copy.deepcopy(df0[:-1])
        # 计算相关参数
        df.loc[public.getDatetime()] = t0

        df = self.add_stock_index(df)
        df['createTime'] = df.index
        param = df.iloc[-1].to_dict()
        return param

    def stand(self, ser):
        return ser / ser.abs().mean()

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    pointColumns= ['pow', 'powm', 'trend', 'isopen']
    def point(self, row, ktype=15):
        close, wd1, s1, s2, atr, atr1, high, low, open, close, top, lower, p_h, p_l, tu, td = (row[key] for key in
                                                                                               "close,widthDelta,slope,slope30,atr,atr1,high,low,open,close,top,lower,p_h,p_l,tu,td".split(
                                                                                                   ","))
        if close > open:
            trend = 1 if high == open else ((close - open) / (high - open) + 0.5)
        else:
            trend = -1 if low == open else ((close - open) / (open - low) - 0.5)

        tr = (4 - atr * atr1) if atr > 2 else atr * atr1
        if tr < 0: tr = 0
        # pow
        pow = (wd1 + abs(s1 + s2 + trend * tr)) / 4

        powm = 2 * np.sign(s1) if (pow > self.turnline and (close > tu or close < td)) else np.sign(s1) if (
                pow > 0.25 or abs(s1) > 0.5) else 0

        isopen = -1 if p_h > top else 1 if p_l < lower else 0

        columns = self.pointColumns

        return pd.Series([pow, powm, trend, isopen], index=columns)

    def sd(self, x):
        s = round(x * 1.5, 0) / 10
        return np.sign(s) * 0.8 - 0.1 if abs(s) > 0.8 else s - 0.1

    def add_stock_index(self, df0):
        ktype, period, scale = self.klineType, self.timePeriods, self.scale
        close = df0["close"]
        df0["ma"] = ma = ta.MA(close, timeperiod=period)
        df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)
        # 上下柜
        # bullWidth
        df0["bullwidth"] = width = (4 * std / ma * 100).fillna(0)

        # 近三分钟width变动
        df0["widthDelta"] = self.stand(ta.LINEARREG_SLOPE(width, timeperiod=3))
        df0["slope"] = self.stand(ta.LINEARREG_SLOPE(ta.MA(close, timeperiod=3), timeperiod=2))
        df0["slope30"] = self.stand(ta.LINEARREG_SLOPE(ta.MA(close, timeperiod=30), timeperiod=3))
        #df0["rsi"] = rsi = ta.RSI(close, timeperiod=14)
        df0['sd'] = sd = df0['slope30'].apply(lambda x: self.sd(x))


        df0['atrb'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=period)
        df0['atr'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=1) / ta.ATR(df0['high'], df0['low'], close,
                                                                                   timeperiod=period)
        df0['atr1'] = self.stand(ta.ATR(df0['high'], df0['low'], close, timeperiod=period))

        # 唐奇安线
        df0['tu'] = ta.MAX(df0['high'].shift(1), timeperiod=20)
        df0['td'] = ta.MIN(df0['low'].shift(1), timeperiod=20)

        # SAR 顶点
        sar = ta.SAR(df0["high"], df0["low"], acceleration=0.04, maximum=0.25)
        df0['sard'] = sard = close - sar
        df0['sarm'] = sard * sard.shift(1)
        df0['sarm'] = df0.apply(lambda row: self.turn(row['sarm'], row['sard'], 1), axis=1)

        # kdj顶点
        kdjK, kdjD = ta.STOCH(df0["high"], df0["low"], close,
                              fastk_period=4, slowk_period=3, slowk_matype=2, slowd_period=3,
                              slowd_matype=2)

        df0["kdj_d2"] = kdj_d2 = self.apart(kdjK, ktype) - self.apart(kdjD, ktype)
        df0["kdjm"] = kdj_d2 * kdj_d2.shift(1)
        df0["kdjm"] = df0.apply(lambda row: self.turn(row['kdjm'], row['kdj_d2'], 1), axis=1)

        df0["top"], df0["lower"] = ma + (scale - sd) * std, ma - (scale + sd) * std

        df0["p_l"] = close * (1 + self.shift)
        df0["p_h"] = close * (1 - self.shift)

        df1 = df0.apply(lambda row: self.point(row, ktype), axis=1)
        for key in self.pointColumns: df0[key] = df1[key]

        return df0

    def orderCheck(self, t0, param, apart=0):
        self.tickParam = param
        period, ini, unit = self.timePeriods, self.iniAmount / self.multiplier, self.multiplier
        vol, pos, status = self.currentVol, self.currentPositionType, self.status

        ma, close, p_l, p_h, top, lower, std, kdjm, sarm, pow, powm, atrb, sd, s1 = (
            param[key] for key in
            "ma,close,p_l,p_h,top,lower,std,kdjm,sarm,pow,powm,atrb,sd,slope".split(","))

        mode, isBuy, isRun = 0, 0, False

        if self.preNode is not None:  mode = self.preNode['mode']

        sd2 = (self.scaleDiff2 - 1.2 * sd) * std / 2
        pd2 = self.turnline - 0.25

        if status == 0 and powm in [2, -2]:
            """
               突变点处理 
               将布林带策略切换为策略 
            """
            status = np.sign(powm)
            if vol > 0 and pos * status < 0:
                if self.order(param, -1, 4, ini):
                    isBuy, mode, isRun = 1, int(4 * status), True

            elif vol == 0:
                isBuy, mode, isRun = 1, int(3 * status), True

        elif sarm * status < 0 or (abs(mode) in [3] and mode * sarm < 0):
            """
                交叉点: 快慢线穿越处理
                1、结束突变状态，变为布林带处理
                2、根据节点状况开平新仓，状态为6 
            """
            # 结束macd状态
            if vol > 0:
                pi, po = self.preNode['price'], self.preNode['pos']
                if self.order(param, -1, 3, ini):
                    isRun = False
                    # 平仓后再开
                    if pi != 0 and (close - pi) * po / pi > 0.015:
                        isBuy, mode, isRun = 1, -po, True

            status = 0

        elif status == 0:
            """
                BULIN策略
            """
            if vol == 0:
                # 突变状态开始
                # 大于上线轨迹
                if p_h >= top and powm == 0:
                    mode, isRun = -1, True

                elif p_l <= lower and powm == 0:
                    mode, isRun = 1, True

                elif (p_h + sd2) >= top and pow < pd2 and kdjm < 0:
                    mode, isRun = -2, True

                elif (p_l - sd2) <= lower and pow < pd2 and kdjm > 0:
                    mode, isRun = 2, True

                isBuy = 1
                # 平仓

            else:
                sign = pos
                cond3 = sign * (close - ma)
                #
                if cond3 >= -sd2 and sign * kdjm < 0:
                    if ((p_h + sd2) >= top or (p_l - sd2) <= lower) and pow < pd2:
                         if self.order(param, -1, 1 , ini):
                            isBuy, mode, isRun = 1, 1, True
                    else:
                         isBuy, mode, isRun = -1, 1, True

                elif cond3 >= 0 and pow ==0:
                    isBuy, mode, isRun = -1, 0, True

                else:
                    # 即时收益超出 atr 2倍
                    if self.preNode is not None:
                        print(self.preNode)
                        pd = self.preNode['price']
                        if (sign * (pd - close) > 2 * atrb) and sign * kdjm < 0:
                            isBuy, mode, isRun = -1, 7, True

        self.debugT((param, isBuy, mode, vol), n=1)


        if isRun:
            if self.order(param, isBuy, mode, ini):
                self.status = status

def main():
    actionMap = {
        "start": 1,  #
        "inform": 0,
    }

    obj = model_sh50_bull()
    if actionMap["start"] == 1:
        obj.start()

    if actionMap["inform"] == 1:
        obj.inform()

if __name__ == '__main__':
    main()
