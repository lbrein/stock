# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

      期货单品种交易

"""

from com.model.model_future_zhao_ctp_v1 import model_future_zhao_v1
import numpy as np


# 期货混合交易模型
class model_future_zhao55_v1(model_future_zhao_v1):

    def __init__(self):
        # 统一参数
        super().__init__()

        self.isTest = False
        self.isTickSave = True
        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.tickInterval = 100  # 间隔处理时间
        #self.indexCodeList = []

        self.methodName = 'mZhao55'  # 策略名称
        self.volumeCalcType = 1  # 交易量计算类别

    def orderCheck(self, cur, param):
        status, isOpen, isRun, isstop = self.procMap.status, self.procMap.isOpen, False, 0
        if status is None: status = 0

        close, date, ma10, ma20, ma55, atrb, tu_s, td_s, tu_d, td_d, tu_s1, td_s1, tu_34, td_34, isout3, isout5, interval = (
            param[key] for key in
            "close,datetime,ma10,ma20,ma55,atrb,tu_s,td_s,tu_d,td_d,tu_s1,td_s1,tu_34,td_34,isout3,isout5,interval".split(
                ","))

        if np.isnan(ma55): return None

        self.preNode = self.procMap.preNode
        code = self.procMap.codes[0]

        # index 只在55策略执行
        indexList = [c[0] for c in self.indexCodeList]

        if isOpen == 0:
            """
                开盘策略 
                1、20分钟线策略
                2、55分钟策略
            """
            if (code in indexList): return
            # dd = str(public.getTime(style='%H:%M:%S'))
            sign = np.sign(close - ma55)
            # opt1 = ((sign * isout3 > 2) or (sign * isout5 > 2)) and sign * isout5 < 5 and ('14:58:00' < dd < '15:00:00')
            opt1 = False
            opt2 = (close > ma55 and close > tu_34) or (close < ma55 and close < td_34)
            if (opt1 or opt2):
                isOpen, isRun = 5 * sign, True

            # 临时特殊补充过滤：
            if isRun:
                posMode = self.Record.openMode(self.relativeMethods, code)
                lastStop = self.Record.lastStop(self.relativeMethods, code)

                opt3 = (posMode[0] == self.relativeMethods[0] and posMode[1] == 1 and posMode[
                    2] * isOpen > 0)  # 同向持有系统1
                opt4 = (posMode[1] == 0 and lastStop[0] == 2)  # 系统1反向平仓，则可任意方向开仓
                opt5 = code in indexList  # 指数期货
                # opt5 = code in indexList and not ((isOpen > 0 and code in self.shortCodeList) or (isOpen < 0 and code in self.longCodeList)) # 指数期货方向平仓

                if not (opt3 or opt4 or opt5):
                    isOpen, isRun = 0, False

        elif isOpen != 0:
            """
                交叉点: sarm结束点 
                1、结束突变状态，变为布林带处理
                2、根据节点状况开平新仓，状态为6 
            """

            # 平仓
            if self.preNode is not None:
                if (code in indexList): return
                # 平仓并反向开仓
                opt1 = (ma10 > ma20 and close > tu_s and isOpen < 0) or (
                        ma10 < ma20 and close < td_s and isOpen > 0)

                opt2 = (close < td_s1 and isOpen > 0) or (close > tu_s1 and isOpen < 0)

                keepDays = len(self.df0[self.df0['datetime'] > self.preNode[0]['createdate']])

                if (keepDays > 13) and ((opt1 and interval > 12) or opt2):
                    # 系统1 反向平仓
                    isOpen, isRun, isstop = 0, True, 3

                else:
                    # 止损平仓
                    Pd, s = self.preNode[0]['price'], self.preNode[0]['createdate']
                    mp = self.getMax(self.df0, s, date, isOpen)
                    mp = close if np.isnan(mp) else mp
                    # print(code, np.sign(isOpen) , mp,3.0 * atrb, np.sign(isOpen) * (mp - close))
                    # 最高点回落 3倍 atr 平仓
                    if not np.isnan(mp) and (np.sign(isOpen) * (mp - close) > 3.0 * atrb):
                        isOpen, isRun, isstop = 0, True, 6

        if isRun:
            self.order(cur, isOpen, param, isstop=isstop)


def main():
    obj = model_future_zhao55_v1()
    obj.pool()

if __name__ == '__main__':
    main()
