# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

      期货多币对 多进程对冲交易程序
      针对 1分钟k线的测试
      多个币对按收盘时间统一为多个进程

"""

from multiprocessing import Pool, Manager
import  time

from com.model.model_future_detect_single_ctp_v10 import  model_future_detect_single
import talib as ta
import pandas as pd

# 期货混合交易模型
class model_future_single_0(model_future_detect_single):

    def __init__(self):
        # 统一参数
        super().__init__()

        self.topUse = True  # 启用定时统计列表
        self.isCTPUse = True  # 是否启用CTP接口（并区分
        self.iniAmount = 250000  # 单边50万
        self.isDeadCheck = False  # 是否启用自动检查并交易沉淀品种
        self.isAutoAlterPosition = True #是否启用自动调仓
        self.banCodeList = ['I', 'MA', 'NI']  # 暂时无权限操作的code
        self.isWorking = True
        self.topTableName = 'train_total_s1'
        self.topFilter = """ (count>10) """
        self.ctpuser = 'fz'

        self.topNumbers = 5  # 最大新币对 (监控币对还包括历史10和未平仓对)
        self.minRate = 0.08  # 最低收益率

    def pool(self):
        pool = Pool(processes=5)
        shareDict = Manager().list([])
        CTP = None

        # 初始化codes列表
        if self.topUse:
            self.filterCodes()

        for codes in self.seperate():
            #self.start(codes, shareDict, CTP)
            try:
                pool.apply_async(self.start, (codes, shareDict, CTP))
                time.sleep(3)
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

        # 计算布林参数
        def paramCalc(self, dfs, cur):
            # 分钟线
            if len(dfs) == 0: return None
            c0 = cur[self.mCodes[0]]

            # 周期和标准差倍数
            period, scale, sd, wperiod = self.procMap.period, self.procMap.scale, self.procMap.scaleDiff, self.procMap.widthTimesPeriod

            size = period + 15
            # print(dfs)
            # 去掉当前的即时k线
            df0 = dfs[self.mCodes[0]][-size:-1]

            # 计算相关参数
            close = df0["close"]
            # 添加最新
            close = close.append(pd.Series(c0["last"]))

            ma = ta.MA(close, timeperiod=period)
            std = ta.STDDEV(close, timeperiod=period, nbdev=1)
            top, lower = ma + (scale - sd) * std, ma - (scale + sd) * std
            #
            width = (top - lower) / ma * 100
            widthDelta = ta.MA(width - width.shift(1), timeperiod=wperiod)

            # 即时K线取2点中值
            # lw = width[-10:]
            # last = lw - (lw.shift(1) + lw.shift(2)) / 2
            # widthDelta = widthDelta[:-1].append(last[-1:])
            wd2 = widthDelta - widthDelta.shift(1)
            wd2m = wd2 * wd2.shift(1)

            # macd 参数计算
            dif, dea, macd = ta.MACD(close, fastperiod=int(period / 3), slowperiod=period, signalperiod=int(period / 4))
            macd2 = macd * macd.shift(1)
            macd2d = macd - macd.shift(1)
            macd2dm = (macd - macd.shift(1)) * (macd.shift(1) - macd.shift(2))

            #
            isBullOut = 0
            for i in range(-1, -5, -1):
                if top[i] < close[i] or lower[i] > close[i]:
                    isBullOut = -1 if top[i] < close[i] else 1
                    break

            return {
                "ma": ma.values[-1],
                "top": top.values[-1],
                "close": c0["last"],
                "lower": lower.values[-1],
                "width": width.values[-1],
                "wmean": width.mean(),
                "std": std.values[-1],
                "widthdelta": widthDelta.values[-1],  # 布林带变化
                "wd2": wd2.values[-1],  # 布林带二阶变化率
                "wd2m": wd2m.values[-1],
                "p_l": c0["asks"][0],  # 买入铜，卖出铅价格
                "p_h": c0["bids"][0],  # 买入铅，卖出铜价格
                "delta": (c0["asks"][0] - c0["bids"][0]) / std.values[-1],
                "macd": macd.values[-1],
                "macd2": macd2.values[-1],
                "macd2d": macd2d.values[-1],
                "macd2dm": macd2dm.values[-1],
                "isout": isBullOut
            }

    pub = 0

    def orderCheck(self, cur, param):
        isOpen, isRun, isstop = self.procMap.isOpen, False, 0
        wline, sd2 = self.procMap.widthline, self.procMap.scaleDiff2
        ma, p_l, p_h, top, lower, std, delta, wd1, wd2, wd2m, macd, macd2, macd2d, macd2dm, isout = (param[key] for key in
                            "ma,p_l,p_h,top,lower,std,delta,widthdelta,wd2,wd2m,macd,macd2,macd2d,macd2dm,isout".split(
                                                                                                         ","))

        if param["delta"] > self.deltaLimit: return None

        cond1, cond2 = False, False
        if wline > 0:
            # 布林宽带变化率
            cond1 = (param['widthdelta'] < wline and param['wd2'] < (wline / 2))
            # 拐点
            cond2 = param['wd2m'] < 0 and param['wd2'] < 0

        # 开仓
        if isOpen == 0:
            # 已关闭的交易对只平仓， 不再开仓
            self.pub += 1
            if self.procMap.codes in self.noneUsed: return None

            # 大于上线轨迹
            # if self.pub%1000 ==1:
            # print(self.procMap.currentUid,isOpen, param['widthdelta'],param['wd2'], wline, cond1, cond2)

            if (param["p_h"] > param["top"]) and cond1:
                isOpen, isRun = -1, True

            # 低于下轨线
            elif (param["p_l"] < param["lower"]) and cond1:
                isOpen, isRun = 1, True


            elif ((param["p_h"]) > param["top"]) and not cond1 and cond2:
                isOpen = -2
                isRun = True


            elif ((param["p_l"]) < param["lower"]) and not cond1 and cond2:
                isOpen = 2
                isRun = True

        # 平仓
        else:
            stopMinutes = self.stopTimeLine * self.procMap.period * self.procMap.kline
            preNode = self.procMap.preNode
            sign, dline = isOpen / abs(isOpen), sd2 * std / 2

            cond3 = (sign * ((p_h if isOpen > 0 else p_l) - ma))
            #
            if cond3 >= -dline and not cond1 and cond2 :
                isOpen, isstop = 0, 2
                isRun = True

            elif cond3 >= 0 and cond1:
                isOpen = 0
                isRun = True

            # 止损
            elif stopMinutes > 0 and preNode is not None:
                tdiff = self.Rice.timeDiff(preNode[0]['createdate'], quick=stopMinutes)
                if tdiff > stopMinutes and (cond3 >= -dline * 2) and cond2:
                    isOpen, isstop = 0, 1
                    isRun = True

        #self.debugT((self.uid, isOpen,isRun, cond1, cond2 ) )
        if isRun:
            self.order(cur, isOpen, param, isstop=isstop)


def main():
    obj = model_future_single_0()
    obj.pool()


if __name__ == '__main__':
    main()
