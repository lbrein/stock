# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ---  与tick 配套对比计算的 kline
    > 1m的 K线 模拟tick预测
    > 定期执行，用于筛选交易对
"""

from com.base.public import public, logger
from com.object.obj_entity import future_baseInfo
import talib as ta
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, Manager
from com.train.train_future_singleExpect import train_future_singleExpect
from com.base.stat_fun import fisher


# 回归方法
class train_future_macd(train_future_singleExpect):
    """

    """
    iniAmount = 250000  # 单边50万
    csvList = [
        "MA_15_2.0_15_3_0.5_1.2",
        "JM_30_2.0_15_3_0.5_1.2",
        "JD_30_2.0_15_3_0.5_1.2",
        "RB_30_2.0_15_3_0.5_1.2",
        "MA_30_2.0_30_3_0.5_1.2",
        "JM_30_2.0_30_3_0.5_1.2",
        "JD_30_2.0_30_3_0.5_1.2",
        "RB_30_2.0_30_3_0.5_1.2",
    ]

    def __init__(self):
        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.isSimTickUse = False  # 是否使用1分钟模拟tick测试，否则直接使用kline回测
        self.topUse = False
        self.isEmptyUse = False
        self.baseInfo = {}

        self.periodList = [15, 30, 45, 60]  # 窗体参数
        self.scaleList = [2.0]

        self.shiftScale = 0.527  # 滑点模拟系数
        self.processCount = 6
        # k线时间
        # self.klineTypeList = ['5m']
        self.klineTypeList = ['15m', '30m', '60m']

        self.widthDeltaLineList = [0.25, 0.5, 0.75]

        self.pointLineList = [1.2, 1.5, 2.0]
        self.pointLine = 2
        self.pointStatusLine = 3

        self.scaleDiff = 0
        self.scaleDiff2 = 1
        self.stopTimeLine = 5

        self.widthTimesPeriodList = [3]

        # 起始时间
        self.startDate = public.getDate(diff=-60)  # 60天数据回测
        self.endDate = public.getDate(diff=0)

        self.total_tablename = 'train_total_1'
        self.detail_tablename = 'train_future_1'
        self.totalMethod = 'macdmix'

        self.method = 'simTick' if self.isSimTickUse else 'quick'
        self.stage = 'dema4'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method + "_" + self.stage

    def iterCond(self):
        # 多重组合参数输出
        keys = ['widthDeltaLine', 'pointLine']
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                self.__setattr__(keys[1], s1)

                yield '%s_%s' % (str(s0), str(s1))

    def Pool(self):
        time0 = time.time()

        pool = Pool(processes=self.processCount)
        shareDict = Manager().list([])

        Base = future_baseInfo()
        # 交易量大的，按价格排序, 类型:turple,第二位为夜盘收盘时间
        lists = Base.all(vol=100)
        tops = self.tops()
        # 清空数据库
        self.empty()

        for rs in lists:
            # 检查时间匹配
            codes = [rs[0]]
            if self.topUse and codes not in tops: continue
            print(rs)
            if codes[0] not in ['JM', 'RB', 'JD', 'MA']:  continue

            for kt in self.klineTypeList:
                #self.start(codes, time0, kt, shareDict)
                try:
                    pool.apply_async(self.start, (codes, time0, kt, shareDict))
                    pass
                except Exception as e:
                    print(e)
                    continue
        pool.close()
        pool.join()

    def point(self, row):
        r= float(self.pointLine)

        width, wd1, wd2, macd, trend, volm, macd2d = (row[key] for key in
                                                "width,widthDelta,widthDelta2,macd,trend,volm,macd2d".split(","))
        mm, isPoint = 0, 0
        if width!=0 and not np.isnan(macd2d) :
            mm = pow(trend * abs(wd1) * abs(wd2) * abs(macd2d) * volm / width, 1/6) / 1.5
            cond0 = (macd2d * macd) > 0 and wd1 > 0 and wd2 > 0
            cond = mm > r
            isPoint = 0 if not (cond0 and cond ) else  np.sign(macd2d) * int(mm)

        columns = ['pow', 'isPoint']
        return pd.Series([mm, isPoint], index=columns)

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    def turn2(self, row):
        mm1 ,mm5, md, md5 = (row[key] for key in "macd2dm1,macd2dm5,macd2d,macd2d5".split(","))

        if mm1 < 0 and abs(mm1) > 0.2 :
           return 1 if md > 0  else -1

        elif mm5 < 0:
           return 1 if md5 > 0 else -1

        return 0

    def trend(self, row):
        close, high, open, low = (row[key] for key in "close,high,open,low".split(","))

        if open > close:
            r = 1 if (open -low) == 0 else ((open - close) / (open -low) - 0.1) * 2
        else:
            r = 1 if (high - open)== 0 else ((close - open) / (high - open) - 0.1) * 2

        return r if r > 0.2 else 0.2

    def total(self, dfs, dfs2=None, period=60):
        # 计算参数
        df0 = dfs[self.mCodes[0]]
        df0["rel_price"] = close = df0["close"]
        df0["datetime"] = df0.index

        s0 = self.shift[0]
        p_l = df0["p_l"] = (df0["close"] + s0)
        p_h = df0["p_h"] = (df0["close"] - s0)

        if self.isSimTickUse:
            # 调用复合apply函数计算混合参数
            close2 = dfs2[self.mCodes[0]]["close"]
            df0_1 = df0.apply(lambda row: self.k_ma(row['datetime'], row['rel_price'], close2, period), axis=1)
            df0 = pd.concat([df0, df0_1], axis=1)

        else:
            df0["ma"] = ma = ta.MA(close, timeperiod=period)
            df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)
            df0['open'] = close.shift(1)

            # 计算影线
            df0['trend'] = trend = df0.apply(lambda x: self.trend(x), axis=1)
            # 成交量
            vol = df0['volume']
            df0['volm'] = volm =  vol / vol.mean()

            # 上下柜
            # bullWidth
            df0["width"] = width = (4 * std / ma * 100).fillna(0)
            df0["bullwidth"] = width / width.abs().mean()

            # 近三分钟width变动
            df0["widthDelta"] = wd1 = ta.DEMA((width - width.shift(1)) * trend * volm, timeperiod=2)
            df0["widthDelta"] = wd1 /wd1.abs().mean()

            df0["widthDelta2"] = wd2 = wd1 - wd1.shift(1)
            df0["widthDelta2"] = wd2 / wd2.abs().mean()

            df0["wd2m"] =  wd1 * wd1.shift(1)
            df0["wd2m"] = df0.apply(lambda row: self.turn(row['wd2m'], row['widthDelta'], 1), axis=1)

            # macd区间
            dif, dea, macd = ta.MACD(close, fastperiod=int(int(period) * 0.8), slowperiod=period,
                                     signalperiod=int(0.4 * period))

            df0["macd"] = macd / macd.abs().mean() #归一化处理
            df0["mastd"] = ta.STDDEV(macd * trend * volm, timeperiod=9, nbdev=1)
            df0["macdmax"] = ta.MAX(df0["macd"].abs(), timeperiod = int(0.4 * period)) #归一化处理

            df0["macdm"] = macd * macd.shift(1)
            df0["macdm"] = df0.apply(lambda row: self.turn(row['macdm'], row['macd'], 1), axis=1)
            # 计算顶点
            for wt  in [5, 1]:
                ma2d = macd - macd.shift(1)
                df0["macd2d" + str(wt)] = ma2d / ma2d.abs().mean() #归一化处理
                m2m = ma2d * ma2d.shift(1)
                df0["macd2dm" + str(wt)] = m2m / m2m.abs().mean()

            df0["macd2d"]= df0["macd2d5"]
            df0["macd2dm"] = df0.apply(lambda row: self.turn2(row), axis=1)

        # 相对波动
        # 相对波动
        df0['delta'] = (p_l - p_h) / df0['std']

        df1 = None
        # 循环 scale
        docs = []
        for scale in self.scaleList:
            for conds in self.iterCond():
                uid = self.uidKey % (
                    '_'.join(self.codes), str(period), str(scale), self.klineType[:-1],
                    str(self.widthTimesPeriod) + '_' + conds)

                df0["top"], df0["lower"] = df0['ma'] + (scale - self.scaleDiff) * df0['std'], df0['ma'] - (
                        scale + self.scaleDiff) * df0['std']

                #print(df0.columns)
                df01 = df0.apply(lambda row: self.point(row), axis=1)
                df0['pow'] = df01['pow']
                df0['isPoint'] = df01['isPoint']

                df0.fillna(0, inplace=True)

                key = '_'.join(uid.split('_')[0:7])
                if key in self.csvList:
                    cs = []
                    bans = 'ma,open,close,high,low,p_l,p_h,top,lower,std,delta,volume,rel_price'.split(',')
                    for c in df0.columns:
                        if c not in bans:
                            cs.append(c)

                    file = self.Rice.basePath + '%s_pre.csv' % (uid)
                    print(uid, '---------------------------- to_cvs', file)
                    df0.to_csv(file, index=0, columns=cs)
                    # self.share.append(self.codes)

                # df0.fillna(0, inplace=True)
                tot = None
                tot = self.detect(df0, df1, period=period, uid=uid)
                if tot is not None and tot['amount'] != 0:
                    tot.update(
                        {
                            "scale": scale,
                            "method": self.totalMethod,
                            "code": self.codes[0],
                            "period": period,
                            "uid": uid,
                            "shift": (p_l - p_h).mean(),
                            "std": df0['width'].mean(),
                            "createdate": public.getDatetime()
                        }
                    )
                    docs.append(tot)
        return docs

    # 核心策略部分
    def stageApply(self, df0, df1, period=15, uid=''):

        doc, docs = {}, []

        """
            布林带策略：
        
            macd 组合策略 
            macd - 快慢线差值
            mastd - 差值标准差
            macdm - macd交叉点 
            macd2d - macd 变化率 
            macd2dm - macd 顶点  macd2d>0 谷点 macd2d<0 顶点
            
            # 开平仓状态 
            开仓：
              1 - 标准布林带开仓（> std)
              2 : 扩展布林带策略开仓 (局部布林带顶点开仓) 5 局部macd谷点开仓
              3： macd策略顶点开仓
              4： macd拐点> 3类开仓
              6： macd 交叉点开仓 
            平仓：
              0：标准布林带平仓
              1、2： 扩展布林带平仓（布林带顶点/macd顶点）    
              3、macd 策略 谷底平仓
              4、macd 突发点强制平仓（结束布林状态）
              5、macd 交叉点平仓（结束macd状态）
              6、macd 交叉点顶点平仓（macd顶点） 
            
        """

        status, isOpen = 0, 0

        for i in range(period, len(df0)):
            isRun, isstop = False, 0

            ma, p_l, p_h, top, lower, std, volm, delta, width, wd1, wd2, wd2m, macd,  macdm, macd2d, macd2dm, isPoint, pow = (
                df0.ix[i, key] for key in
                "ma,p_l,p_h,top,lower,std,volm,delta,bullwidth,widthDelta,widthDelta2,wd2m,macd,macdm,macd2d,macd2dm,isPoint,pow".split(
                    ","))

            if isPoint != 0 and isPoint * status <= 0:
                """
                   突变点处理 
                   将布林带策略切换为macd策略 
                """
                status = isPoint
                # 强制平仓 - 4类和5类
                if isOpen != 0 and isOpen * status < 0:
                    doc = self.order(df0.iloc[i], None, 0, uid, df0, isstop= 4)
                    if doc is not None:
                        isOpen = 0
                        docs.append(doc)

                # 反向持仓
                if isOpen == 0 and pow >= self.pointStatusLine:
                    isOpen = 4 if status > 0 else - 4
                    isRun = True

            elif macdm != 0 and status != 0:
                """
                    交叉点: 快慢线穿越处理
                    1、结束突变状态，变为布林带处理
                    2、根据节点状况开平新仓，状态为6 
                """
                # 结束macd状态
                status = 0

            elif status!=0:
                """
                    macd 策略处理 
                
                """
                if isOpen == 0 and status * macd2dm > 0 :
                    # 开仓
                    isOpen, isRun = 3 if status > 0 else -3, True

                    # 平仓
                elif isOpen != 0 and status * macd2dm < 0:
                    isOpen, isRun, isstop = 0, True, 3

            else:
                """
                    布林带策略处理                 
                """
                wline = self.widthDeltaLine

                cond1, cond2 = False, False
                if wline > 0:
                    # 布林宽带变化率
                    cond1 = (wd1 < 0 and abs(macd2d) < 1) or ((volm + abs(wd1) + abs(macd2d))/3 < wline)

                if isOpen == 0:
                    # 突变状态开始
                    # 大于上线轨迹
                    if p_h >= top and cond1:
                        isOpen = -1
                        isRun = True

                    elif p_l <= lower and cond1:
                        isOpen = 1
                        isRun = True

                    elif (p_h + self.scaleDiff2 * std / 2) >= top and not cond1 and (wd2m < 0 or macd2dm < 0):
                        isOpen = -2 if wd2m < 0 else -5
                        isRun = True

                    elif (p_l - self.scaleDiff2 * std / 2) <= lower and not cond1 and (wd2m < 0 or macd2dm > 0) :
                        isOpen = 2 if wd2m > 0 else 5
                        isRun = True

                # 平仓
                else:
                    sign, dline = isOpen / abs(isOpen), - self.scaleDiff2 * std / 2
                    cond3 = (sign * ((p_h if isOpen > 0 else p_l) - ma))
                    #
                    if cond3 >= -dline and not cond1 and (wd2m  < 0 or macd2dm * isOpen < 0) :
                        isOpen, isstop = 0, 2 if wd2m < 0 else 5
                        isRun = True

                    elif cond3 >= 0 and cond1:
                        isOpen, isstop = 0, 0
                        isRun = True

            # print(i, isOpen, status, isstop, isRun )
            if isRun:
                doc = self.order(df0.iloc[i], None, isOpen, uid, df0, isstop=isstop)
                if doc is not None:
                    docs.append(doc)
        return docs


def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = train_future_macd()
        obj.Pool()

if __name__ == '__main__':
    main()
