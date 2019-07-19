# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ---  与tick 配套对比计算的 kline
    > 1m的 K线 模拟tick预测
    > 定期执行，用于筛选交易对
"""

from com.base.stat_fun import MaxDrawdown
from com.base.public import public, logger
import pandas as pd
import talib as ta
import numpy as np
from com.object.obj_entity import train_future, train_total, future_baseInfo
from com.data.interface_Rice import interface_Rice, tick_csv_Rice
import time
import uuid
from multiprocessing import Pool, Manager
import copy

# 回归方法
class train_future_zhao(object):
    """

    """

    csvList = [
        "JM_40_2.0_1_0_zhao",
        "RB_40_2.0_1_0_zhao",
        "TA_40_2.0_1_0_zhao",
        "CU_40_2.0_1_0_zhao",
    ]

    def __init__(self):
        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.isSimTickUse = False
        self.topUse = False
        self.isEmptyUse = False   # 是否情况记录
        self.iniAmount = 20000000  # 单边50万
        self.stopLine = 0.0025
        self.baseInfo = {}
        self.codeLists = []
        self.shiftScale = 0.527  # 滑点模拟系数
        self.processCount = 4

        # k线时间
        self.klineTypeList = ['1d']
        self.scaleList = [2.0]
        self.periodList = [40]  # 窗体参数

        self.tangStartPeriod0 = 18
        self.tangStartPeriod1 = 27
        self.tangEndPeriod = 12
        self.tangDropPeriod = 40

        self.testDays = 500
        # 起始时间
        self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测

        self.endDate = public.getDate(diff=0)

        self.total_tablename = 'train_total_3'
        self.detail_tablename = 'train_future_3'

        self.totalMethod = 'single'
        self.stage = 'zhao'
        self.method = 'ma'

        self.uidKey = "%s_%s_%s_%s_%s_" + self.stage
        self.isAll = 1
        self.iterCondList = []

    def iterCond(self):
        # 多重组合参数输出
        keys = self.iterCondList
        yield 0
        """
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                self.__setattr__(keys[1], s1)

                for s2 in self.__getattribute__(keys[2] + 'List'):
                    self.__setattr__(keys[2], s2)

                    yield '%s_%s_%s' % (str(s0), str(s1), str(s2))
        """

    def tops(self, num=10):
        Total = train_total()
        Total.tablename = "train_total"
        return [m[0:1] for m in Total.last_top(num=num)]

    def switch(self):
        # 生成all
        if self.isAll == 1:
            self.total_tablename = 'train_total_3'
            self.detail_tablename = 'train_future_3'
        self.empty()

    def empty(self):
        print(self.isEmptyUse)

        if self.isEmptyUse:
            Train = train_future()
            Total = train_total()
            Total.tablename = self.total_tablename
            Train.tablename = self.detail_tablename
            Train.empty()
            Total.empty()

    def Pool(self):
        time0 = time.time()

        pool = Pool(processes=self.processCount)
        shareDict = Manager().list([])

        Base = future_baseInfo()
        # 交易量大的，按价格排序, 类型:turple,第二位为夜盘收盘时间
        lists = Base.all(vol=100)
        tops = self.tops()

        # 清空数据库
        self.switch()

        for rs in lists:
            # 检查时间匹配
            codes = [rs[0]]
            if self.topUse and codes not in tops: continue
            if codes[0] not in ['JM', 'CU']: continue
            print(rs)

            if self.isAll == 0 and codes[0] not in self.codeLists:  continue

            for kt in self.klineTypeList:
                self.start(codes, time0, kt, shareDict)
                try:
                    #pool.apply_async(self.start, (codes, time0, kt, shareDict))
                    pass
                except Exception as e:
                    print(e)
                    continue
        pool.close()
        pool.join()

    cindex = 0

    def start(self, codes, time0, kt, shareDict):
        print("子进程启动:", self.cindex, codes, time.time() - time0)

        self.klineType = kt
        # 主力合约
        self.codes = codes
        self.mCodes = mCodes = [n + '88' for n in codes]

        # 查询获得配置 - 费率和每手单量
        self.Base = future_baseInfo()
        for doc in self.Base.getInfo(codes):
            self.baseInfo[doc["code"] + '88'] = doc

        cs = [self.baseInfo[m] for m in self.mCodes]

        # 计算tick 导致的滑点
        sh = [self.baseInfo[d + '88']['tick_size'] for d in codes]
        self.shift = [sh[i] * self.shiftScale for i in range(len(sh))]

        # 子进程共享类
        self.Rice = tick_csv_Rice()
        self.Rice.setTimeArea(cs[0]["nightEnd"])
        self.Train = train_future()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename
        self.Train.tablename = self.detail_tablename

        # 掉期时间
        self.adjustDates = self.Rice.getAdjustDate(codes[0], self.testDays)

        # 查询获得N分钟K线
        dfs_l = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=60)

        # 获得1分钟K线作为滚动线
        if self.isSimTickUse:
            dfs = self.Rice.kline(mCodes, period='15m', start=self.startDate, end=self.endDate, pre=0)
        else:
            dfs = dfs_l

        # 按时间截取并调整
        print('kline load:', mCodes, [len(dfs[m]) for m in mCodes])

        # 根据配置文件获取最佳交易手数对
        #self.iniVolume = round(self.iniAmount / cs[0]["lastPrice"] / cs[0]["contract_multiplier"], 0)

        # 分参数执行
        results = []

        for period in self.periodList:
                docs = self.total(dfs, dfs_l, period=period)
                if docs is None or len(docs) == 0:  continue
                logger.info((self.codes, period, self.klineType, len(docs), " time:", time.time() - time0))
                self.Total.insertAll(docs)

    #  混合K线-计算平均值和标准差
    def k_ma(self, d, p, close, period):
        # 截
        df = close[close.index < d][-period - 5:]
        df = df.append(pd.Series([p], index=[d]))
        # 平均值
        ma = ta.MA(df, timeperiod=period)
        std = ta.STDDEV(df, timeperiod=period, nbdev=1)
        # bullWidth
        width = (4 * std / ma * 100).fillna(0)

        # 近width变动
        wd1 = ta.MA(width - width.shift(1), timeperiod=self.widthTimesPeriod).fillna(0)
        wd2 = wd1 - wd1.shift(1)
        wd2m = wd2 * wd2 * wd2.shift(1)
        columns = ['ma', 'std', 'bullwidth', 'widthDelta', 'widthDelta2', 'wd2m']
        return pd.Series([x.values[-1] for x in [ma, std, width, wd1, wd2, wd2m]], index=columns)

    def stand(self, ser):
        return ser / ser.abs().mean()

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    def point(self, row):
        close,ma10,ma20,ma55,atrb,tu_s,td_s,tu_e,td_e = (row[key] for key in "close,ma10,ma20,ma55,atrb,tu_s,td_s,tu_e,td_e".split(","))
        return 0

    def isout(self, row):
        close, ma20, ma55 = (row[key] for key in "close,ma20,ma55".split(","))
        return 1 if close > ma55 and ma20 < ma55 else -1 if close < ma55 and ma20 > ma55 else 0

    def k_fix(self, row, mode):
        close, open, high, low = (row[key] for key in ['close', 'open', 'high', 'low'])

        if mode == 1:
           if close > open and (close-open)/open > 0.002:
               trend = 1 if high == open else (close - open) / (high - open)
               return close if trend < 0.2 else high
           else:
               return high

        elif mode == -1:
            if close < open and (open - close)/open > 0.002:
                trend = 1 if low == open else (open-close) / (open - low)
                return close if trend < 0.2 else low
            else:
                return low

    def isMam(self, row):
        return 0 if row['mad'] >= 0 else 1 if row['ma10'] > row['ma20'] else -1

    preNode, batchId = None, {}
    cvsCodes = []
    def total(self, dfs, dfs2=None, period=60):
        # 计算参数
        df0 = dfs[self.mCodes[0]]
        df0["rel_price"] = close = df0["close"]
        df0["datetime"] = df0.index

        s0 = self.shift[0]
        p_l = df0["p_l"] = (df0["close"] + s0)
        p_h = df0["p_h"] = (df0["close"] - s0)

        # MA指标
        df0["ma10"] = ma10 = ta.MA(close, timeperiod=10)
        df0["ma20"] = ma20 = ta.MA(close, timeperiod=20)

        df0["ma55"] = ma55 = ta.MA(close, timeperiod=55)

        df0['atrb'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=21)

        # 调整异常high、low指标
        df0['high1'] = df0['high']
        df0['low1'] = df0['low']
        df0['high'] = df0.apply(lambda row: self.k_fix(row,1), axis=1)
        df0['low'] = df0.apply(lambda row: self.k_fix(row, -1), axis=1)

        mac = ma10 - ma20

        # isPoint
        df0['mad'] = mac * mac.shift(1)
        df0['mam'] = mam =  df0.apply(lambda row: self.isMam(row), axis=1)
        minidx, maxidx = ta.MINMAXINDEX(mam, timeperiod = 80)
        df0['interval'] = abs(minidx - maxidx)

        # 唐奇安线
        df0['tu_s'] = ta.MAX(df0['high'].shift(1), timeperiod=self.tangStartPeriod0)
        df0['td_s'] = ta.MIN(df0['low'].shift(1), timeperiod=self.tangStartPeriod0)

        df0['tu_s1'] = ta.MAX(df0['high'].shift(1), timeperiod=self.tangStartPeriod1)
        df0['td_s1'] = ta.MIN(df0['low'].shift(1), timeperiod=self.tangStartPeriod1)

        df0['tu_d'] = ta.MIN(df0['high'].shift(1), timeperiod=self.tangDropPeriod)
        df0['td_d'] = ta.MIN(df0['low'].shift(1), timeperiod=self.tangDropPeriod)

        df0['tu_e'] = ta.MAX(df0['high'].shift(1), timeperiod=self.tangEndPeriod)
        df0['td_e'] = ta.MIN(df0['low'].shift(1), timeperiod=self.tangEndPeriod)

        df0['isout'] = isout = df0.apply(lambda row: self.isout(row), axis=1)

        df0['isout3'] = ta.SUM(isout, timeperiod=3)
        df0['isout5'] = ta.SUM(isout, timeperiod=5)

        df1 = None
        # 循环 scale
        docs = []
        for scale in self.scaleList:
            for conds in self.iterCond():
                uid = self.uidKey % (
                    '_'.join(self.codes), str(period), str(scale), self.klineType[:-1], conds)

                isCvs = False
                for key in self.csvList:
                    if uid.find(key) > -1:
                        isCvs = True
                        break

                if isCvs :
                    cs = []
                    bans = 'ma,p_l,p_h,top,lower,std,delta,volume,sard,rel_price,width,volm,'.split(
                        ',')
                    for c in df0.columns:
                        if c not in bans:
                            cs.append(c)

                    file = self.Rice.basePath + '%s_pre.csv' % (uid)
                    print(uid, '---------------------------- to_cvs', file)
                    df0.to_csv(file, index=0, columns=cs)

                tot = self.detect(df0, df1, period=period, uid=uid)

                if tot is not None and tot['amount'] != 0:
                    tot.update(
                        {
                            "scale": scale,
                            "code": self.codes[0],
                            "period": period,
                            "uid": uid,
                            "shift": (p_l - p_h).mean(),
                            "createdate": public.getDatetime()
                        }
                    )
                    docs.append(tot)
        return docs

    def detect(self, df0, df1, period=15, uid=''):
        docs = []
        for i in range(period, len(df0)):
            doc = self.stageApply(df0, df1, period=period, uid=uid)



        res = pd.DataFrame(docs)
        #print(res)
        if len(res) > 0:
            if self.saveDetail:
                print('detail save:', uid, len(res))
                self.Train.insertAll(docs)
            try:
                diff = res[res['diff'] > 0]['diff'].mean()
            except:
                diff= 0

            # 计算最大回测
            sum = res['income'].cumsum() + self.iniAmount

            inr = res['income'] / self.iniAmount
            # 计算夏普指数
            sha = 0
            if inr.std() != 0:
                sha = (res['income'].sum()/self.iniAmount - 0.02 * self.testDays/365) / inr.std()

            return {
                "count": int(len(docs) / 2),
                "amount": self.iniAmount,
                "price": res['rel_price'].mean(),
                "income": res["income"].sum(),
                "maxdown": MaxDrawdown(sum),
                "sharprate": sha,
                "timediff": int(0 if np.isnan(diff) else diff)
            }
        else:
            return None

        # 核心策略部分
    def getMax(self, df0, s, e, mode):
        if mode > 0:
            return df0[(df0.index>=s) & (df0.index<e)].ix[:-1, 'close'].max()
        else:
            return df0[(df0.index >= s) & (df0.index < e)].ix[:-1,'close'].min()

    def stageApply(self, df0, df1, period=40, uid=''):
        self.records = []
        """
            布林带策略：            
        """
        status, isOpen = 0, 0
        b0 = self.baseInfo[self.mCodes[0]]
        # 修正品种
        bulltype = b0['bulltype']

        adjusts = [c[1] for c in self.adjustDates]

        for i in range(period, len(df0)):
            isRun, isstop = False, 0

            close, date, ma10, ma20, ma55, atrb, tu_s, td_s, tu_s1, td_s1, tu_d, td_d, interval = (df0.ix[i, key] for key in
                                                                     "close,datetime,ma10,ma20,ma55,atrb,tu_s,td_s,tu_s1,td_s1,tu_d,td_d,interval".split(
                                                                         ","))

            if isOpen == 0:
                """
                    开盘策略 
                    1、20分钟线策略
                    2、55分钟策略
                """
                opt1 = (ma10 > ma20 and close > tu_s) or (ma10 < ma20 and close < td_s)
                opt2 = (close < td_s1) or (close > tu_s1)

                if opt1 and interval > 12:
                    isOpen, isRun = np.sign(ma10 - ma20), True

                elif opt2:
                    isOpen, isRun = -2 if close < td_s1 else 2, True

            elif isOpen!=0:

                """
                    交叉点: sarm结束点 
                    1、结束突变状态，变为布林带处理
                    2、根据节点状况开平新仓，状态为6 
                """
                # 调仓
                if date in adjusts:
                    self.adjustOrder(df0.iloc[i], date)

                # 平仓并反向开仓
                opt1 = (ma10 > ma20 and close > tu_s and isOpen<0) or (ma10 < ma20 and close < td_s and isOpen>0)
                opt2 = (close < td_s1 and isOpen > 0) or (close > tu_s1 and isOpen < 0)

                if (opt1 and interval > 12) or opt2 :
                    # 平仓并反向开仓
                    if self.order(df0.iloc[i], None, 0, uid, df0, isstop=2):
                        isOpen, isRun = np.sign(ma10 - ma20) * 3 , True

                elif self.preNode is not None:
                    # stop_price 止损
                    stop_price = self.preNode[0]['stop_price']
                    # 开仓止损线止损
                    if isOpen * (close - stop_price ) < 0:
                        isOpen, isRun, isstop = 0, True, 4

            if isRun:
                self.order(df0.iloc[i], None, isOpen, uid, df0, isstop=isstop)

        return self.records

    def adjustOrder(self, n0, date):
        preCode = ''
        for c in self.adjustDates:
            if c[1] == date:
                preCode = c[2]
                break

        s = str(date).split(" ")[0]
        df1 = self.Rice.kline([preCode], period='1d', start=public.getDate(diff=-2, start= s), end=s, pre=0)
        oldP = df1[preCode]['open'].values[-1]
        newP = n0['open']

        # 调仓卖出
        doc = copy.deepcopy(self.preNode[0])
        sign = np.sign(self.preNode[0]['mode'])
        pp = self.preNode[0]['price']

        doc['price'] = oldP
        doc['isopen'] = 0
        doc['mode'] = -doc['mode']
        doc['isstop'] = 6
        doc['createdate'] = date
        doc['income'] = sign * (oldP - pp - 2 * sign * self.shift[0]) * doc["vol"] - doc["fee"]

        self.records.append(doc)

        doc1 = copy.deepcopy(self.preNode[0])
        doc1['createdate'] = date
        doc1['mode'] = int(6 * sign)
        doc1['price'] = newP
        doc1['isopen'] = 1
        doc1['batchid'] = self.batchId = uuid.uuid1()

        self.records.append(doc1)
        self.preNode = [doc1]

    batchId = None
    def order(self, n0, n1, mode, uid,  df0, isstop=0):
        # baseInfo 配置文件，查询ratio 和 每手吨数
        b0 = self.baseInfo[self.mCodes[0]]
        if mode != 0:
            self.batchId = uuid.uuid1()

        dp = n0['td_d'] if mode > 0 else n0['tu_d']
        if np.isnan(dp) or dp == n0['close']:
            dp = n0['close'] * (1-0.0025)

        if self.preNode is not None:
            iniVolume = self.preNode[0]['hands']
            # 交易量
        else:
            iniVolume = int(self.iniAmount * self.stopLine / abs(n0['close'] - dp) / b0["contract_multiplier"])
            if iniVolume==0: iniVolume = 1

        v0 = iniVolume * b0["contract_multiplier"]
        # 费率
        fee0 = (iniVolume * b0["ratio"]) if b0["ratio"] > 0.5 else ((b0["ratio"]) * n0["close"] * v0)
        type = mode if not self.preNode else -self.preNode[0]["mode"]

        doc = {
            "createdate": n0["datetime"],
            "code": self.codes[0],
            "price": n0["close"],
            "vol": self.preNode[0]["vol"] if self.preNode else v0,
            "mode": int(type),
            "isopen": 0 if mode == 0 else 1,
            "fee": fee0,
            "income": 0,
            "isstop": isstop,
            "rel_price": n0["rel_price"],
            "atr": n0['atr'] if 'atr' in n0 else 0,
            "batchid": self.batchId,
            'p_l': n0["p_l"],
            'p_h': n0["p_h"],
            'hands': iniVolume,
            'stop_price': 0 if mode==0 else n0['td_d'] if mode>0 else n0['tu_d'],
            'method': self.method,
            "diff": 0 ,
            "uid": uid
        }

        if mode == 0 :
            if self.preNode is not None:
                p0 = self.preNode[0]
                doc['income'], doc['highIncome'], doc['lowIncome'] = self.calcIncome(n0, p0, df0)
                doc["diff"] = int(public.timeDiff(str(n0['datetime']), str(p0['createdate'])) / 60)
                self.preNode = None
        else:
            doc["income"] = -doc["fee"]
            self.preNode = [doc]

        self.records.append(doc)
        return doc

    def calcIncome(self, n0, p0, df0):
        # 计算收益，最大/最小收益
        high = df0[(p0['createdate'] <= df0.index) & (df0.index <= n0['datetime'])]['high'].max()
        low = df0[(p0['createdate'] <= df0.index) & (df0.index <= n0['datetime'])]['low'].min()
        close = n0["close"]
        sign = p0["mode"] / abs(p0["mode"])

        # 收入
        income = sign * (close - p0["price"] - 2 * sign * self.shift[0]) * p0["vol"] - p0["fee"]
        # 最大收入
        highIncome = sign * ((high if sign > 0 else low) - p0["price"] - 2 * sign * self.shift[0]) * p0["vol"] - p0[
            "fee"]
        # 最大损失
        lowIncome = sign * ((high if sign < 0 else low) - p0["price"] - 2 * sign * self.shift[0]) * p0["vol"] - p0[
            "fee"]

        return income, highIncome, lowIncome

def main():
    action = {
        "kline": 1,
    }
    if action["kline"] == 1:
        obj = train_future_zhao()
        obj.Pool()

if __name__ == '__main__':
    main()
