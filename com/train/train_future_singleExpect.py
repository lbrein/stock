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
from com.object.mon_entity import mon_trainOrder
import copy

# 回归方法
class train_future_singleExpect(object):
    """

    """
    iniAmount = 250000  # 单边50万
    csvList = [
        "SM_15_2.0_20_5_0.025_0.5_0.1_quick_single",
        "JM_15_2.0_20_1_0.25_1.0_0.1_quick_single",
        "JD_15_2.0_20_5_0.025_0.5_1.25_1_quick_single",
        "RB_15_2.0_20_5_0.2_0.5_1.25_1_quick_single",
        "AP_15_2.0_20"
    ]
    def __init__(self):
        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.isSimTickUse = False  # 是否使用1分钟模拟tick测试，否则直接使用kline回测
        self.topUse = False
        self.isEmptyUse = False  # 是否情况记录

        self.savePoint = False
        self.baseInfo = {}

        self.periodList = [15, 20]  # 窗体参数
        self.scaleList = [2.0]
        self.shiftScale = 1 # 滑点模拟系数
        self.deltaLine = 0.8
        self.processCount = 6
        self.scaleDiffList = [0.1]
        self.scaleDiff2List = [1.0]
        self.codeLists = ['JM', 'SM', 'V', 'I', 'AP', 'J']
        # k线时间
        #self.klineTypeList = ['5m']
        self.klineTypeList = ['15m', '30m']

        self.widthDeltaLineList = [3]
        self.widthDeltaLine = 0

        self.stopTimeLine = 5
        self.widthTimesPeriodList = [3]
        self.testDays = 90
        # 起始时间
        self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测

        self.endDate = public.getDate(diff=0)

        self.total_tablename = 'train_total_1'
        self.detail_tablename = 'train_future_1'
        self.totalMethod = 'single'
        self.method = 'simTick' if self.isSimTickUse else 'quick'
        self.stage = 'single15'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method + "_" + self.stage
        self.isAll = 0
        self.iterCondList = ['widthDeltaLine', 'scaleDiff2', 'scaleDiff']

    def iterCond(self):
        # 多重组合参数输出
        keys = self.iterCondList
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                self.__setattr__(keys[1], s1)

                for s2 in self.__getattribute__(keys[2] + 'List'):
                    self.__setattr__(keys[2], s2)

                    yield '%s_%s_%s' % (str(s0), str(s1), str(s2))

    def tops(self, num=10):
        Total = train_total()
        Total.tablename = "train_total"
        return [m[0:1] for m in Total.last_top(num=num)]

    def switch(self):
        # 生成all
        if self.isAll == 1:
            self.isEmptyUse = True
            if self.turnlineList:  self.turnlineList = [1.75, 2.5]
            if self.superlineList: self.superlineList = [3.25]
            self.klineTypeList = [ '15m', '30m']
            self.total_tablename = 'train_total_0'
            self.detail_tablename = 'train_future_0'
            self.testDays = 180
            # 起始时间
            self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测
            self.endDate = public.getDate(diff=1)

        self.empty()

    def empty(self):
        if self.isEmptyUse:
            Train = train_future()
            Total = train_total()
            Total.tablename = self.total_tablename
            Train.tablename = self.detail_tablename
            Train.empty()
            Total.empty()

        if self.savePoint:
            TrainOrder = mon_trainOrder()
            TrainOrder.drop(self.stage)
            print("empty mongodb trainOrder",self.stage)

    def Pool(self):
        time0 = time.time()

        pool = Pool(processes=self.processCount)
        shareDict = Manager().list([])

        Base = future_baseInfo()
        # 交易量大的，按价格排序, 类型:turple,第二位为夜盘收盘时间
        lists = Base.all(vol=280)
        tops = self.tops()
        # 清空数据库
        self.switch()

        for rs in lists:
            # 检查时间匹配
            codes = [rs[0]]
            if self.topUse and codes not in tops: continue
            print(rs)
            if codes[0] not in self.codes:  continue

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
        self.TrainOrder = mon_trainOrder()


        # 查询获得N分钟K线
        dfs_l = dfs = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=60)

        # 掉期时间
        self.adjustDates = self.Rice.getAdjustDate(codes[0], self.testDays)

        # 按时间截取并调整
        #dfs= self.dateAdjust(codes, dfs, sh)
        print('kline load:', mCodes, [len(dfs[m]) for m in mCodes])

        # 根据配置文件获取最佳交易手数对
        self.iniVolume = round(self.iniAmount / cs[0]["lastPrice"] / cs[0]["contract_multiplier"], 0)

        # 分参数执行
        results = []
        for period in self.periodList:
            for wdp in self.widthTimesPeriodList:
                self.widthTimesPeriod = wdp

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

    def stand(self, ser, period=20):
        return ser / ta.MA(ser.abs(), timeperiod=period * 5)

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    def point(self, row):
        r = float(self.pointLine)
        rsi, rsid = (row[key] for key in "rsi,rsid".split(","))

        if abs(rsi-50) > r or (abs(rsi-50) > (r - 5) and abs(rsid) > 1.25):
            return 1 if rsi > 50 else -1
        return 0

    preNode, batchId = None, {}
    cvsCodes = []
    def total(self, dfs, dfs2=None, period=60):
         pass

    def detect(self, df0, df1, period=15, uid=''):
        self.mon_records, self.preTick = [], None
        self.batchId = None
        docs = self.stageApply(df0, df1, period=period, uid=uid)
        res = pd.DataFrame(docs)
        if len(res) > 0:
            if self.saveDetail:
                self.Train.insertAll(docs)

            print( uid,len(self.mon_records))
            if self.savePoint and len(self.mon_records)>0:
                print("monsave", uid, len(self.mon_records))
                self.TrainOrder.col.insert_many(self.mon_records)

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
                sha = (res['income'].sum()/self.iniAmount - 0.02 * self.testDays/252) / inr.std()

            return {
                "count": int(len(docs) / 2),
                "amount": self.iniAmount,
                "price": res['rel_price'].mean(),
                "income": res["income"].sum(),
                "std": res['rel_std'].mean(),
                "maxdown": MaxDrawdown(sum),
                "sharprate": sha,
                "timediff": int(0 if np.isnan(diff) else diff)
            }
        else:
            return None

    # 核心策略部分
    def stageApply(self, df0, df1, period=15, uid=''):
        pass

    def adjustOrder(self, n0, date):
        preCode = ''
        print(self.adjustDates)
        for c in self.adjustDates:
            if str(c[1]) == date or (str(c[1])[:10] + ' 09:15:00')  == str(date):
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
        doc['isstop'] = 9
        doc['createdate'] = date
        doc['income'] = sign * (oldP - pp - 2 * sign * self.shift[0]) * doc["vol"] - doc["fee"]

        self.records.append(doc)

        doc1 = copy.deepcopy(self.preNode[0])
        doc1['createdate'] = date
        doc1['mode'] = int(9 * sign)
        doc1['price'] = newP
        doc1['isopen'] = 1
        doc1['batchid'] = self.batchId = uuid.uuid1()

        self.records.append(doc1)
        self.preNode = [doc1]

    def mon_saveTick(self, n0, doc):
        tick = copy.deepcopy(n0.to_dict())
        tick.update(doc)

        for key in ['kdjm', 'sarm', 'powm', 'isout', 'mode']:
            if key in tick: tick[key] = int(tick[key])

        if doc['isopen'] == 0:
            self.mon_records.append(tick)
            if self.preTick is not None:
                self.preTick['income'] = doc['income']
                self.preTick['enddate'] = doc['createdate']
                self.preTick['isstop'] = doc['isstop']
                self.mon_records.append(copy.deepcopy(self.preTick))
            self.preTick = None
        else:
            self.preTick = tick


    def order(self, n0, n1, mode, uid,  df0, isstop=0):
        # baseInfo 配置文件，查询ratio 和 每手吨数
        b0 = self.baseInfo[self.mCodes[0]]
        if mode != 0:
            self.batchId = uuid.uuid1()
        # 交易量
        v0 = self.iniVolume * b0["contract_multiplier"]
        # 费率
        fee0 = (self.iniVolume * b0["ratio"]) if b0["ratio"] > 0.5 else ((b0["ratio"]) * n0["close"] * v0)
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
            "rel_std": n0["std"],
            "atr": n0['atr'] if 'atr' in n0 else 0,
            "batchid": self.batchId,
            'p_l': n0["p_l"],
            'p_h': n0["p_h"],
            #'mastd': n0['pow'],
            #'macd2d':n0['macd2d'],
            "method":self.stage,
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

        self.mon_saveTick(n0, doc)
        self.records.append(doc)
        return True

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
        obj = train_future_singleExpect()
        obj.Pool()


if __name__ == '__main__':
    main()
