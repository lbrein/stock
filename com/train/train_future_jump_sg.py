# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

   # 5分钟短线策略回测
    策略1：异常追涨杀跌，以5分钟k线为基准计算
    策略2：跳空预测


"""

from com.base.public import public, logger
import pandas as pd
import talib as ta
import numpy as np
from com.object.obj_entity import train_future, train_total, future_baseInfo
from com.data.interface_Rice import interface_Rice
import time
import uuid
from multiprocessing.managers import BaseManager
from multiprocessing import Pool
import copy

# 回归方法
class MyManager(BaseManager):
    pass

MyManager.register('interface_Rice', interface_Rice)
MyManager.register('future_baseInfo', future_baseInfo)
def Manager2():
    m = MyManager()
    m.start()
    return m

# 回归方法
class train_future_jump_sg(object):
    """

    """

    csvList = ['NI']

    def __init__(self):
        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.onlyCVS = False
        self.isEmptyUse = True    # 是否清空记录

        self.processCount = 5
        self.periodwidth = 10
        self.bullwidth = 2

        self.keepDaysList = [3]
        self.keepDays = 3
        self.dissLineList = [2]
        self.dissLine = 2
        self.maPeriodList = [20]
        self.maPeriod = 5

        self.iniAmount = 5000000  # 单边50万

        # self.iniAmount = 2000000 * 0.0025  # 单边50万
        self.stopLine = 0.0025
        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.indexList = [c[0] for c in self.indexCodeList]
        self.klineList = ['5m']

        # 起始时间
        self.startDate = '2019-04-01'
        self.endDate = '2019-06-27'

        self.total_tablename = 'train_total_3'
        self.detail_tablename = 'train_future_3'
        self.method = 'jumpsg2'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method
        self.isAll = 0

    def iterCond(self):
        # 多重组合参数输出
        #for i in range(1):
        #    yield '%s_%s' % (str(self.preDays), str(self.keepDays))

        keys = ['keepDays', 'maPeriod']
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                 self.__setattr__(keys[1], s1)
                 yield '%s_%s' % (str(s0),str(s1))

        #        for s2 in self.__getattribute__(keys[2] + 'List'):
        #            self.__setattr__(keys[2], s2)

    def switch(self):
        # 生成all
        if self.isAll == 1:
            self.isEmptyUse = True
            self.total_tablename = 'train_total_0'
            self.detail_tablename = 'train_future_0'
        self.empty()

    def empty(self):
        if self.isEmptyUse and not self.onlyCVS:
            Train = train_future()
            Total = train_total()
            Total.tablename = self.total_tablename
            Train.tablename = self.detail_tablename
            Train.empty()
            Total.empty()

    def Pool(self):
        time0 = time.time()
        # 清空数据库
        self.switch()

        pool = Pool(processes=self.processCount)
        #share = Manager2()
        Base = future_baseInfo()
        lists = Base.getUsedMap(hasIndex=False, isquick=False) + ['PB', 'SN', 'CY']

        print(len(lists))

        for rs in lists:
            # 检查时间匹配
            #if not rs in self.csvList: continue

            codes = [rs]
            for kline in self.klineList:
                # try:
                self.start(codes, time0, kline)
                #pool.apply_async(self.start, (codes, time0, kline))

            # except Exception as e:
            #     print(e)
            #     continue

        pool.close()
        pool.join()

    cindex = 0

    def start(self, codes, time0, kline=None, Base=None, Rice=None):
        print("子进程启动:", self.cindex, codes, kline)

        # 主力合约
        self.codes = codes
        self.code, self.mCode = codes[0], codes[0] + '88'

        self.mCodes = mCodes = [n + '88' for n in codes]
        # 查询获得配置 - 费率和每手单量

        self.Base = future_baseInfo()

        self.BS = {}

        for doc in self.Base.getInfo(codes):
            self.BS[doc["code"]] = self.BS[doc["code"] + '88'] = doc

        cs = [self.BS[m] for m in self.mCodes]
        self.klineType = self.BS[self.code]['quickkline'] if kline is None else kline


        # 子进程共享类
        self.Rice = Rice if Rice is not None else interface_Rice()
        self.Rice.setTimeArea(cs[0]["nightEnd"])

        self.Train = train_future()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename
        self.Train.tablename = self.detail_tablename

        if len(self.indexCodeList) > 0:
            self.Rice.setIndexList(self.indexCodeList)

        self.adjustDates = self.Rice.getAdjustDate(self.code, start=self.startDate)

        print(codes, self.adjustDates)

        # 查询获得N分钟K线
        dfs = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=1)

        print('kline load:', mCodes, [len(dfs[m]) for m in mCodes])

        # 根据配置文件获取最佳交易手数对
        self.iniVolume = round(self.iniAmount / cs[0]["lastPrice"] / cs[0]["contract_multiplier"], 0)
        if self.iniVolume == 0: self.iniVolume = 1

        # 分参数执行
        docs = self.total(dfs, period=self.periodwidth)
        if docs is None or len(docs) == 0: return

        logger.info((self.codes, self.klineType, len(docs), " time:", time.time() - time0))
        self.Total.insertAll(docs)


    pointColumns = ['jump', 'open0', 'mp', 'powm', 'powm1', 'ma', 'shift', 'delta', 'widthDelta', 'p_h', 'p_l', 'mastd', 'macd2d', 'std', 'atr']
    def jump(self, row, df0, df1):
        # df1 日线 ，df0 5分钟线
        r0 = df1.loc[row['trading_date']]
        jump, open0, ma = r0['jump'], r0['open'],  r0['ma']
        rs = [r0[k] for k in self.pointColumns[-2:]]

        # 早盘跳空
        jump_n, open_n = r0['jump_n'], r0['open_n']
        #
        close = row['close']
        #
        period = self.periodwidth if self.maPeriod==1 else self.maPeriod
        # ma修正
        ma = (ma * period - r0['close'] + close) / period
        # bias
        bias = (close - ma) / ma * 100 if not np.isnan(ma) else 0

        sub = df0[(df0['trading_date']==row['trading_date'])].loc[:row['datetime']]
        max = sub['low'].min() if jump > 0 else sub['high'].max()

        opt0 = jump * (close - open0) > 0
        opt1 = (max - (open0 - jump)) * jump > 0
        powm = np.sign(jump) if opt0 and opt1 else 0

        # -----辅助判断----- ---------
        opts= [0 for i in range(6)]
        # 补缺反向
        # max_1 = r0['high_1'] if jump < 0 else r0['low_1']  # 前一日最高/最低
        opts[0] = - np.sign(jump) if (jump * (close - open0 + jump) < 0) else 0

        # 早盘方向
        opts[1] = np.sign(jump_n) if jump_n * (close - open_n) > 0 else 0

        # 均线方向
        opts[2] = np.sign(bias) if bias != 0 else 0

        # bias 过大
        opts[3] = -np.sign(bias) if abs(bias) > 2.75 else 0

        # 跨越
        opts[4] = np.sign(bias) if bias * r0['bias_1'] < 0 else 0

        # 超大ATR
        opts[5] = - np.sign(bias) if r0['atrc'] > 3.2 else 0

        # 影线

        return pd.Series([jump, open0, max, powm, [opts], ma, bias] + opts + rs, index=self.pointColumns)

    def getMax(self, df0, s, e, mode):
        if mode > 0:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'high'].max()
        else:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'low'].min()

    preNode, batchId = None, {}

    def total(self, dfs, period=21):
        # 计算参数
        df0 = dfs[self.mCodes[0]]
        close = df0["close"]
        df0["datetime"] = df0.index
        df0["time"] = df0["datetime"].apply(lambda x: str(x).split(' ')[1])
        # 夜盘
        df2 = df0[(df0['time']=='09:05:00') | (df0.shift(-1)['time']=='09:05:00')]
        df2['jump_n'] = df2['open'] - df2['close'].shift(1)

        # 日线数据
        self.df_day = df1 = self.Rice.kline(self.mCodes, period='1d', start=self.startDate, end=self.endDate, pre=21)[self.mCodes[0]]
        df1['jump'] = df1['open'] - df1['close'].shift(1)
        df1['atr'] = ta.ATR(df1['high'], df1['low'], df1['close'], timeperiod=21).fillna(1).shift(1)
        df1["std"] = ta.STDDEV(df1["close"].shift(1), timeperiod=period, nbdev=1)

        # 夜盘跳空写入日线
        for dd in df1.index:
            df1.loc[dd, 'jump_n'] = df2.loc[str(dd)[:10] + ' 09:05:00', 'jump_n'] if str(dd)[:10] + ' 09:05:00' in df2.index else 0
            df1.loc[dd, 'open_n'] = df2.loc[str(dd)[:10] + ' 09:05:00', 'open'] if str(dd)[
                                                                                     :10] + ' 09:05:00' in df2.index else 0

        df1['mVol'] = self.iniAmount * self.stopLine / self.BS[self.codes[0]]["contract_multiplier"] / (2 * df1['atr'])

        # 循环 scale
        docs = []
        for conds in self.iterCond():
            uid = self.uidKey % ('_'.join(self.codes), str(period),  self.klineType, str(self.bullwidth), conds)

            df1["ma"] = ta.MA(df1["close"], timeperiod= period if self.maPeriod==1 else self.maPeriod) if self.maPeriod> 0 else 0
            df1['bias_1'] = (100 * (df1['close'] / df1['ma'] - 1)).shift(1)  # 前一个Bias 值
            df1['atrc'] = (ta.ATR(df1['high'], df1['low'], df1['close'], timeperiod=1) / df1['atr']).shift(1)

            df0 = copy.deepcopy(df0[(df0['time'] >'09:10:00') & (df0['time'] <'09:45:00')][self.startDate:])

            df2 = df0.apply(lambda row: self.jump(row, df0, df1), axis=1)
            for key in self.pointColumns:  df0[key] = df2[key]

            #df0.fillna(0, inplace=True)

            if self.code in self.csvList:
                file = self.Rice.basePath + '%s_sg.csv' % (uid)
                file1 = self.Rice.basePath + '%s_sg1.csv' % (uid)
                print(uid, '---------------------------- to_cvs', file, df0.columns)
                df0.to_csv(file, index=0, columns= ['datetime', 'time', 'volume', 'close']+self.pointColumns)
                #df1.to_csv(file1, index=1)

            if self.onlyCVS: return None

            #tot = None
            tot = self.detect(df0, period=period, uid=uid)
            if tot is not None and tot['amount'] != 0:
                tot.update(
                    {
                        "method": self.method,
                        "code": self.code,
                        "period": period,
                        "uid": uid,
                        "createdate": public.getDatetime()
                    }
                )
                docs.append(tot)
        return docs

    def detect(self, df0, period=14, uid=''):
        docs = self.stageApply(df0, period=period, uid=uid)
        res = pd.DataFrame(docs)

        if len(res) > 0:
            if self.saveDetail:
                self.Train.insertAll(docs)

            diff = res[res['diff'] > 0]['diff'].mean()
            self.testDays = public.dayDiff(self.endDate, self.startDate)

            # 计算最大回测
            sum = res['income'].cumsum() + self.iniAmount
            inr = res['income'] / self.iniAmount
            # 计算夏普指数
            sha = (res['income'].sum() / self.iniAmount - 0.02 * self.testDays / 252) / inr.std()

            doc = {
                "count": int(len(docs) / 2),
                "amount": self.iniAmount,
                "price": res['price'].mean(),
                "income": res["income"].sum(),
                # "delta": res['delta'].mean(),
                "maxdown": ((sum.shift(1) - sum) / sum.shift(1)).max(),
                "sharprate": sha,
                "timediff": int(0 if np.isnan(diff) else diff)
            }

            print(self.code, doc['count'], doc['income'])
            return doc
        else:
            return None

    # 核心策略部分
    def stageApply(self, df0, period=0, uid=''):
        isOpen, preDate, prePrice = 0, None, 0
        doc, docs = {}, []
        self.preNode, self.jumpNode, self.jump_index = None, None, 0

        adjusts = [c[1] for c in self.adjustDates]

        for i in range(period, len(df0)):
            isRun, isstop = False, 0
            powm,powm1,close, ma, atr,dd, date,time = (df0.ix[i, key] for key in
                                             "powm,powm1,close,ma,atr,datetime,trading_date,time".split(","))

            opt0 = ((close - ma) * powm > 0) if self.maPeriod > 1 else True

            pow = powm + np.sign(powm1)

            if isOpen == 0:
                if powm != 0 and opt0:
                    isRun, isOpen = True, int(powm)

            elif self.preNode is not None and isOpen!=0:
                # 调仓
                 if date in adjusts and time== '09:15:00':
                    docs += self.adjustOrder(df0.iloc[i], date, dd)

                 pN = self.preNode[0]
                 sign = np.sign(pN['mode'])
                 pmax = self.getMax(df0, pN['createdate'], dd, sign)

                 keep = len(df0[self.preNode[0]['createdate']:dd]) // 5

                 if sign * pow < 0 and keep >= self.keepDays:
                     doc = self.order(df0.iloc[i], 0, uid,  df0, isstop=2)
                     isOpen = 0
                     if doc is not None:
                         docs.append(doc)
                         if opt0 and powm!=0: # 反向开仓
                             isRun, isOpen = True, int(powm)

                 elif sign * (pmax - close) > self.dissLine * atr:
                     isRun, isOpen, isstop = True, 0, 4

            if isRun:
                doc = self.order(df0.iloc[i], isOpen, uid, df0, isstop=isstop)
                if doc is not None:
                    docs.append(doc)
        return docs

    def adjustOrder(self, n0, date, dd):
        preCode = ''
        for c in self.adjustDates:
            if c[1] == date:
                preCode = c[2]
                break

        s = str(date).split(" ")[0]

        df1 = self.Rice.kline([preCode], period=self.klineType, start=public.getDate(diff=-2, start= s), end=s, pre=0)
        oldP = df1[preCode].loc[dd, 'close']
        newP = n0['close']

        # 调仓卖出
        doc = copy.deepcopy(self.preNode[0])
        sign = np.sign(self.preNode[0]['mode'])
        pp = self.preNode[0]['price']

        doc['price'] = oldP
        doc['isopen'] = 0
        doc['mode'] = -doc['mode']
        doc['isstop'] = 6
        doc['createdate'] = date
        doc['income'] = sign * (oldP - pp) * doc["vol"] - doc["fee"]

        doc1 = copy.deepcopy(self.preNode[0])
        doc1['createdate'] = date
        doc1['mode'] = int(6 * sign)
        doc1['price'] = newP
        doc1['isopen'] = 1
        #doc1['batchid'] = self.batchId = uuid.uuid1()

        #self.records.append(doc1)
        self.preNode = [doc1]
        return [doc, doc1]

    batchId = None
    def calcIncome(self, n0, p0, df0):
        # 计算收益，最大/最小收益
        high = df0[(p0['createdate'] <= df0.index) & (df0.index <= n0['datetime'])]['high'].max()
        low = df0[(p0['createdate'] <= df0.index) & (df0.index <= n0['datetime'])]['low'].min()
        close = n0["close"]
        sign = p0["mode"] / abs(p0["mode"])

        # 收入
        income = sign * (close - p0["price"]) * p0["vol"] - p0["fee"]
        # 最大收入
        highIncome = sign * ((high if sign > 0 else low) - p0["price"]) * p0["vol"] - p0["fee"]
        # 最大损失
        lowIncome = sign * ((high if sign < 0 else low) - p0["price"]) * p0["vol"] - p0["fee"]

        return income, highIncome, lowIncome

    def order(self, n0, mode, uid, df0, isstop=0):
        # BS 配置文件，查询ratio 和 每手吨数
        b0 = self.BS[self.mCode]
        if mode != 0:
            self.batchId = uuid.uuid1()

        # 交易量
        if np.isnan(n0['atr']) or n0['atr'] < 2:
            hands = int(self.iniAmount/10/n0['close']/b0["contract_multiplier"])
        else:
            hands = int(self.iniAmount * self.stopLine / 2 / n0['atr'] / b0["contract_multiplier"])

        #print(hands)

        v0 = hands * b0["contract_multiplier"]
        # 费率
        fee0 = (hands * b0["ratio"] * 1.1) if b0["ratio"] > 0.5 else ((b0["ratio"] * 1.1) * n0["close"] * v0)

        ps = n0["close"]
        amount = v0 * ps * float(b0['margin_rate'])

        doc = copy.deepcopy(n0.to_dict())
        doc.update({
            "code": self.codes[0],
            "createdate": n0["datetime"],
            "price": ps,
            "vol": self.preNode[0]["vol"] if self.preNode else v0,
            "hands": self.preNode[0]["hands"] if self.preNode else hands,
            "amount": amount if not self.preNode else self.preNode[0]["amount"],
            "mode": int(mode) if not self.preNode else -self.preNode[0]["mode"],
            "isopen": 0 if mode == 0 else 1,
            "fee": fee0,
            "income": 0,
            "isstop": isstop,
            #"result": int(n0["result"]) if 'result' in n0 else 0,
            #"options": int(n0["powm1"]) if 'powm1' in n0 else 0,
            "rel_std": n0["std"] if 'std' in n0 else 0,
            "macd": n0["ma"] if 'ma' in n0 else 0,
            "batchid": self.batchId,
            "diff": 0 if mode != 0 else self.Rice.timeDiff(str(self.preNode[0]['createdate']), str(n0["datetime"])),
            "uid": uid
        })

        if mode == 0 and self.preNode:
            p0 = self.preNode[0]
            doc['income'], doc['highIncome'], doc['lowIncome'] = self.calcIncome(n0, p0, df0)
            doc["diff"] = int(public.timeDiff(str(n0['datetime']), str(p0['createdate'])) / 60)
            self.preNode = None
        else:
            doc["income"] = -doc["fee"]
            self.preNode = [doc]

        return doc

def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = train_future_jump_sg()
        obj.Pool()


if __name__ == '__main__':
    main()
