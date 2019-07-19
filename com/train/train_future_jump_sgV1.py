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
from com.object.obj_entity import train_future, train_total, future_baseInfo, future_orderForm
from com.object.obj_model import future_code
import math
from com.data.interface_Rice import interface_Rice
import time
import uuid
from multiprocessing import Pool
import copy

# 回归方法
class train_future_jump_sg(object):
    """

    """

    csvList = ['P', 'RB', 'MA', 'I']


    def __init__(self):
        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.onlyCVS = False
        self.isEmptyUse = True   # 是否清空记录

        self.testDays = 0

        # 固定参数
        self.bullwidth = 2
        self.processCount = 4
        # 交易量
        self.iniAmount = 5000000  # 单边50万
        self.stopLine = 0.0025

        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.klineList = ['5m']

        # 交易时间起始时间
        self.startDate = '2014-03-13'
        self.endDate = self.int_endDate = '2019-06-27'

        # 数据源类型 0:主力合约 1：sg对应合约时间 2：所有满足交易量合约
        self.sourceListType = 2

        # ----------- 可变参数 -------------
        # 持仓时间
        self.keepDaysList = [1]
        self.keepDays = 3

        # 均线
        self.periodwidthList = [20]
        self.periodwidth = 10

        # 单向容量
        self.maxMultiList = [3]
        self.maxMulti = 3  # 相对于2倍ATR的 单边最大容量

        # 平仓交易比例
        self.closeRatioList = [3]
        self.closeRatio = 3

        # 主交易类型选择：
        self.tradeTypeList = [1]
        self.tradeType = 0  # 0:只正向 1:允许反向

        # bias 反转
        self.biasBigLineList = [2.5]
        self.biasBigLine = 2.75

        # 止损倍数
        self.sellLineList = [2]
        self.sellLine = 2

        # 止损倍数
        self.sameParamList = [1, 3]
        self.sameParam = 1

        # mid比例
        self.midParamList = [0.30]
        self.midParam = 0.2

        # 辅助交易集成参数
        self.powmsParamConfig = {
           'count': 5, 'range': [0, 0.5, 1, 2]
        }

        self.powmsParamList = [0.5, 0, 1, 0, 0]

        print(self.endDate)
        self.total_tablename = 'train_total_3'
        self.detail_tablename = 'train_future_3'
        self.method = 'jumpsg'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method
        self.isAll = 0


    def iterCond(self):
        # 多重组合参数输出
        # for i in range(1):
        #    yield '%s_%s' % (str(self.preDays), str(self.keepDays))

        keys = ['midParam', 'tradeType']
        for s0 in self.__getattribute__(keys[0] + 'List'):
            self.__setattr__(keys[0], s0)

            for s1 in self.__getattribute__(keys[1] + 'List'):
                self.__setattr__(keys[1], s1)

                #print(s0,s1)
                yield '%s_%s' % (str(s0), str(s1))

        #        for s2 in self.__getattribute__(keys[2] + 'List'):
        #            self.__setattr__(keys[2], s2)

        """
        count = self.powmsParamConfig['count']
        for s0 in range(count):
            params = [1 for i in range(count)]
            for s1 in self.powmsParamConfig['range']:
                params[s0] = s1
                self.powmsParamList = params
                yield '%s_%s' % (str(s0), str(s1))
         """

    def switch(self):
        # 生成all
        if self.isAll == 1:
            self.isEmptyUse = True
            self.total_tablename = 'train_total_0'
            self.detail_tablename = 'train_future_0'
        self.empty()

    def empty(self):
        if self.isEmptyUse:
            Train = train_future()
            Total = train_total()
            Total.tablename = self.total_tablename
            Train.tablename = self.detail_tablename
            Train.empty()
            Total.empty()

    def getList(self, type=0):
        """
            查询获取可交易的合约模式
        """
        lists = []
        if type == 0:
            # 主力合约模式
            Base = future_baseInfo()
            res = Base.getUsedMap(hasIndex=False, isquick=False) + ['PB', 'SN', 'CY']
            lists = [
                {'code': d[0] + '88',
                 'name': d[0],
                 'startdate': self.startDate,
                 'enddate': self.endDate,
                 } for d in res]

        elif type == 1:
            # 与SG完全对应合约模式
            Order = future_orderForm()
            Order.tablename = 'future_orderForm_sg'
            lists = Order.sg_codes()

        elif type == 2:
            # 非主力合约模式
            Order = future_code()
            lists = Order.getMap(start=self.startDate, end=self.endDate)

        return lists

    def Pool(self):
        """
            主控入口
        """
        time0 = time.time()
        # 清空数据库
        self.switch()

        pool = Pool(processes=self.processCount)
        lists = self.getList(self.sourceListType)

        for rs in lists:
            #if not rs['name'] in self.csvList: continue
            codes = [rs]
            for kline in self.klineList:
                #self.start(codes, time0, kline)
                try:
                    pool.apply_async(self.start, (codes, time0, kline))
                    #time.sleep(0.5)
                    pass

                except Exception as e:
                    print(e)
                    continue

        pool.close()
        pool.join()

    cindex = 0

    def start(self, rs, time0, kline=None):
        """
            子进程入口
        """
        print("子进程启动:", self.cindex, rs, kline)

        self.Rice =  interface_Rice()

        #####   子程序初始化

        self.codes = codes = [d['name'] for d in rs]
        if self.sourceListType == 1:
            self.mCodes = mCodes = [self.Rice.id_convert(d['code']) for d in rs]
            self.ctpCode = rs[0]['code']

        else:
            self.mCodes = mCodes = [d['code'] for d in rs]
            self.ctpCode = rs[0]['code']

        # 单一合约
        self.code, self.mCode = self.codes[0], self.mCodes[0]

        self.iniVolume = 0
        if 'startdate' in rs[0]:
            self.startDate = str(rs[0]['startdate'])[:10]
            self.endDate = str(rs[0]['enddate'])[:10]

        if 'iniVolume' in rs[0]:
            self.iniVolume = rs[0]['iniVolume']

        self.klineType = kline

        self.Train = train_future()
        self.Total = train_total()
        self.Total.tablename = self.total_tablename
        self.Train.tablename = self.detail_tablename

        self.Base = future_baseInfo()
        self.BS = {}
        # 查询获得配置 - 费率和每手单量
        for doc in self.Base.getInfo(codes):
            self.BS[doc["code"]] = self.BS[self.mCode] = doc

        cs = [self.BS[m] for m in self.mCodes]
        # 子进程共享类
        self.Rice.setTimeArea(cs[0]["nightEnd"])

        if len(self.indexCodeList) > 0:
            self.Rice.setIndexList(self.indexCodeList)

        #self.adjustDates = self.Rice.getAdjustDate(self.code, start=self.startDate)
        #print(codes, self.adjustDates)

        """"
           -------------- 调用sub程序 -----------------
        
        """
        # 查询获得N分钟K线
        dfs = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=1)
        print('kline load:', mCodes, [len(dfs[m]) for m in mCodes])

        # 分参数执行
        docs = self.total(dfs, period=self.periodwidth)
        if docs is None or len(docs) == 0: return

        logger.info((self.codes, self.klineType, len(docs), " time:", time.time() - time0))
        self.Total.insertAll(docs)

    pointColumns = ['jump', 'open0', 'mid', 'powm', 'powm1', 'powms', 'ma', 'jump_n', 'bias', 'bias_1', 'atr', 'std',
                    'mVol']

    def jump(self, row, df0, df1):
        # df1 日线 ，df0 5分钟线
        r0 = df1.loc[row['trading_date']]

        #print(r0['date'], r_1['date'])

        jump, open0, ma = r0['jump'], r0['open'], r0['ma']
        rs = [r0[k] for k in self.pointColumns[-4:]]
        #
        # 早盘跳空
        jump_n, open_n = r0['jump_n'], r0['open_n']
        #
        close = row['close']
        # ma修正
        ma = (ma * self.periodwidth - r0['close'] + close) / self.periodwidth
        # bias 值
        bias = (close - ma) / ma * 100 if not np.isnan(ma) else 0
        # bias 跨越

        sub = df0[(df0['trading_date'] == row['trading_date'])].loc[:row['datetime']]
        max = sub['low'].min() if jump > 0 else sub['high'].max()

        # ---- 主判断, 日盘跳空且未补缺 -----------------
        opt0 = jump * (close - open0 + jump/2) > 0
        opt1 = ((max - (open0 - jump)) * jump > 0)

        #powm1 = np.sign(jump) if (opt0 and opt1) else 0

        #  ---- 补缺反向 ----

        sign = np.sign(close - open0)
        max_1 = r0['high_1'] if sign > 0 else r0['low_1']  # 前一日最高/最低
        opt11 = sign * (close - max_1) > 0

        min, max = sub['low'].min(), sub['high'].max()
        idmin , idmax = sub['low'].idxmin(), sub['high'].idxmax()

        ss = (close - min ) / (max-min) - 0.5
        opt12 = (idmin > idmax) * ss > 0 and (max - min) / r0['std'] > 0.8

        opt22 = np.sign(ss) * (ss - np.sign(ss) * self.midParam) > 0

        powm = np.sign(ss) if opt22 and (max - min) / r0['std'] > 1 else 0
        #powmL = [sign if opt11 else 0, np.sign(jump) if (opt0 and opt1) else 0 , np.sign(ss) if opt12 else 0]
        powmL = [np.sign(ss) if opt12 else 0]
        powm1 = sum(powmL)

        # -----辅助判断----- ---------
        opts = [0 for i in range(self.powmsParamConfig['count'])]
        params = self.powmsParamList

        # 超大ATR
        opts[0] = powm * params[0] * int(math.log(r0['atrc']))  if r0['atrc'] > 1 else 0

        # 早盘方向
        opts[1] = np.sign(jump_n) * params[1] if jump_n * (close - open_n) > 0 else 0

        # 均线方向
        opts[2] = np.sign(bias) * params[2] if bias != 0 else 0

        # bias 过大
        opts[3] = -np.sign(bias) * params[3] if abs(bias) > 3 else 0

        # 跨越
        opts[4] = np.sign(bias) * params[4] if bias * r0['bias_1'] < 0 else 0

        powms = opts

        return pd.Series([jump, open0, powmL, powm, powm1, powms, ma, jump_n, bias] + rs, index=self.pointColumns)

    preNode, batchId = None, {}

    def total(self, dfs, period=21):
        # 计算参数
        df01 = dfs[self.mCodes[0]]
        close = df01["close"]
        df01["datetime"] = df01.index
        df01["time"] = df01["datetime"].apply(lambda x: str(x).split(' ')[1])

        # 日线数据
        self.df_day = df1 = self.Rice.kline(self.mCodes, period='1d', start=self.startDate, end=self.endDate, pre=21)[
            self.mCodes[0]]
        df1['date'] = df1.index
        df1['jump'] = df1['open'] - df1['close'].shift(1)
        df1["low_1"], df1["high_1"] = df1["low"].shift(1), df1["high"].shift(1)
        df1["std"] = ta.STDDEV(df1["close"], timeperiod=period, nbdev=1)
        df1['atr'] = ta.ATR(df1['high'], df1['low'], df1['close'], timeperiod=21).fillna(1).shift(1)

        ss = int(self.iniAmount / self.BS[self.code]["contract_multiplier"])
        # 计算交易量
        df1['mVol'] = df1.apply(lambda r: ss * self.stopLine / 2 / r['atr'] if r['atr'] > 1 else ss/10/r['close'], axis=1)

        # 夜盘跳空写入日线
        # 夜盘
        df2 = df01[(df01['time'] == '09:05:00') | (df01.shift(-1)['time'] == '09:05:00')]
        df2['jump_n'] = df2['open'] - df2['close'].shift(1)
        for dd in df1.index:
            df1.loc[dd, 'jump_n'] = df2.loc[str(dd)[:10] + ' 09:05:00', 'jump_n'] if str(dd)[
                                                                                     :10] + ' 09:05:00' in df2.index else 0
            df1.loc[dd, 'open_n'] = df2.loc[str(dd)[:10] + ' 09:05:00', 'open'] if str(dd)[
                                                                                   :10] + ' 09:05:00' in df2.index else 0

        # 循环 scale
        docs = []
        for conds in self.iterCond():
            self.uid = uid = self.uidKey % ('_'.join(self.mCodes), str(period), self.klineType, str(self.bullwidth), conds)
            # print(uid)
            df1["ma"] = ta.MA(df1["close"], timeperiod=period)
            df1['bias_1'] = (100 * (df1['close'] / df1['ma'] - 1)).shift(1)  # 前一个Bias 值
            df1['atrc'] = (ta.ATR(df1['high'], df1['low'], df1['close'], timeperiod=1) / df1['atr']).shift(1)

            df0 = copy.deepcopy(df01[(df01['time'] > '09:10:00') & (df01['time'] < '10:15:00')][self.startDate:])


            df3 = df01.apply(lambda row: self.jump(row, df01, df1), axis=1)
            for key in self.pointColumns: df0[key] = df3[key]

            if self.code in self.csvList:
                file = self.Rice.basePath + '%s_sg.csv' % (uid)
                file1 = self.Rice.basePath + '%s_sg1.csv' % (uid)
                print(uid, '---------------------------- to_cvs', file, df0.columns)
                df0.to_csv(file, index=0, columns=['datetime', 'time', 'volume', 'close'] + self.pointColumns)
                #df1.to_csv(file1, index=1)

            if self.onlyCVS: return None

            # return
            # tot = None
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

            # 计算最大回测
            sum = res['income'].cumsum() + self.iniAmount
            inr = res['income'] / self.iniAmount
            # 计算夏普指数
            #sha = (res['income'].sum() / self.iniAmount - 0.02 * self.testDays / 252) / inr.std()

            doc = {
                "count": int(len(docs) / 2),
                "amount": self.iniAmount,
                "price": res['price'].mean(),
                "income": res["income"].sum(),
                # "delta": res['delta'].mean(),
                "maxdown": ((sum.shift(1) - sum) / sum.shift(1)).max(),
                "sharprate": 0,
            }
            print(self.uid, doc['count'], doc['income'])
            return doc
        else:
            return None

    def getMax(self, df0, s, e, mode):
        if mode > 0:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'high'].max()
        else:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'low'].min()


    # 根据交易辅助条件和每日mVol，计算每5分钟可交易数量
    def getUnit(self, mVol, keepUnit, powm, powm1, powms=0):
        skip, unit = 0, 0

        # 每天执行一半操作
        pow = powm + powm1 * 0.5
        if pow != 0:
            pow = abs(powm + powm1 * 0.5 + sum(powms) * 0.25)
        #elif powm != 0:
        #    pow = abs(powm + sum(powms) * 0.25)

        else:
            return (0, 0, 0)

        mV = int((mVol * pow + 0.3) / 2)

        if mV == 0: return (0, 0, 0)

        while unit < 1:
            skip += 1
            unit = (mV * skip) // keepUnit

        # 每日可交易频次
        dim = keepUnit if skip == 1 else int((keepUnit + 1) / skip)

        return (skip, unit, mV % dim)

    # 核心策略部分
    def stageApply(self, df0, period=15, uid=''):
        isOpen, preDate, prePrice = 0, None, 0
        doc, docs = {}, []
        self.preNode, self.jumpNode, self.jump_index = None, None, 0

        #adjusts = [c[1] for c in self.adjustDates]
        # 所有交易列表清单
        self.perservList,self.batchStartDate = [], None

        stockIndexList = [c[0] for c in self.indexCodeList]# 股指

        unit = (0, 0, 0)

        # 处理时间
        start = '09:40:00' if self.code in stockIndexList else '09:10:00'
        timeList = df0[(df0['trading_date'] == self.startDate) & (df0['time'] > start) & (df0['time'] < '10:15:00')][
            "time"].tolist()

        keepUnit = len(timeList)

        for i in range(period, len(df0)):
            isRun, isstop, sp = False, 0, None
            powm, powm1, powms, close, atr, dd, date, tt, mVol = (df0.ix[i, key] for key in
                                                                  "powm,powm1,powms,close,atr,datetime,trading_date,time,mVol".split(
                                                                      ","))

            if date != preDate and (powm != 0 or powm1 != 0):
                unit = self.getUnit(mVol, keepUnit, powm, powm1, powms)
                preDate = date

            # print(self.code, powm1, powms, mVol, unit)

            # 计算每天交易量单位和时间间隔
            self.totalVol = total = self.getCurrentVol()
            preMode = self.perservList[0]['mode'] if len(self.perservList) > 0 else 0

            # 新批次
            if total == 0:
                self.batchId = uuid.uuid1()

            if start < tt < '10:15:00':  # 限定交易时间
                index = timeList.index(tt)

                # 为空或者 unit间隔
                if unit[1] == 0 or index % unit[0] != 0: continue
                self.unit = unit[1] + 1 if (unit[2] > 0 and (index // unit[0]) < unit[2]) else unit[1]

                # 选择交易模式
                pp = (powm + powm1) if self.tradeType == 1 else powm

                if (total < self.maxMulti * mVol and pp * preMode > 0) or (total == 0 and abs(pp) != 0):
                    # 新开仓记录开仓时间用于止损
                    if total == 0: self.batchStartDate = dd

                    # 加仓
                    if powm != 0:
                        # 正向
                        isRun, isOpen = True, int(powm)
                    else:
                        # 反向
                        isRun, isOpen = True, int(powm1) * 2

                if total > 0 and pp * preMode < 0:
                    # 减仓
                    if powm != 0:
                        isRun, isOpen, isstop = True, 0, int(powm)

                    else:
                        isRun, isOpen, isstop = True, 0, int(powm1) * 2

            elif total > 0:
                # 2倍ATR止损
                sign = np.sign(preMode)

                # 从批次开仓时间计算止损
                pmax = self.getMax(df0, self.batchStartDate, dd, sign)

                if sign * (pmax - close) > self.sellLine * atr:
                    isRun, isOpen, isstop = True, 0, 4

                elif date == self.endDate and date!= self.int_endDate:
                    # 批次结束
                    isRun, isOpen, isstop = True, 0, 5

            if isRun:
                doc = self.order(df0.iloc[i], isOpen, uid, df0, isstop=isstop)
                if doc is not None and len(doc) > 0:
                    docs.extend(doc)
        return docs

    def getCurrentVol(self):
        return  sum([d['hands'] for d in self.perservList]) if len(self.perservList) > 0 else 0

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

    def closeOrder(self, n0, p0, df0, amount, isstop):
        doc = copy.deepcopy(p0)
        doc.update({
            "createdate": n0["datetime"],
            "price": n0['close'],
            "amount": amount,
            "mode": -p0['mode'],
            "isopen": 0,
            "isstop": isstop
        })
        doc['income'], doc['highIncome'], doc['lowIncome'] = self.calcIncome(n0, p0, df0)
        doc["diff"] = int(public.timeDiff(str(n0['datetime']), str(p0['createdate'])) / 60)
        return doc

    def order(self, n0, mode, uid, df0, isstop=0):
        #
        # 计算单笔量, 卖出翻倍
        closeV = self.closeRatio * self.unit if (self.totalVol > self.closeRatio * self.unit) else self.totalVol
        v0 = self.unit if mode != 0 else closeV

        # 费率
        b0 = self.BS[self.code]
        tuns = v0 * b0["contract_multiplier"]

        fee0 = (v0 * b0["ratio"] * 1.1) if b0["ratio"] > 0.5 else ((b0["ratio"] * 1.1) * n0["close"] * tuns)
        amount = tuns * n0['close'] * float(b0['margin_rate'])

        if mode == 0 and isstop in [4, 5] and len(self.perservList) > 0:
            # print(self.uid, n0['datetime'], isstop)
            # 止损全部平仓
            tmp = []
            for p0 in self.perservList:
                doc = self.closeOrder(n0, p0, df0, amount, isstop)
                doc['shift'] = self.getCurrentVol()
                tmp.append(doc)

            self.perservList = []
            return tmp

        elif mode == 0 and len(self.perservList) > 0:
            # 部分平仓
            rareVol, tmp = v0, []

            while rareVol > 0:
                p0 = self.perservList[0]
                doc = self.closeOrder(n0, p0, df0, amount, isstop)
                # pop
                rareVol += -p0['hands']  # 减去交易量
                self.perservList.pop(0)
                doc['shift'] = self.getCurrentVol()
                tmp.append(doc)
            return tmp

        elif mode != 0:
            # 新开仓或加仓
            doc = {
                "createdate": n0["datetime"],
                "code": self.codes[0],
                "price": n0['close'],
                "vol": tuns,
                "hands": v0,
                "amount": amount,
                "mode": int(mode),
                "isopen": 1,
                "fee": fee0,
                "income": -fee0,
                "isstop": 0,
                "batchid": self.batchId,
                "uid": uid,
                "pid": uuid.uuid1()
            }
            self.perservList.append(doc)
            doc['shift'] = self.getCurrentVol()

            return [doc]


def main():
    action = {
        "kline": 1,
    }

    if action["kline"] == 1:
        obj = train_future_jump_sg()
        obj.Pool()


if __name__ == '__main__':
    main()
