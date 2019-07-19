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

from com.data.interface_Rice import interface_Rice
from com.ctp.interface_pyctp import BaseInfo
import time
import uuid
from multiprocessing.managers import BaseManager
from multiprocessing.connection import Listener,  Client
from multiprocessing import Pool
import copy
import math 

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

    csvList = ['RU', 'RM', 'TA', 'CF', 'P']

    def __init__(self):
        # 费率和滑点
        self.saveDetail = True  # 是否保存明细
        self.onlyCVS = False
        self.isEmptyUse = True     # 是否清空记录
        self.isPool = True  ## 采用共享进程

        # 数据源类型 0:主力合约 1：sg对应合约时间 2：所有满足交易量合约
        self.sourceListType = 2

        self.processCount = 5

        self.bullwidth = 2
        self.isMaUseList = [1]
        self.isMaUse = 1

        self.keepDaysList = [3]
        self.keepDays = 3

        self.dissLineList = [2]
        self.dissLine = 2

        self.periodwidthList = [20]
        self.periodwidth = 20
        # mid比例
        self.midParamList = [0.25]
        self.midParam = 0.3

        # atrc比例1
        self.backParamList = [2.25]
        self.backParam = 1.8

        # mid比例
        self.atrcParamList = [0.5]
        self.atrcParam = 1

        self.wavePeriod = 7

        self.iniAmount = 4000000  # 单边50万
        self.stopLine = 0.005

        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.klineList = ['5m']

        # 起始时间
        self.startDate = '2018-04-01'
        self.endDate = self.ini_endDate = '2019-06-26'

        self.total_tablename = 'train_total_2'
        self.detail_tablename = 'train_future_2'
        self.method = 'jumpsg2'
        self.uidKey = "%s_%s_%s_%s_%s_" + self.method
        self.isAll = 0

        self.address = ('localhost', 6000)
        self.authkey = b'ctpConnect'

    def iterCond(self):
        # 多重组合参数输出
        #for i in range(1):
        #    yield '%s_%s' % (str(self.preDays), str(self.keepDays))

        keys = ['midParam', 'backParam']
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
        if self.isEmptyUse :
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

    def listen(self, main):
        listener = Listener(self.address, authkey=self.authkey)
        Train = train_future()
        Total = train_total()
        Total.tablename = self.total_tablename
        Train.tablename = self.detail_tablename

        while True:
            conn = listener.accept()
            data = conn.recv()

            if 'end' in data and data['end']==1:
                print('------------end-------------')
                conn.close()
                break

            result=b'0'
            try:
                if data[0]==0:
                    Total.insertAll(data[1])
                else:
                    Train.insertAll(data[1])

                result = b'1'
                conn.send_bytes(result)
            except Exception as e:
                print(e)
                conn.send_bytes(result)

            finally:
                conn.close()


    def closeListen(self, tt):
        conn = Client(self.address, authkey=self.authkey)
        conn.send({'end':1, 'a':tt})
        conn.close()


    def Pool(self):
        """
            主控入口
        """
        time0 = time.time()
        # 清空数据库
        self.switch()
        print(1111)
        pool = Pool(processes=self.processCount)
        lists = self.getList(self.sourceListType)
        print(len(lists))

        share = Manager2()
        Base = share.future_baseInfo()

        pool.apply_async(self.listen, (1,))

        for rs in lists:
            #if not rs['name'] in self.csvList: continue
            codes = [rs]
            for kline in self.klineList:
                #self.start(codes, time0, kline, Base)
                try:
                    pool.apply_async(self.start, (codes, time0, kline, Base))
                    time.sleep(0.5)
                    pass

                except Exception as e:
                    print(e)
                    continue

        while True:
            num = len(pool._cache)
            if num < 2:
                pool.apply_async(self.closeListen, (1,))
                break
            time.sleep(1)

        pool.close()
        pool.join()

    cindex = 0

    def start(self, rs, time0, kline=None, Base=None):
        """
            子进程入口
        """
        print("子进程启动:", self.cindex, rs, kline)

        self.Rice = interface_Rice()

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


        #Base = future_baseInfo()
        self.BS = {}

        print('1-----------------------')
        # 查询获得配置 - 费率和每手单量
        docs = Base.getInfo(codes)
        print('2-----------------------')
        for doc in docs:
            self.BS[self.code] = self.BS[self.mCode] = doc

        cs = [self.BS[m] for m in self.mCodes]

        # 子进程共享类
        self.Rice.setTimeArea(cs[0]["nightEnd"])

        print('3-----------------------')
        if len(self.indexCodeList) > 0:
            self.Rice.setIndexList(self.indexCodeList)

        """"
           -------------- 调用sub程序 -----------------

        """

        # 查询获得N分钟K线
        dfs = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=1)
        print('kline load:', mCodes, self.startDate, self.endDate, [len(dfs[m]) for m in mCodes])
        print('4-----------------------')
        # 分参数执行
        docs = self.total(dfs, period=self.periodwidth)
        if docs is None or len(docs) == 0: return
        logger.info((self.mCodes, self.klineType, docs[0]['count'], docs[0]['income'], ''" time:", time.time() - time0))

        if docs is not None:
            if self.isPool:
                conn = Client(self.address, authkey=self.authkey)
                conn.send([0, docs])
                conn.close()

            else:
                Total = train_total()
                Total.tablename = self.total_tablename
                Total.insertAll(docs)

        #self.Total.insertAll(docs)


    pointColumns = ['jump', 'open0', 'midratio', 'atr', 'atrc', 'ma',  'powm',   'powm1', 'bias', 'Options']
    def jump(self, row, df0, df1):
        # df1 日线 ，df0 5分钟线
        r0 = df1.loc[row['trading_date']]
        r1 = df1.shift(1).loc[row['trading_date']]

        jump, open0, atr1 = r0['jump'], r0['open'], r1['atr']
        #
        close, close1 = row['close'], r1['close']
        # ma修正
        ma = (r0['ma'] * self.periodwidth - r0['close'] + close) / self.periodwidth
        map = {}
        for n in [5, 10, 20]:
            map[n] = (r0['ma'+str(n)] * n - r0['close'] + close) / n

        bias = (close -ma) / ma * 100

        # isout
        isup = np.sign(bias) if abs(bias) > 0 else 0

        sub = df0[(df0['trading_date'] == row['trading_date'])].loc[:row['datetime']]

        mim, mam = sub['low'].min(), sub['high'].max()
        mum = mim if jump > 0 else mam

        # 即时ATR 即 ATRC
        atr_t = max([float(mam -mim), abs(close1 - mim), abs(close1-mam)])
        atr = (atr1 * 13 + atr_t) / 14
        atrc = round(atr_t * 14 / (atr1 * 13 + atr_t), 2)

        rsv = (close - mim) / (mam - mim) * 100 # 影线
        # 价格高位比例
        ss = rsv / 100 - 0.5
        opt2 = np.sign(ss) * (ss - np.sign(ss) * self.midParam) > 0

        rmum = r1['high'] if ss > 0 else r1['low']

        sign_r1 = np.sign(r1['close']-r1['open'])
        opt20 = (map[5] - map[10]) * (r1['ma5'] - r1['ma10']) < 0 and  (map[5] - map[10]) * ss > 0 #  突破三角区
        opt21 = ss * (close - rmum) > 0 and atrc > 1 and abs(ss) > 0.35  #超过最高最低

        opt22 = (r1['atrc'] > self.backParam) and ss * sign_r1 < 0 # 前一个大ATR

        # 日盘跳空
        opt0 = jump * (close - open0) > 0
        opt1 = (mum - (open0 - jump)) * jump > 0

        ll = [opt20, opt21, opt22, opt2]
        options = sum([2 ** k if ll[k] else 0 for k in range(len(ll))])

        powm1 = np.sign(ss) if opt2 and (opt20) else 0 if opt2 else 0
        powm =  np.sign(jump) if opt0 and opt1 else 3 * np.sign(ss) if opt2 and opt22 else 0

        return pd.Series([jump,  open0, ss,  atr, atrc, ma, powm, powm1, bias, options], index=self.pointColumns)

    def getMax(self, df0, s, e, mode):
        if mode > 0:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'high'].max()
        else:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'low'].min()

    preNode, batchId = None, {}

    def total(self, dfs, period= 21):

        # 计算参数
        df01 = dfs[self.mCodes[0]]
        close = df01["close"]
        df01["datetime"] = df01.index
        df01["time"] = df01["datetime"].apply(lambda x: str(x).split(' ')[1])

        # 日线数据
        self.df_day = df1 = self.Rice.kline(self.mCodes, period='1d', start=self.startDate, end=self.endDate, pre=21)[self.mCodes[0]]
        df1['jump'] = df1['open'] - df1['close'].shift(1)
        df1['atr'] = ta.ATR(df1['high'], df1['low'], df1['close'], timeperiod=14)
        df1['atrc'] = ta.ATR(df1['high'], df1['low'], df1['close'], timeperiod=1) / df1['atr']

        df1['mVol'] = self.iniAmount * self.stopLine / self.BS[self.codes[0]]["contract_multiplier"] / (2 * df1['atr'])

        # 循环 scale
        docs = []
        for conds in self.iterCond():
            uid = self.uidKey % ('_'.join(self.mCodes), str(period),  self.klineType, str(self.bullwidth), conds)

            df1["ma"] = ta.MA(df1["close"], timeperiod=period)
            df1["ma5"] = ta.MA(df1["close"], timeperiod=5)
            df1["ma10"] = ta.MA(df1["close"], timeperiod=10)
            df1["ma20"] = ta.MA(df1["close"], timeperiod=20)

            df0 = copy.deepcopy(df01[(df01['time'] >'09:10:00') & (df01['time'] <'10:15:00')][self.startDate:])

            df2 = df0.apply(lambda row: self.jump(row, df01, df1), axis=1)
            for key in self.pointColumns:  df0[key] = df2[key]

            if self.code in self.csvList:
                file = self.Rice.basePath + '%s_sg.csv' % (uid)
                file1 = self.Rice.basePath + '%s_sg1.csv' % (uid)
                print(uid, '---------------------------- to_cvs', file, df0.columns)
                df0.to_csv(file, index=0, columns= ['datetime', 'time', 'volume', 'close']+self.pointColumns)
                #df1.to_csv(file1, index=1)

            if self.onlyCVS:
                df = df0[df0['powm'] !=0]
                df['code'] = self.code
                df['pid'] = self.ctpCode
                df['uid'] = uid
                df['shift'] = df['atrc']
                df['delta'] = df['midratio']

                df['mode'] = df['powm'].astype(int)
                df['options'] = df['powm1'].astype(int)
                df['createdate'] = df['datetime']
                self.Train.insertAll(df.to_dict(orient='record'))

            if self.onlyCVS: continue

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
                if docs is not None:
                    if self.isPool:
                        conn = Client(self.address, authkey=self.authkey)
                        conn.send([1, docs])
                        conn.close()
                    else:
                        Train = train_future()
                        Train.tablename = self.total_tablename
                        Train.insertAll(docs)
                        #self.Train.insertAll(docs)

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

            #print(self.code, doc['count'], doc['income'])
            return doc
        else:
            return None

    # 核心策略部分
    def stageApply(self, df0, period=0, uid=''):

        isOpen, preDate, prePrice = 0, None, 0
        doc, docs = {}, []
        self.preNode, self.jumpNode, self.jump_index = None, None, 0

        keepUnit=len(df0[df0['trading_date']==df0.ix[0, 'trading_date']])

        #adjusts = [c[1] for c in self.adjustDates]

        for i in range(len(df0)):
            isRun, isstop = False, 0
            powm, close, ma, atr, dd, date,time = (df0.ix[i, key] for key in
                                             "powm,close,ma,atr,datetime,trading_date,time".split(","))

            opt0 = ((close - ma) * powm > 0) if self.isMaUse==1 else True

            if isOpen == 0:
                if powm != 0 and opt0:
                    isRun, isOpen = True, int(powm)

            elif self.preNode is not None:
                 # 调仓

                 pN = self.preNode[0]
                 sign = np.sign(pN['mode'])
                 pmax = self.getMax(df0, pN['createdate'], dd, sign)

                 keep = len(df0[self.preNode[0]['createdate']:dd]) // keepUnit

                 if sign * powm < 0 and keep >= self.keepDays:
                     doc = self.order(df0.iloc[i], 0, uid,  df0, isstop=2)
                     isOpen = 0
                     if doc is not None:
                         docs.append(doc)
                         if opt0: # 反向开仓
                             isRun, isOpen = True, int(powm)

                 elif sign * (pmax - close) > self.dissLine * atr:
                     isRun, isOpen, isstop = True, 0, 4

                 elif date == self.endDate and date != self.ini_endDate:
                     # 批次结束
                     isRun, isOpen, isstop = True, 0, 5

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
        doc1['batchid'] = self.batchId = uuid.uuid1()

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
        if self.iniVolume != 0:
            hands = self.iniVolume

        elif np.isnan(n0['atr']) :

            hands = int(self.iniAmount/10/n0['close']/b0["contract_multiplier"])

        else:
            hands = int(self.iniAmount * self.stopLine / 2 / n0['atr'] / b0["contract_multiplier"])


        v0 = hands * b0["contract_multiplier"]
        # 费率
        fee0 = (hands * b0["ratio"] * 1.1) if b0["ratio"] > 0.5 else ((b0["ratio"] * 1.1) * n0["close"] * v0)

        ps =  n0["close"]
        amount = v0 * ps * float(b0['margin_rate'])

        doc = copy.deepcopy(n0.to_dict())
        doc.update({
            "createdate": n0["datetime"],
            "code": self.codes[0],
            "pid": self.mCode,
            "price": ps,
            "vol": self.preNode[0]["vol"] if self.preNode else v0,
            "hands": self.preNode[0]["hands"] if self.preNode else hands,
            "amount": amount if not self.preNode else self.preNode[0]["amount"],
            "mode": int(mode) if not self.preNode else -self.preNode[0]["mode"],
            "isopen": 0 if mode == 0 else 1,
            "fee": fee0,
            "income": 0,
            "isstop": isstop,
            "options": int(n0['options']) if 'options' in n0 else 0,
            "atr": n0['atr'] if 'atr' in n0 else 0,
            "macd": n0['ma'] if 'ma' in n0 else 0,
            "rel_std": n0["std"] if 'std' in n0 else 0,
            "delta": n0['diss'] if 'diss' in n0 else 0,
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
        "area":0
    }

    if action["kline"] == 1:
        obj = train_future_jump_sg()
        obj.Pool()

    if action["area"] == 1:
        obj = train_future_jump_sg()
        Rice = interface_Rice()
        for t in [1, 2]:
            lists = obj.getList(t)
            df = pd.DataFrame(lists)
            if t==1:
                df['diff'] = df.apply(lambda r: public.timeDiff(r['enddate'], r['startdate'])//86400, axis=1)
            else:
                df['diff'] = df.apply(lambda r: public.dayDiff(r['enddate'], r['startdate']), axis=1)

            file = Rice.basePath + 'area_%s_sg.csv' % (t)
            df.to_csv(file, index=0, columns=['code', 'name', 'startdate', 'enddate', 'diff'])


if __name__ == '__main__':
    main()
