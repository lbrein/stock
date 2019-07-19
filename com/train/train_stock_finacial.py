# -*- coding: utf-8 -*-
"""
Created on 12 19 -2018
@author: lbrein

      zhao 基金长线策略： bias 》 1 或bias 《 -1
      日线，每分钟持续跟踪

"""

from com.base.public import public, logger
import numpy as np
import pandas as pd
import talib as ta
import uuid
from com.data.interface_Rice import interface_Rice
from com.object.obj_entity import stock_orderForm, stock_baseInfo, stock_PE_year
from com.object.mon_entity import mon_trainOrder
from multiprocessing import Pool, Manager
import time
import copy

# 选股
class train_stock_finacial(object):

    def __init__(self):
        self.period = '1d'
        self.pageCount = 50
        self.isEmptyUse = False

        self.iniAmount = 100000
        self.shift = 0.002
        self.ratio = 0.0006


        self.PE_upRatio = 0.5
        self.incomeRatioLine = 15

        self.saveMongo = False
        self.methodName = 'stock5'

        self.saveMongo = False
        self.startDate = public.getDate(diff=-200)  # 60天数据回测
        self.endDate = public.getDate(diff=0)

        self.columns = ['code', 'symbol', 'date', 'pe_ratio', 'total_equity_and_liabilities', 'total_assets', 'total_liabilities','rate']

    # 财报输出
    def pool(self, Func = None):
        pool = Pool(processes=5)
        #self.empty()
        Base = stock_baseInfo()
        lists = Base.getCodes(isBound=0)

        self.dict = Base.getDict(isBound=0)
        self.Rice = interface_Rice()

        df = []

        for k in range(0, len(lists), self.pageCount):
            codes = lists[k:k+self.pageCount]
            docs = self.start(codes, int(k/self.pageCount+1))
            df += docs
            #break
            try:
                #pool.apply_async(self.start, (codes, int(k/self.pageCount+1)))
                pass
            except Exception as e:
                print(e)
                continue

        if len(docs)>0:
            df0 = pd.DataFrame(docs, columns=self.columns)
            df0 = df0.sort_values(['code', 'date'], ascending=[True,False])
            file = self.Rice.basePath + 'stock_finacial_pre.csv'
            print('---------------------------- to_cvs', file, df0.columns)
            df0.to_csv(file, index=0)

        pool.close()
        pool.join()

    # 分段布林策略
    def start(self, codes, n):
       print(n)
       docs = []
       dfs = self.Rice.get_fundamentals(codes, start_date='2019-01-01')
       for c in codes:
           if c not in dfs.keys():continue
           df = dfs[c]
           df['code'] = c
           df['symbol'] = self.dict[c]
           df['date'] = df.index
           df['rate'] = df['total_liabilities']/df['total_assets']
           ds = df.to_dict(orient='records')
           docs += ds
       #print(df.columns)
       return docs


    keyCode = '600256.XSHG'
    # 净资产收益率模型
    def pool_5(self, Func = None):
        pool = Pool(processes=5)
        Base = stock_baseInfo()
        lists = Base.getCodes(isBound=0)
        Record = stock_orderForm()

        print('total', len(lists))
        if self.isEmptyUse:
            Record.empty()

        if Func is None:
            Func= self.start_5

        for k in range(0, len(lists), self.pageCount):
            codes = lists[k:k + self.pageCount]
            #if self.keyCode not in codes: continue
            #print(codes)
            #Func(codes, int(k/self.pageCount+1))
            #break
            try:
                pool.apply_async(Func, (codes, int(k/self.pageCount+1)))
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

        # 分段布林策略

    def stat_pe(self, codes, n):
        print(n)
        self.Rice = interface_Rice()
        self.Record = stock_PE_year()

        df = self.Rice.get_factor(codes, start='2000-01-01', end=public.getDate())

        df['date'] = df.index
        #df['date'] = df['date'].apply(lambda x: str(x))
        df['year'] = df['date'].apply(lambda x: str(x)[:4])
        docs = []

        for y in range(2000, 2020):
           df0 = df[df['year'] == str(y)]
           for c in codes:
              doc = {"code": c, 'year': y, 'pe_ratio': round(df0[c].mean(), 2)}
              #print(doc)
              docs.append(doc)

        if len(docs)>0:
            self.Record.insertAll(docs)

        #print(docs)

    def start_5(self, codes, n):
        time0 = time.time()
        self.Rice = interface_Rice()
        self.Record = stock_orderForm()

        Base = stock_baseInfo()
        self.dict = Base.getDict(isBound=0)
        #l =  [str(c) for c in self.Rice.getValidDate1(start='1999-01-01', end=public.getDate())]
        #self.validDates= pd.Series(l,name='date')
        #print(self.validDates)

        dfs = self.Rice.get_financials(codes, years=20)
        for c in codes:
            if c not in dfs.keys(): continue
            self.uid = '%s_stock5' % (c)
            df = dfs[c]
            df = df.iloc[::-1]
            df.loc[:, 'code'] = c
            #df['code'] = c
            df['date'] = df['announce_date'].apply(lambda x: public.parseTime(str(x),format='%Y%m%d',style='%Y-%m-%d') if not np.isnan(x) else '')

            df['powm'] = df['adjusted_return_on_equity_diluted'].apply(lambda x: 1 if x >= self.incomeRatioLine else 0)

            df['sum'] = ta.SUM(df['powm'], timeperiod=5)

            if self.keyCode == c :
                print(df)

            self.saveStage(df)

        print(n , 'finished time:', time.time()-time0)

    def saveStage(self, df2):
        self.preNode = []
        period = 4
        self.records = []

        for i in range(period, len(df2)):
            sum, date = (df2.ix[i, key] for key in "sum,date".split(","))
            isBuy, isRun = 0, False

            pN = self.preNode
            # 连续5年满足条件
            if sum == 5:
                isBuy, isRun, mode = 1, True, 1

            elif len(pN) > 0 and sum < 5:
                isBuy, isRun, mode = -1, True, 1

            if isRun:
                # print(i, isBuy, pos, vol)
                self.order(df2.iloc[i], isBuy, 1)

        # 结束时用当前日计算股价
        if len(self.preNode) > 0:
            last = copy.deepcopy(df2.iloc[-1])
            last['date'] = public.getDate()
            self.order(last, -1, 1)

        # print(self.uid, len(self.records))
        # 保存明细
        if len(self.records) > 0:
            #print('----', len(self.records))
            self.Record.insertAll(self.records)

    def getVday(self, date):
        df = self.Rice.getValidDate1(date, public.getDate(diff=15, start=date))
        return str(df[0])

    def order(self, n0, isBuy, mode):
        vol, fee, amount, income, p0 = 0, 0, 0, 0, 0
        code = n0['code']

        d = public.getDate(diff=1, start=str(n0["date"]))
        vd = self.getVday(d)
        d1, price = self.Rice.getOpen(code, vd, type='stock')
        #print(vd, d1)
        if price is None:
            print(code, vd, 'None')
            return

        if isBuy > 0:
            if len(self.preNode)==0:
                self.batchid = uuid.uuid1()

            p0 = price * (1 + self.shift)
            vol = int(self.iniAmount/ p0 /100) * 100
            amount = vol * p0
            fee = vol * p0 * self.ratio
            income = -fee

        elif isBuy < 0:
            p0 = price * (1 - self.shift)
            vol, amount, fee, income = 0,0,0,0
            if len(self.preNode)>0:
                for pN in self.preNode:
                    vol += pN['vol']
                    amount += pN['vol'] * p0
                    fee += pN['vol'] * p0 * self.ratio

                    # 计算总的Income
                    income += pN['vol'] * p0 - pN['amount'] - pN['vol'] * p0 * self.ratio

        doc = {
            "code": n0['code'],
            "name": self.dict[n0['code']],
            "createdate": vd,
            'reportdate': d1,
            "price": p0,
            "vol": vol,
            "mode": int(mode),
            "isBuy": int(isBuy),
            "fee": fee,
            "amount": amount,
            "income": income,
            "method": self.methodName,
            "batchid": self.batchid,
            "uid": self.uid
        }

        self.records.append(doc)
        # 设置上一个记录
        if isBuy > 0:
            self.preNode.append(doc)
        else:
            self.preNode = []

        return True

def test():
        Rice = interface_Rice()
        d = public.getDate(diff=1, start='2014-05-26')
        p = Rice.getOpen('002120.XSHE', d)
        print(p)

def main():
    actionMap = {
        "start": 0,  #
        "stock5": 1,
        "pe": 0,

    }
    obj = train_stock_finacial()

    if actionMap["start"] == 1:
        obj.pool()

    if actionMap["stock5"] == 1:
        obj.pool_5()

    if actionMap["pe"] == 1:
        obj.pool_5(Func=obj.stat_pe)


if __name__ == '__main__':
        main()
        #test()