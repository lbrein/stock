# -*- coding: utf-8 -*-
"""
Created on  2018-01-04 
@author:
      T+0

-
"""
from com.base.stat_fun import per
import statsmodels.api as sm  # 协整
from com.base.public import public
from com.data.interface_Rice import interface_Rice
import numpy as np
import talib as ta
import time
from com.object.obj_entity import stock_baseInfo, stock_record_t0, stock_uniform
from multiprocessing import Pool
import itertools

class data_stock_Rice(object):
    #
    filepath = "E:\lbrein\python\stock/file/"
    times = 15

    def __init__(self):
        self.isEmptyUse = True

        self.period = '1m'

        self.klineTypeList = ['1d', '5m', '15m', '60m']

        self.startDate = '2016-03-01'
        self.endDate = '2019-06-12'
        self.timeperiod = 14

    def update_stockBase(self):
        # 更新stockBase
        Base = stock_baseInfo()
        Rice = interface_Rice()

        codes = Base.getAllCodes()
        #Base.empty()

        df0 = Rice.allCodes(type='CS')
        df = df0[~df0['order_book_id'].isin(codes)]
        #print(df)

        df['code'] = df['order_book_id'].apply(lambda x: x[:x.find('.')])
        s = Rice.index_compose('000980.XSHG')
        df['is50'] = df['order_book_id'].apply(lambda x: 1 if x in s else 0)
        df['isST'] = df['symbol'].apply(lambda x: 1 if x.find('*ST') > -1 else 0)

        print(len(df))
        Base.insertAll(df.to_dict('records'))

    def uniformEmpty(self):
        if self.isEmptyUse:
            Record = stock_uniform()
            Record.empty()

    def pool(self):
        pool = Pool(processes=5)
        # T+0历史记录
        T0 = stock_record_t0()
        lists = T0.getCodes()

        self.uniformEmpty()

        # base信息
        BI = stock_baseInfo()
        time0 = time.time()
        indList = []

        for code in lists:
            # 查询行业代码及所有成员股
            ind, codes = BI.getSameIndustry(code)
            # 检查行业是否存在
            if ind in indList: continue
            indList.append(ind)
            # 组合遍历
            print(ind, codes)
            #break
                # 遍历
            for kline in self.klineTypeList:
                 #self.start(rs,  time0, kline)
                 try:
                     pool.apply_async(self.start, (codes, code, time0, kline))
                     pass
                 except Exception as e:
                        print(e)
                        continue

        pool.close()
        pool.join()
    cindex = 0

    def start(self, codes, main, time0, kt):
        print("子进程启动:", self.cindex, codes, kt, time.time() - time0)
        self.klineType = kt
        self.Record = stock_uniform()
        # 主力合约
        self.Rice = interface_Rice()

        # 查询获得配置 - 费率和每手单量

        if kt[-1] == 'm':
            self.startDate = public.getDate(diff=-200)
        else:
            self.startDate = public.getDate(diff=-1200)

        # 查询获得N分钟K线
        dfs = self.Rice.kline(codes, period=self.klineType, start=self.startDate, end=self.endDate, pre=60)
        for rs in list(itertools.combinations(codes, 2)):
            if main not in rs: continue
            self.codes = rs
            doc = self.total(dfs, kt)
            if doc is not None:
                self.Record.insert(doc)

    def total(self, dfs, kt):
        df0, df1 = dfs[self.codes[0]], dfs[self.codes[1]]

        df0 = df0.dropna(axis=0)
        df1 = df1.dropna(axis=0)

        if len(df0) != len(df1):
            # print('------------------change', kt, len(df0), len(df1))
            if kt[-1] == 'm':
                if len(df0) > len(df1):
                    df0 = df0[df0.index.isin(df1.index)]
                    df1 = df1[df1.index.isin(df0.index)]
                else:
                    df1 = df1[df1.index.isin(df0.index)]
                    df0 = df0[df0.index.isin(df1.index)]

            elif len(df0) > len(df1):
                df0 = df0.iloc[(len(df0) - len(df1)):]
                # print('------------------change', len(df0), len(df1))
            else:
                df1 = df0.iloc[(len(df0) - len(df1)):]

        # 涨跌一致性
        diff0 = df0['close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        diff1 = df1['close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

        df0 = df0.dropna(axis=0)
        df1 = df1.dropna(axis=0)
        ss = ((diff0 + diff1)).apply(lambda x: 1 if x != 0 else 0)

        delta = ss.sum() / len(ss)


        # 交易量差距
        volrate = df0['volume'].mean()/df1['volume'].mean()

        # 波动比
        s0 = ta.MA(df0['close'], timeperiod=15).mean()
        sd0 = ta.STDDEV(df0['close'], timeperiod=15, nbdev=1).mean()

        s1 = ta.MA(df1['close'], timeperiod=15).mean()
        sd1 = ta.STDDEV(df1['close'], timeperiod=15, nbdev=1).mean()
        std = sd0 * s1 / s0 / sd1

        # 跌涨标准差
        dd0 = ta.STDDEV(df0['close'].diff(), timeperiod=15, nbdev=1).mean()
        dd1 = ta.STDDEV(df1['close'].diff(), timeperiod=15, nbdev=1).mean()
        diffstd = dd0 * s1 / s0 / dd1

        #
        coint = sm.tsa.stattools.coint(df0["close"], df1["close"])
        # x相关性
        relative = per(df0["close"], df1["close"])
        doc = {
            "code": self.codes[0],
            "code1": self.codes[1],
            "kline":self.klineType,
            "relative": relative,
            "samecount": len(df0),
            "samerate": delta,
            "diffstd":diffstd,
            "coint_1": 0 if np.isnan(coint[0]) else coint[0],
            "coint": 0 if np.isnan(coint[1]) else coint[1],
            "std": std,
            "vol": volrate,
            "type": "stock"
        }
        #print(doc)
        # 按时间截取并调整
        #print('kline load:', kt, self.codes, len(df0), len(df1))
        return doc

def main():
    action = {
        "csv": 0,
        "update": 0,
        "param": 1
    }

    obj = data_stock_Rice()
    if action["csv"] == 1:
       obj.csv()

    if action["update"] == 1:
       obj.update_stockBase()

    if action["param"] == 1:
       obj.pool()

if __name__ == '__main__':
    main()
    #test()
