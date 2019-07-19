# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein



 """

from com.base.public import public, logger
import pandas as pd
import talib as ta
from com.object.obj_entity import train_future, future_baseInfo
from com.data.interface_Rice import tick_csv_Rice
import time
from multiprocessing import Pool, Manager
import uuid
import os

import itertools
# 一次生成中间文件
class tick_create():
    # 查询存在tick的code
    basePath = 'E:/stock/csv/eight/' if os.name == 'nt' else "/root/home/stock/csv/"
    lists = [("RU","BU"), ("J","I"), ("SR","I"),("OI","P"),("MA", "ZC")]
    cvs_start = '2018-08-01 09:00:00'
    cvs_end = '2018-08-31 15:00:00'

    def __init__(self):
        self.dd = ['1901']
        self.columns = ['a1', 'a1_v', 'b1', 'b1_v', 'close', 'code', 'datetime', 'high',
                   'last', 'low', 'n_a1', 'n_a1_v', 'n_b1', 'n_b1_v', 'n_code', 'n_datetime', 'n_last', 'open', 'isnew']

        self.columns1 = ['datetime', 'a1',  'b1',  'n_a1', 'n_b1', 'close', 'high', 'low', 'isnew', 'isdeal']

    def tickList(self, long = 8):
        Base = future_baseInfo()
        l = []
        files = os.listdir(self.basePath)
        ticks = [file[0:-8] for file in files]

        for c in Base.all(vol=400):
            if c in ticks:
                l.append(c)
        return l


    def exitTicks(self):
        files = os.listdir(self.basePath)
        return [file[0:-4] for file in files if file.find('_')>-1 and len(file)<10]


    def PoolReduce(self):
        pool = Pool(processes=4)
        lists = self.tickList()
        for rs in lists:
            #self.reduce(rs, None)
            try:
                pool.apply_async(self.reduce,  (rs, None))
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

    def Pool(self, ):
        pool = Pool(processes=4)
        shareDict = Manager().dict({})
        lists = self.tickList()
        # 已生成

        exists = self.exitTicks()

        for rs in list(itertools.combinations(lists, 2)):
            #self.start(rs, shareDict)
            try:
                pool.apply_async(self.start, (rs, shareDict))
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

    # 合并
    kk = 0
    def start(self, codes, share):
        print("子进程:", codes)
        self.time0 = time0 = time.time()
        self.Base = future_baseInfo()
        self.Rice = tick_csv_Rice()
        self.Rice.shareDict = share

        self.Rice.basePath = self.basePath
        self.Rice.cvs_start = self.cvs_start
        self.Rice.cvs_end = self.cvs_end
        self.Rice.dd = self.dd

        self.new_docs = []
        self.codes = codes
        self.baseInfo = {}
        self.filePath = self.basePath + "%s.csv" % "_".join(codes)

        # 查询获得配置 - 费率和每手单量
        for doc in self.Base.getInfo(codes):
            self.baseInfo[doc["code"]] = doc

        cs0, cs1 = self.baseInfo[codes[0]], self.baseInfo[codes[1]]
        if cs0["nightEnd"] != cs1["nightEnd"]:  return None

        # 设置结束时间,用于数据清洗
        try:
            res = self.Rice.curKlineByCsv(codes, period=1, callback=self.new_tick)

        except Exception as e:
            print('error', codes, len(self.new_docs), e )
            res = True

        if res and len(self.new_docs) > 0 :
            # 保存文件
            df = pd.DataFrame(self.new_docs)
            # 数据整理
            df = self.detail(df)

            print(public.getDatetime(), " 已保存", codes, "耗时:", time.time()-time0)

            df.to_csv(self.filePath, columns=self.columns1, index=0)
            self.new_docs= []
            return True

    indexN = 0
    def new_tick(self, tick):
        doc = {}
        doc.update(tick)
        self.new_docs.append(doc)

        self.indexN+=1
        if self.indexN % 10000 ==0:
            print(self.codes, self.indexN, time.time()-self.time0)

    # 数据对比, kline-tick
    def PoolUpdate(self):
        pool = Pool(processes=4)
        for rs in self.lists:
            try:
                pool.apply_async(self.compare,  (rs, None))
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

    def sel(self, a1,  b1):
        return min(a1, b1)

        # 删除不用的列和行

    #iniTime = ['2018-04-10 21:00:00', '2018-05-03 21:00:00']

    #
    def reduce(self, c, share):
        Rice = tick_csv_Rice()
        Base = future_baseInfo()
        Rice.codes = [c]
        s = self.dd
        file = self.basePath + "%s%s.csv" % (c, s[0])

        df = pd.read_csv(file)

        # 设置结束时间,用于数据清洗
        for doc in Base.getInfo([c]):
             ne = doc["nightEnd"]

        Rice.setTimeArea(ne)
        df = Rice.clean(df)

        len0 = len(df)
        df = df[(df['datetime'] > self.cvs_start) & (df['datetime'] < self.cvs_end)]
        print(c,len0, len(df))

        df.to_csv(file, columns= Rice.cvs_columns, index=0)


    def dfRound(self,df, unit, columns):
        decimals = pd.Series([unit for i in range(len(columns))], index=columns)
        return df.round(decimals)

    def detail(self, df):

        df["isnew"] = df["datetime"].apply(lambda x: 1 if self.Rice.isNew(x, inner=60, limit=5) else 0)
        df["isdeal"] = df["datetime"].apply(lambda x: 1 if self.Rice.isNew(x, inner=5, limit=0) else 0)

        decimals = pd.Series([5, 5, 5], index=['close', 'low', 'high'])
        df = df.round(decimals)
        return df


    def compare(self, codes, share):
        # ticks
        columns =['datetime','price_t','p_l_t','p_h_t']

        self.Rice = tick_csv_Rice()
        ticks = self.Rice.getTicks(codes)
        ticks = ticks[ticks['isnew'] == 1]

        cvs_start = ticks["datetime"].values[0]
        cvs_end = ticks["datetime"].values[-1]
        #
        key = "_".join(codes)

        ticks["datetime"] = ticks["datetime"].apply(lambda x : x[0:-4] if x.find('.')> 0 else x)
        ticks["datetime"] = ticks["datetime"].str.replace(':01$|:02$|:03$|:04$',':00')

        ticks["price_t"] = ticks["close"]
        ticks["p_l_t"] = ticks["a1"] / ticks["n_b1"]
        ticks["p_h_t"] = ticks["b1"] / ticks["n_a1"]
        ticks = self.dfRound(ticks,5,['close','p_l','p_h'])

        ticks.to_csv(self.basePath + "tick_%s.csv" % key,  columns=columns, index=0)

        Base = future_baseInfo()
        mCodes = [ c+'88' for c in codes]

        (b0, b1) =(doc['tick_size'] for doc in Base.getInfo(codes))


        print(b0,b1)

        dfs = self.Rice.kline(mCodes, start=cvs_start, end=cvs_end)
        df0, df1= dfs[mCodes[0]], dfs[mCodes[1]]

        df0 = df0[(df0.index >= cvs_start) & (df0.index <= cvs_end)]

        df0["datetime"] = df0.index
        df0["datetime"] = df0["datetime"].apply(str)
        c0, c1 = df0["close"], df1["close"]

        df0["price"] = c0 / c1
        shift = ((c0+b0)/(c1-b1)-(c0-b0)/(c1+b1))/2
        df0["p_l"] = c0/c1 + shift
        df0["p_h"] = c0/c1 - shift
        df0["p_l_1"] = (c0+b0/2)/(c1-b1/2)
        df0["p_h_1"] = (c0-b0/2)/(c1+b1/2)

        df0 = self.dfRound(df0, 5, ['price', 'p_l', 'p_h', 'p_l_1', 'p_h_1'])

        df0 =pd.merge(df0, ticks, on='datetime', how='left')

        columns = ['datetime_t', 'datetime', 'price_t', 'price',
                   'p_l_t', 'p_l' , 'p_l_1', 'p_h_t', 'p_h' , 'p_h_1']
        res = {}
        res['p_l'] = df0['p_l'].mean() - df0['p_l_t'].mean()
        res['p_l_1'] = df0['p_l_1'].mean() - df0['p_l_t'].mean()
        res['p_h'] = df0['p_h'].mean() - df0['p_h_t'].mean()
        res['p_h_1'] = df0['p_h_1'].mean() - df0['p_h_t'].mean()

        print(res)
        df0.to_csv(self.basePath + "total_%s.csv" % key, columns=columns, index=0)


def main():
    action = {
        "tick": 1,
        "reduce": 0,
    }

    if action["reduce"] == 1:
        obj = tick_create()
        obj.PoolReduce()

    if action["tick"] == 1:
        obj = tick_create()
        obj.Pool()


if __name__ == '__main__':
    main()
