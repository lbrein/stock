# -*- coding: utf-8 -*-
"""
Created on 2019-3-11
@author: lbrein

    计算单品种的 各种kline 下的 std 均值， 评估合理的 k线区间匹配

"""

from com.base.public import public, logger
import pandas as pd
import talib as ta
import numpy as np

from com.ctp.interface_pyctp import BaseInfo
from com.object.obj_entity import future_baseInfo
from com.object.mon_entity import mon_tick
from com.data.interface_Rice import interface_Rice
import time
from multiprocessing import Process,Value,Lock
from multiprocessing.managers import BaseManager
from multiprocessing import Pool
from com.ctp.interface_pyctp import interface_pyctp
# 回归方法
class MyManager(BaseManager):
      pass

MyManager.register('interface_Rice', interface_Rice)
MyManager.register('future_baseInfo', future_baseInfo)
MyManager.register('BaseInfo', BaseInfo)
MyManager.register('interface_pyctp', interface_pyctp)

def Manager2():
    m = MyManager()
    m.start()
    return m

# 回归方法
class future_tickStage(object):
    """

    """
    def __init__(self):
        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.quickPeriodList = ['2m', '3m', '5m']
        self.periodList = ['15m', '30m', '60m']

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    def getStart(self, nt, ktype):
        key = nt + ' ' + ktype
        if nt in self.preDict:
            start = self.preDict[key]
        else:
            start = self.preDict[key] = self.Rice.getPreDate(period=ktype, num=self.count)
        return start

    def pool(self):
        manager = Manager2()
        Rice = manager.interface_Rice()
        Base = manager.future_baseInfo()

        Rice.setIndexList(self.indexCodeList)
        codes = Base.getUsedMap(hasIndex=False, isquick=True)

        mCodes = Rice.getMain(codes)
        #print(mCodes)

        pool = Pool(processes=4)
        i = 0
        for c in mCodes:
            #print('main', c)
            self.start(i, Rice, Base, c)
            #pool.apply_async(self.start, (i, Rice, Base, c))
            i += 1
            break

        pool.close()
        pool.join()

    def start(self,pic, Rice, Base, code):
        print(pic, code)
        self.code = code

        Rice = interface_Rice()
        time0 = time.time()
        k = 0
        while True:
            df = Rice.getTicks(code)
            self.calc(df)
            k += 1
            print(k, len(df), time.time() - time0)
            if k > 5: break

        print(time.time() - time0)


    def getAvg(self, row, df):
        index = row['No']

        if index == 0:
            return row['last']

        df0 = df[df['No'] <= index]

        s = df0['last'] * df0['volume']
        v = s.sum() / df0['volume'].sum()
        return v

    def calc(self, df0):
        # 计算均值
        close = df0["last"]
        #df0["datetime"] = df0.index
        cc = close.reset_index(drop=True)
        df0["No"] = cc.index
        #df0['avg'] = df0.apply(lambda row: self.getAvg(row, df0), axis=1)
        index = len(df0)-1

        df1 = df0[df0['No']<=index]
        df2 = df1[((index-100) < df1['No']) & (df1['No']< index)]

        s = df1['last'] * df1['volume']
        avg = s.sum() / df1['volume'].sum()
        last = df1['last'].values[-1]

        avgc = df2['volume'].mean()

        if last > avg:
           max = df2['last'].max()
        else:
           max = df2['last'].min()

        doc = df0.iloc[index].to_dict()
        doc.update({
            "avg": avg,
            "max": max,
            "avgc":avgc
        })
        print(doc)
        return doc
        #df['No'] = df.re




        pass


def main():
    obj = future_tickStage()
    obj.create()

if __name__ == '__main__':
    main()



