# -*- coding: utf-8 -*-
"""
Created on 2019-4-4
@author: lbrein

    期限套利回测统计

"""
from multiprocessing import Process,Value,Lock
from multiprocessing.managers import BaseManager
from multiprocessing import Pool, Manager

from com.base.public import public, logger
import pandas as pd
import talib as ta
import numpy as np

from com.ctp.interface_pyctp import BaseInfo
from com.object.obj_entity import future_baseInfo
from com.object.mon_entity import mon_tick
from com.data.interface_Rice import interface_Rice
from com.ctp.interface_pyctp import interface_pyctp
import time

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

class future_period_arbi(object):
    """

    """
    def __init__(self):
        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.quickPeriodList = ['2m', '3m', '5m']
        self.periodList = ['15m', '30m', '60m']
        self.startdate = '2016-01-01'

    def create(self):
        Rice = interface_Rice()
        Base = future_baseInfo()
        startdate= '2016-01-01'
        # 选用的
        used = Base.getUsedMap()

        df = Rice.allHisFuture()
        u = df["underlying_symbol"].unique()
        for c in u:
            if not c in used: continue
            sdf = df[(df["underlying_symbol"]==c) & (df["listed_date"]> '2015-01-01')].sort_values(by='listed_date').reset_index(drop=True)
            codes = sdf.loc[:, 'order_book_id']

            dfs = Rice.kline(codes, period='1d', start= startdate)
            print(sdf)
            mcodes = Rice.getAllMain(c, start=startdate)
            print(pd.DataFrame(mcodes))
            break

    def compare(self, dfs):
        pass

    def sub(self, pid, Rice, Base, code):
        print(pid, code)
        df =Rice.getAllMain(code, start=self.startdate)
        print(df)
        m = Base.att(code,'tick_size')
        print(code,m)


    def testProcess(self):
        manager = Manager2()
        Rice = manager.interface_Rice()
        #Base = manager.BaseInfo([])
        Base = BaseInfo([])
        CTP = manager.interface_pyctp(use=True, baseInfo=Base,  userkey='fz')


        pool = Pool(processes=6)
        df = Rice.allHisFuture()
        u = df["underlying_symbol"].unique()
        i=0
        for c in u:
           print('main',c)
           pool.apply_async(self.sub, (i, Rice, CTP, c))
           i+=1
        pool.close()
        pool.join()

def main():
    obj = future_period_arbi()
    obj.testProcess()

if __name__ == '__main__':
    main()



