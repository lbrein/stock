# -*- coding: utf-8 -*-
"""
Created on  2019-6-21
@author:
      @ options get

-
"""

from com.base.public import public
import pandas as pd
from com.data.interface_Rice import interface_Rice
import numpy as np
import talib as ta
import copy
import uuid
from com.object.obj_entity import stock_orderForm, stock_baseInfo, stock_record_t0, stock_t0_param
from multiprocessing import Pool


class data_Option_Rice(object):
    #
    filepath = "E:\lbrein\python\stock/file/"
    times = 15

    def __init__(self):
        self.period = '1d'
        self.startDate = '2015-02-11'
        self.endDate = '2019-06-12'
        self.timeperiod = 14

    def allCodes(self, mode = 'C'):
        Rice = interface_Rice()
        df = Rice.allCodes(type='Option')
        c = df[df['underlying_order_book_id']=='510050.XSHG']['order_book_id'].values[-1]
        codeList = {}
        for i in range(10000001, int(c)+1):
            s = Rice.detail(str(i))
            if s is not None and s.option_type == mode:
                codeList[s.order_book_id] = s.maturity_date
        return codeList

    # 分段布林策略
    def empty(self):
        Record = stock_t0_param()
        #Record.tablename = self.recordTableName
        Record.empty()

    # 查询参数写入到param表
    def pool(self):
        #self.empty()

        for m in ['C']:
            codeList = self.allCodes(mode=m)
            self.start(codeList, m)


    def start(self, codeList, mode):
        self.Rice = interface_Rice()

        self.codes= cs = [c for c in codeList.keys()]
        res = self.Rice.kline(self.codes, period=self.period, start=self.startDate, end=self.endDate, pre=1, type='Option')

        df = res[cs[0]]
        df['dd'] = df.index
        ed = codeList[cs[0]]

        df.loc[:, 'monthdiff'] = df['dd'].apply(lambda x: getMonthDay(str(x), ed))
        print(df)




def main():
    action = {
        "codes": 0,
        "pool": 1,
        "param": 0
    }

    obj = data_Option_Rice()
    if action["codes"] == 1:
       obj.allCodes()

    if action["pool"] == 1:
       obj.pool()

if __name__ == '__main__':
    main()
    #test()
