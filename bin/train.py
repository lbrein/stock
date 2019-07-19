# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Created on Sat Apr  1 14:35:13 2017



"""
# 配对交易
from com.train.train_future_klineExpect import train_future_klineExpect
# 单品交易历史策略
from com.train.train_future_singleExpect0 import train_future_singleExpect0
# 单品交易最新策略
from com.train.train_future_singleExpect import train_future_singleExpect

from com.base.public import public

if __name__ == '__main__':
    cmdKeys = "w:"
    fw = public.getCmdParam("w:", ("-w", 5))
    if fw == 1:
        obj = train_future_klineExpect()
        # tick模拟
        obj.isSimTickUse = True
        #obj.klineTypeList = ['5m', '15m', '30m']
        obj.Pool()

    elif fw == 2:
        obj = train_future_klineExpect()
        obj.isSimTickUse = False
        #obj.klineTypeList = ['1m', '5m', '15m', '30m']
        obj.Pool()

    elif fw ==3 :
        obj = train_future_singleExpect()
        obj.isSimTickUse = True
        obj.Pool()

    elif fw ==4 :
        obj = train_future_singleExpect()
        obj.isSimTickUse = False
        obj.Pool()


    elif fw ==4 :
        obj = train_future_singleExpect()
        obj.isSimTickUse = False
        obj.Pool()


    elif fw ==5 :
        obj = train_future_singleExpect0()
        obj.isSimTickUse = False
        obj.Pool()

