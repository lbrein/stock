# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Created on Sat Apr  1 14:35:13 2017

@author: admin

      sh50自动记录

"""
import sys
from com.base.public import public
from com.model.model_sh50_MA import model_sh50_ma
from com.model.model_sh50_Bull import model_sh50_bull
from  com.data.data_interface import sinaInterface

if __name__ == '__main__':
    sys.path.append("/root/home/project/stock")
    fw = public.getCmdParam("w:", ("-w", 1))
    if fw == 1:
        obj = model_sh50_ma()
        obj.start()

    elif fw == 2:
        obj = model_sh50_bull()
        obj.start()

    elif fw ==5 :
        obj = sinaInterface()
        obj.baseInfo()
