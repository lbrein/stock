# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Created on Sat Apr  1 14:35:13 2017

@author: admin

      sh50自动记录

"""

from com.model.model_future_detect_1m import model_future_detect_1m
from com.model.model_future_detect_multi_ctp import model_future_detect_multi
#from com.model.model_future_detect_single_ctp import model_future_detect_single
from com.model.model_future_fellow5_v1 import model_future_fellow5_v1
from com.model.model_future_detect_singlev20 import model_future_single_0
from com.model.model_future_zhao_ctp import model_future_zhao
from com.model.model_future_zhao55_ctp import model_future_zhao55

from com.model.model_future_zhao_ctp_v1 import model_future_zhao_v1
from com.model.model_future_zhao55_ctp_v1 import model_future_zhao55_v1


from com.data.interface_Rice import updateNightEnd, interface_Rice
from com.base.public import public

if __name__ == '__main__':
    fw = public.getCmdParam("w:", ("-w", 7))
    if fw == 1:
        obj = model_future_detect_1m()
        obj.isCTPUse = False
        obj.pool()

    elif fw == 2:
        obj = model_future_detect_multi()
        obj.pool()

    elif fw == 3:
        obj = model_future_fellow5_v1()
        obj.isCTPUse = False
        obj.isDeadCheck = False
        obj.pool()

    elif fw == 4:
        obj = model_future_single_0()
        obj.isCTPUse = True
        obj.pool()

    elif fw == 6:
        obj = model_future_zhao()
        obj.pool()

    elif fw == 7:
        obj = model_future_zhao55()
        obj.pool()

    #
    elif fw == 8:
        obj = model_future_zhao_v1()
        obj.pool()

    elif fw == 9:
        obj = model_future_zhao55_v1()
        obj.pool()

    elif fw ==5 :
        updateNightEnd()
        obj = interface_Rice()
        obj.baseUpdate()


