# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Created on Sat Apr  1 14:35:13 2017

@author: admin

      sh50自动记录

"""

# from com.data.model_warrant import warModel
from com.base.public import public, config_ini, logger
from com.model.model_sh50 import model_sh50
from com.data.data_base import Stock

import time
import sys
import os

if __name__ == '__main__':
    inter = int(config_ini.get("interval", session="SH50"))
    obj = model_sh50()
    ST = Stock()
    print(inter)
    while 1:
        # 非交易日，退出
        time0 = time.time()
        if not ST.isTradeDate():
            print("非交易日")
            break

        isT = ST.isTradeTime("SH")
        if isT[0]:
            # 交易时间记录
            obj.record()
            #logger.info("sh50记录并计算，耗时: %s" % str(time.time()-time0))

        elif not isT[1]:
            # 收市则停止记录
            print("over")
            break

        time.sleep(inter)

    #
    try:
        os.system("ps -ef | grep sh50 | awk '{print $2}' | xargs kill -9")
    except:
        pass

    sys.exit()

