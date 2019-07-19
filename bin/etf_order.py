# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Created on Sat Apr  1 14:35:13 2017

@author: admin

      sh50自动记录

"""

# from com.data.model_warrant import warModel
from com.base.public import logger
from com.model.etf_stage1 import ETF_stage
from com.data.data_base import Stock

import time
import sys

if __name__ == '__main__':
    inter = 59
    obj = ETF_stage()
    ST = Stock()
    isEnd = False

    while not isEnd:
        # 非交易日，退出
        time0 = time.time()
        if not ST.isTradeDate():
            print("非交易日")
            isEnd = True
            break

        isT = ST.isTradeTime("SH")
        if isT[0]:
            # 交易时间记录
            obj.create(stage=0)
            logger.info("基金策略运行，耗时: %s" % str(time.time()-time0))

        elif not isT[1]:
            # 收市则停止记录
            print("over")
            isEnd = True
            break

        time.sleep(inter)

    sys.exit()
