# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Created on Sat Apr  1 14:35:13 2017

@author: admin

      手动通知ETF买单

"""

# from com.data.model_warrant import warModel
from com.base.public import public, config_ini, logger
from com.model.model_etf4 import etf4_compare
import time
import sys

if __name__ == '__main__':
    inter = int(config_ini.get("interval", session="ETF"))
    obj = etf4_compare()
    
    while 1:
        try:
            obj.FT.reconnect()
        except:
            sys.exit()
            break

        # 非交易日，退出
        if not obj.FT.isTradeDate("SH"):
            print("非交易日")
            exit()
            
        d, t = public.getDate(), public.getDatetime()
        start, end = d + " 14:30:00", d + " 15:00:00"
        
        # 检查交易时间
        if not obj.FT.isTradeDateTime('SH'):
            # 超过收盘时间则终止程序
            if t > end:
                sys.exit()
                break

            print("link connect")
            delay = 10
        
        elif t > start and t < end:
            # 大于2点45分，则每30秒运行一次直到结束
            obj.inform(1)
            delay = 30 
            logger.info("ETF - inform -1 执行成功 delay: %s" % str(delay))

        else:
            # 小于2点30分，则每15分钟运行一次
            obj.inform(0)
            delay = inter * 60 -0.5
            logger.info("ETF - inform -0 执行成功 delay: %s" % str(delay))

        # 关闭连接
        obj.FT.close()
        time.sleep(delay)
  
    sys.exit()    
          