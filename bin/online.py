# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Created on Sat Apr  1 14:35:13 2017

@author: admin
     
     窝轮在线检测入口
 
"""

from com.data.model_warrant import warModel 
from com.base.public import public 
from com.data.data_base import fytuObject
from com.model.model_etf4 import etf4_compare
import time              

def main():
    cmdKeys = "w:"
    fw = public.getCmdParam(cmdKeys, ("-w", 2))
    
    FT = fytuObject()
    # 每10分钟检查一次窝轮行情
    if fw==1:
        valid = FT.isTradeDate('HK')  
        if valid:
            obj = warModel()
            obj.online()
    
    # 每15分钟检查一次基金申购:
    if fw==2:        
        if FT.isTradeDate('SH'):
            obj = etf4_compare()
            d, t = public.getDate(),  public.getDatetime()
            # 大于2点45分，则每15秒运行一次直到结束
            if t > d +" 14:45:00" and t < d +" 15:00:00":
                while True: 
                    t = public.getDatetime()
                    if not (t > d +" 14:45:00" and t < d +" 15:00:00"): break 
                    obj.create(1)
                    time.sleep(15)        
            else:
                # 小于2点45分，则每15分钟运行一次
                obj.create(0)

  
    
if __name__=='__main__':
    main()
  
 
    