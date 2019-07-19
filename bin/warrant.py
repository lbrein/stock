# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Created on Sat Apr  1 14:35:13 2017

@author: admin

"""
from com.data.model_warrant import warModel 
from com.data.model_owner import OwnerWave 
from com.base.public import public 
from com.data.data_base import stockBase

def main():
    cmdKeys = "w:"
    fw = public.getCmdParam(cmdKeys, ("-w", 5))
    
    # 初始化涡轮基础信息，访问腾讯
    if fw==1:
        obj = warModel()
        obj.initWar()
    
    # 计算正股波动率和每日收盘价    
    elif fw==2:
        obj1 = OwnerWave()
        obj1.iniHK()
    
    # sd公式 计算每日涡轮价格，对比记录 
    elif fw==3:
        obj = warModel()
        obj.price()

    # 每日更新     
    elif fw==4:
       obj1 = OwnerWave()
       obj1.update() 
        
       # obj = warModel()
       # obj.update()
    
    # 保持链接测试  
    elif fw==5:
       obj = stockBase()
       obj.keepLink()
       
        
if __name__=='__main__':
    main()
  
 
    