# -*- coding: utf-8 -*-

""" 
Created on  2018-01-04 
@author: lbrein

        计算每日正股波动率     
 
 ----- 
"""
from com.base.public import  public
from com.object.obj_entity import stockdetail, history_qfq 
from com.object.mon_entity import m_history  

#from com.data.data_base import fytuObject
import pandas as pd     
    
# 港股涡轮对比模型 -  正股每日波动率计算   
class Industry_compare:
    
    def __init__(self):
        self.Ind = stockdetail()
        self.His = history_qfq()
        self.mHis = m_history()
        
        
    def create(self):
        # 所有行业前2位 
        
        docs = [doc for doc in self.Ind.getIndu2()]
        for res in public.eachPage(docs,2):
            #分列  
            maxD = res[0]['startdate']
            if maxD < res[1]['startdate']:
                 maxD = res[1]['startdate'] 
                
            coms = self.His.getCompare(res[0]['code'], res[1]['code'], maxD)
            df = pd.DataFrame([ doc for doc in coms])
            print(df)
            break
        
    
    def check(self,df):
        pass
        
def main():
    actionMap = {
            "new":1, #历史数据初始化
            "check":0, 
            "price":0, # 计算每日涡轮实际价和预测价格
            "update":0
            }
    
    obj  = Industry_compare()
    if actionMap["new"]==1:
        obj.create()
   
    if actionMap["check"]==1:
        obj.recheck()
        
    if actionMap["price"]==1:
        obj.price()
    
    if actionMap["update"]==1:
        obj.update()
        
   
if __name__ == '__main__':
    main()
