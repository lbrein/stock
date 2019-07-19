# -*- coding: utf-8 -*-

""" 
Created on  2018-04-10 
@author: lbrein
         -- 计算正股波动率 -- 
         包括futu数据和 sqlserver的数据  
      
 ----- 
"""
from com.base.public import csvFile, public 
from com.object.obj_entity import history 
from futuquant.open_context import *
import numpy as np 


class StockBase:
    Quant = OpenQuoteContext(host='127.0.0.1', port=11111)
    
    def getStockInfo(self, market="HK", stock_type="STOCK"):
       ret, df = self.Quant.get_stock_basicinfo(market= market, stock_type=stock_type)
       return df 
    
    def getH(self,code):
       date = public.getDate(diff=-365) 
       ret, df = self.Quant.get_history_kline(code, start= date, end=None, ktype='K_DAY', autype='qfq', fields=[KL_FIELD.ALL])
       return df 
    
    def getHistory(self, code, autype='qfq'):
       date = public.getDate(diff=-365) 
       ret, df = self.Quant.get_history_kline(code, start= date, end=None, ktype='K_DAY',
                                              autype=autype, fields=[KL_FIELD.ALL])
       
       if ret ==-1: return 0 
       aa = df["close"]
       ss = []
       for i in range(len(aa)-1):
           if aa[i+1]!=0:
               ss.append(aa[i]/aa[i+1]-1)
       return len(ss), np.mean(ss), np.std(ss)       
    
    def getBase(self,code):
        df = self.getH(code) 
        CSV = csvFile(folder="/data/csv",
                     filename= "k_%s.csv" % (code),
                     headers=df.columns.tolist())
        for index, row in df.iterrows():
            doc = {}
            for key in df.columns.tolist():
                doc[key] = row[key]
            CSV.writeLine(doc)
        CSV.close()
        
    # 按条件输出波动率
    def getWave(self, market='HK', stock_type="STOCK", autype='qfq'):
       date = public.getDate(diff=0) 
       CSV = csvFile(folder="/data/csv",
                     filename= "rate_%s_%s_%s.csv" % (market, autype, date),
                     headers=["code","name", "count", "avg", "rate"])

       df = self.getStockInfo(market, stock_type) 
       i = 0 
       for index, row in df.iterrows():
           _c, _m, _s = self.getHistory(row["code"], autype)
           doc = {
                   "code": row["code"],
                   "name":row["name"],
                   "count": _c,
                   "avg":_m, 
                   "rate": _s
                   }
           #print(doc)
           #break  
           if i%100==0:
               print(i,doc)
           i+=1
           CSV.writeLine(doc)
           
       CSV.close()      

import math 

class warrens:
    Sql = history()   
 
    def create(self):
        CSV = csvFile(folder="/data/csv",
                     filename= "rate_%s_%s.csv" % ('SZ','05'),
                     headers=["code","name", "count", "avg", "rate",'firstday'])
        
        # 查询sql，获取code列表
        res = self.Sql.getCodes()
        i = 0 
        for item in res:
            #if item[0] not in ["601601","002142","601238","601211","601818"]: continue 
            # 计算波动率 
            _f, _c,_m,_sd = self.calc(item[0])
            doc = {
                   "code":str(item[0]),  
                   "name":item[1],
                   "count":_c,
                   "avg":_m,  # 均值
                   "rate": _sd, # 标准差
                   "firstday":_f  
                    }
            #print(doc)
            #if i% 100==0: print(i, doc)
            CSV.writeLine(doc)
            i+=1 
        
        CSV.close()    
    
    def calc(self, code):
       res = self.Sql.getByCode(code)
       
       ts = [ doc for doc in res ]
       docs = [round(doc[1],2) for doc in ts ]
       
       c = len(docs)
       ss =[]
       for i in range(0, c-1):
           a = math.log(docs[i+1]/docs[i]) 
           ss.append(a)
       
       ns = np.array((ss))
       return ts[0][0], len(ss), np.mean(ns), np.std(ns)    
    

def wave():
   obj = StockBase()
   maps = [
           #["SH","qfq"],
           ["SZ","qfq"],
           #["HK","qfq"],
           #["HK", None]
          ]         
   
   for n in maps:
       print(n)
       obj.getWave(n[0],n[1])

def main():
    actionMap = {
              "stock":0, # 沪深正股波动率 
              "war":1    # 港股涡轮波动率
                }
    if actionMap["stock"] == 1:
       obj = warrens() 
       obj.create()
    
    if actionMap["war"] == 1:
        obj = StockBase()
        obj.getWave("HK","WARRANT","qfq")
    
if __name__ == '__main__':
    main()
