# -*- coding: utf-8 -*-

""" 
Created on  2018-01-04 
@author: lbrein
       计算正股的波动率 
       包括港股 和 
 ----- 
"""
from com.base.public import csvFile, public , public_basePath
import pandas as pd 
from futuquant.open_context import *
import numpy 


class StockBase:
    Quant = OpenQuoteContext(host='127.0.0.1', port=11111)
    
    def getStockInfo(self, market="HK", stock_type="STOCK"):
       ret, df = self.Quant.get_stock_basicinfo(market= market, stock_type=stock_type)
       return df 
    
    def getH(self,code, start = None):
       if start is None:
            start = public.getDate(diff=-365) 
            
       ret, df = self.Quant.get_history_kline(code, start= date, end=None, ktype='K_DAY', autype='qfq', fields=[KL_FIELD.ALL])
       return df 
    
    def getHistory(self, code, autype='qfq'):
       date = public.getDate(diff=-365) 
       ret, df = self.Quant.get_history_kline(code, start= date, end=None, ktype='K_DAY', autype=autype, fields=[KL_FIELD.ALL])
       
       if ret ==-1: return 0 
       
       aa = df["close"]
       ss = []
       for i in range(len(aa)-1):
           if aa[i+1]!=0:
               ss.append(aa[i]/aa[i+1]-1)
       return numpy.std(ss)       
    
    def getBase(self,code):
        df = self.getH(code,3)
        #print(type(df))
        if type(df) == str:  
            print("None")
            return None
        df.to_csv(public_basePath+"/data/csv/"+"k_%s.csv" % (code.replace(".","_")))
        
    # 按条件输出波动率
    def getWave(self, market='HK', autype='qfq'):
       CSV = csvFile(folder="/data/csv",
                     filename= "rate_%s_%s.csv" % (market,autype),
                     headers=["code","name","rate"])

       df = self.getStockInfo(market) 
       i = 0 
       for index, row in df.iterrows():
           i+=1
           doc = {
                   "code": row["code"],
                   "name":row["name"],
                   "rate":self.getHistory(row["code"], autype) 
                   }
           if i%200==0:
               print(i,doc)
               
           if i>2: return  
            
           CSV.writeLine(doc)
       CSV.close()     
    

def wave():
   obj = StockBase()
   maps = [
           ["HK","qfq"],
          ]         
   
   for n in maps:
       print(n)
       obj.getWave(n[0],n[1])

def main():
    obj = StockBase() 
    obj.getBase('HK.10377')
   
if __name__ == '__main__':
    main()
