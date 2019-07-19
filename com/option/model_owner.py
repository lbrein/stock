# -*- coding: utf-8 -*-

""" 
Created on  2018-01-04 
@author: lbrein

        计算每日正股波动率     
 
 ----- 
"""
from com.base.public import  public
from com.object.obj_entity import wave
from com.data.data_base import fytuObject
import numpy as np 
import time
    
    
# 港股涡轮对比模型 -  正股每日波动率计算   
class OwnerWave:
    sql_owner = """
        insert into wave 
        (code, sdate, wave, [count])
        select a.code, '$enddate',  STDEVP(log(b.sclose/a.sclose)) as sd , count(b.sclose/a.sclose) as [count]
         from  
        (select code, sclose, 
         row_number() over ( partition by code order by sdate asc) rn
                          
         from history 
         where SUBSTRING(code,1,2) in ('00','51') and code = '510050'
         and DATEDIFF(day , sdate, '$enddate') BETWEEN 0 and 365) a  
         LEFT JOIN 
        (select code, sclose, 
         row_number() over ( partition by code order by sdate asc) rn
                          
         from history 
         where SUBSTRING(code,1,2) in ('00','51') and code = '510050' 
         and DATEDIFF(day , sdate, '$enddate') BETWEEN 0 and 365) b 
         on a.code = b.code and a.rn = b.rn+1 
         GROUP BY a.code having count(b.sclose/a.sclose)> 0
    """
    
    def __init__(self):
        self.Futu = fytuObject()
        self.Wave = wave()
        self.dates = []

    # sh-sz 正股波动率通过 sql进行计算 
    def iniSH(self):
        time0 = time.time()
        dates = self.Futu.get_trading_days("HK",start_date='2016-01-01')
        for i in range(len(dates)-1,0,-1):
            sql = self.sql_owner.replace("$enddate", dates[i])
            self.Wave.update(sql)
            print(dates[i],time.time()-time0)
            
    # 计算港股历史每日的波动率        
    def iniHK(self, start = '2015-01-01'):
        time0 = time.time()

        owners = self.Futu.getWarrantOwners()
        j = 0 
        for code in owners:
            docs = self.calcWave( code, start ) 
            
            self.Wave.insertAll(docs)
            print(j,code,time.time()-time0)
            j+=1
        
        self.Futu.close()
        
    def recheck(self):
        # 所有正股
        time0 = time.time()
        owners = self.Futu.getWarrantOwners()
        sql =  "select DISTINCT code from wave"
        # 已有
        exists = [d[0] for d in self.Wave.execSql(sql,isJson=False)]
        j = 0 
        for code in owners:
            if not code[3:] in exists:
                j += 1
                docs = self.calcWave( code,'2015-01-01') 
                
                self.Wave.insertAll(docs)
                print(j, code,time.time()-time0)
                
    #每日更新涡轮股价计算
    def update(self):
        # 甲岸村最新日 
        res = self.Wave.execOne("select max(sdate) from wave", isJson=False)
        start = public.getDate(-364,res[0])
        self.iniHK(start)
   
     
    def calcWave(self, code, start):
        df = self.Futu.get_history_kline(code, start)
        
        #print(df)
        if len(self.dates)==0: 
            dates = self.dates = self.Futu.get_trading_days("HK", start_date = public.getDate(365, start))
        else:
            dates = self.dates
            
        docs = []  
       
        # 计算从2016-01-01开始的波动率 
        for i in range(len(dates)-1, -1,-1): 
            s, e = public.getDate(-365, dates[i])+" 00:00:00", dates[i]+" 00:00:00"
            
            sub1 = df[(df["time_key"]>s) & (df["time_key"]<=e)] # 过滤起止时间
            
            dd2 = sub1["close"].div(sub1["close"].shift(1))  # 相除 
            ss = np.log(dd2[1:]) # 指数
          
            if ss.count() > 0:
                sc = 0 
                
                if df[df["time_key"]==e]["close"].values:
                    sc = df[df["time_key"]==e]["close"].values[0]
                    
                doc = {
                       "code": code[3:],
                       "sdate": dates[i],
                       "wave": np.std(ss), # 标准差
                       "count": ss.count(),
                       "sclose": sc  
                       }
                if sc!=0: 
                    docs.append(doc)
                
        return docs     

    def stat(self):
        pass


def main():
    actionMap = {
            "new":0, #历史数据初始化
            "check":0, 
            "price":0, # 计算每日涡轮实际价和预测价格
            "update":1
            }
    
    obj  = OwnerWave()
    if actionMap["new"]==1:
        obj.iniHK()
   
    if actionMap["check"]==1:
        obj.recheck()
        
    if actionMap["price"]==1:
        obj.price()
    
    if actionMap["update"]==1:
        obj.update()
        
   
if __name__ == '__main__':
    main()
