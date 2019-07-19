# -*- coding: utf-8 -*-

""" 
Created on  2018-04-14 
@author: lbrein
       从文件导入历史数据到sqlserver,包括：
       1、股票code清单 （futu)
       2、交易所软件导出的历史数据
"""

#import math
import os
import re
import time
from futuquant.open_context import *

from com.base.public import csvFile, public
from com.object.obj_entity import baseInfo, history, history_qfq
from com.object.mon_entity import m_history

import pandas as pd

class StockImport:
    Quant = None
    History = None

    def getStockInfo(self, market="HK", stock_type="STOCK"):
        ret, df = self.Quant.get_stock_basicinfo(market=market, stock_type=stock_type)
        return df

    def importStockInfo(self):
        self.Quant = OpenQuoteContext(host='127.0.0.1', port=11111)
        Sql = baseInfo()

        Areas = ["SH", "SZ", "HK"]
        types = ["STOCK", "IDX", "ETF", "BOND"]

        for n in Areas:
            for m in types:
                df = self.getStockInfo(market=n, stock_type=m)
                docs = []
                print(n, m, df.count())
                for index, row in df.iterrows():
                    doc = {}
                    doc = {
                        "code": row["code"][3:],
                        "name": row["name"],
                        "market": n,
                        "stocktype": m,
                        "listing_date": row["listing_date"]
                    }
                    docs.append(doc)
                # print(docs)
                Sql.insertAll(docs)

    def importHistory(self):
        folder = "/data/source"
        time0 = time.time()
        self.History = history()

        i = 0
        for fname, file in self.getFile(folder):
            docs = self.read(file)
            self.History.insertAll(docs)
            if i % 100 == 0:
                print(i, fname, "waste:",  time.time() - time0)
            i += 1

    def getFile(self, folder):
        list = os.listdir(folder)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(folder, list[i])
            fname = path[path.find("据\\") + 2:]
            if os.path.isfile(path):
                file = open(path, "r")
                yield fname, file
   
                
    def check(self):
        folder = "D:\projects\python\stock\data\source"
        self.History = history()
        sql = "select DISTINCT code from history"
       
        res = self.History.execSql(sql,isJson=False)
        ls = [str(n[0]) for n in res]
        for fname, file in self.getFile(folder):
            key = fname[:-4]
            if not key in ls :
               print( key) 
               #i+=1
               #if i < 5: continue 
               
               docs = self.read(file)
               self.History.insertAll(docs)
               """
               j = 0     
               print(docs[1])
               for doc in docs:
                   try:
                       self.History.insert(doc)
                       j += 1
                   except Exception as e:
                       print(j, e, doc)
                       break 
               #break 
               """
    def read(self, file):
        line = True
        names = self.History.keylist[2:]
        docs = []
        k = 0
        code =  ""
        while line:
            line = file.readline()
            # 标题
            if k == 0:
                txt = line.strip()
                #name = txt[:txt.find("(")].strip()
                code = txt[txt.find("(") + 1:txt.find(")")].strip()
                #print(code, name)
            #
            elif k > 2:
                doc = {"code": code, "area": "sz"}
                txt = re.sub('\s+', "|", line.strip())
                ary = txt.split("|")
                if len(ary) < 10: continue
                for j in range(0, 6):
                    doc[names[j]] = ary[j]
                if int(doc["volumn"])>0 : 
                    docs.append(doc)
            k += 1
        return docs
        
class stockTrans:
     
     def __init__(self):
         self.m_His = m_history()
         self.Qfq = history_qfq()
         
     def create(self):
         sql = """select DISTINCT code 
                from history 
                where code > '300000' and code not in (select DISTINCT code from history_qfq) 
                ORDER BY code  
         """
         i = 0 
         for item in self.Qfq.execSql(sql,isJson=False):
             #print(item[0])
         
             res = self.m_His.getPrice(item[0])
             if res.count()> 0: 
                 df = pd.DataFrame([doc for doc in res])
                 df['sdate'] = df['date']
                 df['sopen'] = df['open']
                 df['sclose'] = df['close']
                 
                 for key in ['_id','date','open','close','limit_down','limit_up']:
                    del df[key]
                 self.Qfq.insertAll(df.to_dict(orient='records'))  
                 print(i, item[0])
             i+=1
                 #break 
    
def main():
    actionMap = {
        "baseInfo": 0,
        "history": 0,
        "check":0,
        "trans":1

    }
    obj = StockImport()
    if actionMap["baseInfo"] == 1:
        obj.importStockInfo()

    if actionMap["history"] == 1:
        obj.importHistory()

    if actionMap["check"] == 1:
        obj.check()

    if actionMap["trans"] == 1:
        obj = stockTrans()
        obj.create()


if __name__ == '__main__':
    main()
