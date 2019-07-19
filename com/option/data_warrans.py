# -*- coding: utf-8 -*-

""" 
Created on  2018-01-04 
@author: lbrein
      从各网站读取symbol信息  
 ----- 
"""

from com.base.public import csvFile, public 
from futuquant.open_context import *
from multiprocessing import Process
import numpy as np 
import time
from com.data.data_interface import c_interface
 
class OrderBookTest(OrderBookHandlerBase):
    """
    获得摆盘推送数据
    """
    def on_recv_rsp(self, rsp_str):
        """数据响应回调函数"""
        ret_code, content = super(OrderBookTest, self).on_recv_rsp(rsp_str)
        print(ret_code) 
        
        if ret_code != RET_OK:
            print("OrderBookTest: error, msg: %s" % content)
            return RET_ERROR, content
        print("OrderBookTest\n", content)
        return RET_OK, content


class StockWarrans:
    Quant = OpenQuoteContext(host='127.0.0.1', port=11111)
    df = None
    codes = []
    
    def getStockInfo(self, stock_type="STOCK"):
       ret, df = self.Quant.get_stock_basicinfo(market='HK', stock_type=stock_type)
       df = df[(df["stock_child_type"]=='PUT')|(df["stock_child_type"]=='CALL')]
       return df.reset_index()
    
    def start(self,s):
        if self.df is None:
            self.df = self.getStockInfo(stock_type="WARRANT")
        
        stock_code_list = [n for n in self.df["code"][s*50:(s+1)*50]]
        print(stock_code_list)
        for stk_code in stock_code_list:
           ret_status, ret_data = self.Quant.subscribe(stk_code, "ORDER_BOOK", push=True)
           if ret_status != RET_OK:
               print("%s %s: %s" % (stk_code, "ORDER_BOOK", ret_data))
               exit()
       
        self.Quant.set_handler(OrderBookTest())
        self.Quant.start()  
   
    # 历史K线图
    def getK(self,num = None):
        if self.df is None:
            df = self.df = self.getStockInfo(stock_type="WARRANT")
        else:
            df = self.df
     
        count = df["code"].count()
        if num is None:
            num = count
            
        for i in range(num):
            res = self.getHistory(df["code"][i])
            yield df["code"][i], df["owner_stock_code"][i], df["name"][i], res 
            
    #         
    def check(self):
        time0 = time.time()
        CSV = csvFile(folder="/data/csv", filename="history.csv", 
                      headers=["代码", "正股", "名称","最低点时间","最低点价格",
                               "最高价时间","最高点", '行使价','换股比率','到期日',
                               '初始日low','初始日high','购后最低价','购后最低价时间','购后最高价','购后最高价时间'
                               ])
        doc_list = []
        i = 0 
        
        for code,owner, name, res in self.getK():
            if res:
                na = np.array((res))
                # 检查存在0.01的标的
                # n = np.where((na[1,:]<= 0.010) & (na[1,:]> 0.0))
                n = np.where(na[1,:] == 0.010)
                if n[0].size > 0:
                    nw = na[:, n[0][0]:]
                    imax = np.max(nw[2]) # 取最高价
                    if imax > 0.0:
                        # 查询最大值所在时间
                        nmax = np.where(nw[2,:]==imax) 
                        doc = {
                                "代码": code,
                                "code": code,
                                "正股":owner,
                                "owner":owner,
                                "名称": name,
                                "最低点时间": na[0][n[0][0]],
                                "starttime": na[0][n[0][0]],
                                "最低点价格":na[1][n[0][0]] ,
                                "最高价时间": nw[0][nmax[0][0]],
                                "endtime": nw[0][nmax[0][0]],
                                "最高点": imax
                                }
                        doc_list.append(doc)
                        # 添加其他信息
                        #CSV.writeLine(doc)
                if i % 100==0: 
                    print(i, code, time.time()-time0) 
                i += 1            
        # 添加其他信息 
        doc_list = self.updateDoclist(doc_list)
        # 添加正股信息
        for doc in doc_list:
            for key in ['endtime', 'starttime', 'owner', 'code']:
                del doc[key]
            CSV.writeLine(doc)
        CSV.close()              
    
    # 查询历史K线图                
    def getHistory(self, code, start = None, end = None):
        # 默认为一年
        if start is None:
            start = public.getDate(diff=-365) 
     
        ret, df = self.Quant.get_history_kline(code, start= start, end= end, ktype='K_DAY', autype='qfq', fields=[KL_FIELD.ALL])
        if ret ==-1: return 0 
        return [df["time_key"], df["low"], df["high"]]
    
    
    Interface = None
    # 查询腾讯API，获取其他数据
    def getQt(self,code_list):
        if self.Interface is None :
            self.Interface = c_interface()
        
        ss = [n.replace("HK.","hk") for n in code_list]
        time0  = time.time()    
        href = "http://qt.gtimg.cn/q=" + ",".join(ss)
        #print(href)
        res = self.Interface.getI(href)
        
        if res.find('";')>0:
            codes = res.split(";")
        docMap = {}
        for txt in codes:
            doc = {'行使价':0,'换股比率':0,'到期日': ''}
            kl = txt[txt.find('="')+2:-2].split("~")
            if len(kl) > 47:
                doc.update({'行使价':kl[44],'换股比率':kl[45],'到期日':kl[47]})
                docMap['HK.'+kl[2]] = doc

        return docMap    
        
    # 更新正股、到期日等信息
    def updateDoclist(self,doclist):
        # 查询正股信息
        owners = {}
        for doc in doclist:
            if not doc["owner"] in owners.keys():
                owners[doc["owner"]]= self.getOwnerMax(doc)
            
            if owners[doc["owner"]]:
                doc.update(owners[doc["owner"]])
        
        # 查询到期日等数据
        pageCount = 50
        count = len(doclist)
        pages = (count-1)// pageCount +1 
        #print(count,pages)
        for i in range(pages):
            start, end = i*pageCount, (i + 1)* pageCount 
            if end > count :  end = count 
            k = 0
            ls = [doclist[k]["code"] for k in range(start,end)]
            # 更新
            maps = self.getQt(ls)
            for k in range(start,end):
                if doclist[k]["code"] in maps.keys():
                    doclist[k].update(maps[doclist[k]["code"]])
                
        return doclist    
        
    # 查询并获得正股    
    def getOwnerMax(self,doc):
        if doc["owner"] =="": return None
        
        df = self.getHistory(doc["owner"], doc["starttime"][:-9], doc["endtime"][:-9]) 
        res= {"初始日low":0,"初始日high":0, "购后最低价":0,"购后最高价":0}
       
        if df:
            na =  np.array((df))
            # 添加最初日的最高最低价
            res.update({
                        "初始日low": na[1,0],
                        "初始日high": na[2,0]
                        })
                            
            # 添加之后的最高最低价                
            if doc["starttime"] != doc["endtime"]:
                fmin = np.min(na[1,1:])
                fmax = np.max(na[2,1:])
                imin = np.where(na[1,1:]==fmin)[0][0] # 最小值所有行                  
                imax = np.where(na[2,1:]==fmax)[0][0] #值所有行
                res.update({
                        "购后最低价": fmin,
                        "购后最低价时间": na[0][imin],
                        "购后最高价": fmax,
                        "购后最高价时间": na[0][imax]
                        })
        return res    
                 
    
    def getMore(self,doc_list):
        total = len(doc_list)
        pageCount = 100 
        pages = total//100+1 
        for i in range(pages):
            start, end = i* pageCount , (i+1)*pageCount
            if (i+1)* pageCount > total: end = total   
            ls = [doc_list[k]["code"] for k in range(start,end,1)]
            self.getQuotes(ls)
    
    
    def getOwner(self,stock_code_list):
              # subscribe "QUOTE"
            for stk_code in stock_code_list:
                ret_status, ret_data = self.Quant.subscribe(stk_code, "QUOTE")
                if ret_status != RET_OK:
                     exit()
        
            ret_status, ret_data = self.Quant.get_stock_quote(stock_code_list)
            if ret_status == RET_ERROR:
                print(ret_data)
                exit()
            
            df = ret_data
            # 取消订阅
            for stk_code in stock_code_list:
                ret_status, ret_data = self.Quant.unsubscribe(stk_code, "QUOTE")
            
            print(df)
            return [df["code"]] 
       
    # 市场快照
    def snap(self,s):
       if self.df is None:
            self.df = self.getStockInfo(stock_type="WARRANT")
        
       stock_code_list = [n for n in self.df["code"][s*200:(s+1)*200]]
       ret_status, ret_data = self.Quant.get_market_snapshot(stock_code_list)
       if ret_status == RET_ERROR:
           exit()
       else:    
          return self.test(ret_data)     
    
    def test(self,result):    
        ay = np.array((result["code"],result["last_price"]))
        ny = np.where((ay[1,:]<0.02) & (ay[1,:]> 0.01))
        res = [[ay[0,i],ay[1,i]] for i in ny] 
        return res
    
    def process(self):
        time0 = time.time()
        self.codes = []
        self.df = df = self.getStockInfo(stock_type="WARRANT")
        count = df["code"].count()
        
        pageCount = 200 
        pages = count//pageCount+1
        print(pages)
        for i in range(pages):
            d = self.snap(i)
            if d:
                self.codes+=d 
            print(i,time.time()-time0)
            time.sleep(1)
        print(self.codes)    
        
            
def go(k,j=0):
    print("---",k)
   # obj.start(k)       

def main():
    obj = StockWarrans()
    obj.check()
 

if __name__ == '__main__':
    main()
    """
    obj = StockWarrans()
    process_list = []
    for i in range(5):
        print(i)
        p = Process(target=go, args=(i,0)) 
        process_list.append(p)
        p.start()
        
    for j in process_list:
        j.join()
    """

    