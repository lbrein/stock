# -*- coding: utf-8 -*-

""" 
Created on  2018-05-03 
@author: lbrein
      
        基金监控买单 ---            
 ----- 
"""
from com.base.public import  public, config_ini, logger
from com.object.obj_entity import ETFRecord , account  
from com.data.Interface_Matic import Matic_interface 
from futuquant import *
from com.data.data_base import fytuObject
from com.data.data_interface import s_webSocket
import time , math 

class etf4_compare:

    def __init__(self):
        self.stage = "ETF"
        self.FT = fytuObject()
        # 股票列表
        self.stock_code_list = "SZ.159901,SZ.159902,SZ.159915,SH.510050,SH.510900, SH.510300,SH.519920,\
                                         SH.512160,SH.512880,SH.512800,SH.510180,SH.512660".split(",")
        #账户
        self.account = self.iniKey('account')
            
        #interval = self.iniKey('interval')
        self.k_amount = int(self.iniKey('k_amount'))
        self.k_type = self.iniKey('k_type')   

        #self.ETF = ETFRecord() #
        #self.Acc = account() # 账户
        
        self.Matic = Matic_interface() # matic 接口类
        # 上一k线时段成交量
        self.lastVolume_Map = {}
        self.ws = s_webSocket("ws://%s:8090/forward" % config_ini.get("ws.host", session="db"))

    def iniKey(self,key):
        return config_ini.get( key, session = self.stage)

    # 监听仅通知 - 手动操作  
    def inform(self, type = 0):
        docs = self.spy(type) 
        # 查询获得最新报价        
        docs = self.priceCheck(docs)

        # 生成twap 操作文档
        results = self.Matic.Twap(docs)
        # 发送到ws服务度端
        self.ws.send(results)

    # 监听并自动执行下单, TPI下单模式
    def create(self,type = 0):
        # 检测满足购买条件文件
        docs = self.spy(type)

        #计算下单文件
        orders = self.order(docs)
        # 生成VWAP,生成批量模型, 并通过Matic下单

        self.Vwap(orders)
        logger.info("批量下单完成!")
    
    # 数据监听
    def spy(self, type = 0):
        # 订阅数据
        self.FT.subscribe(self.stock_code_list, self.k_type) 
        
        # 统计
        recs = []
        for code in self.stock_code_list:
             # 获得基金320个K线历史数据
            ret_code, ret_data = self.FT.Quant.get_cur_kline(code, self.k_amount+1, self.k_type) 
            if ret_code == RET_ERROR: 
                continue 
            
            # 分2个时段策略计算比值 
            df = ret_data
            rec = self.calc(code, df, type= type) 
                               
            if rec["type"]!=0:
                recs.append(rec)  
        return sorted(recs, key= lambda d: d["value"]) 
    
    # 计算bias值        
    def calc(self, code, df, type = 0 ):
         
        if type ==0:
            avg = df["close"][:-1].mean()
            c = df['close'][self.k_amount-1] 
            self.lastVolume_Map[code] = df['volume'][self.k_amount-1]
        else:    
            avg = df["close"][1:].mean()
            c = df['close'][self.k_amount] 
            self.lastVolume_Map[code] = df['volume'][self.k_amount]
        
        res = (c - avg) / avg * 100
        
        # 购买方式
        if res>1:
            type = 1 
        elif res< -1:
            type = -1
        else:
            type = 0
            
        # 生成记录
        doc = {
               "code": code, 
               'returnid':0,
               'type': type,
               'price': round(c,3),
               'volume':  int(round(500000/c/100,0)*100),
               'amount':0 ,
               'value': res,
               'createtime': public.getDatetime()
               }
        return doc  
    
    # 检查资金账户，按value值 
    Funds = None
    def order(self,docs, price = None):
        if self.Funds is None:
            self.Funds = self.Matic.getFund() 
        
        # 查询记录，获取总量
        df = self.Funds
        
        #可用余额和已存在股票清单 
        free = df[df["stock_code"]=="free"]["enable_amount"].values[0]
        exists = df["stock_code"].tolist()
                
        # 每份额度 ，当前可买份数和实际份额 
        eachAmount = int(self.iniKey("eachAmount")) 
        pages = int(round(free / eachAmount, 0))  
        realAmount = free / pages 
        
        recs = []
        # 先卖后买
        for doc in docs:
            # 卖出
            if doc["type"] == -1:
                if not doc["code"] in exists: continue 
                # 检查是否允许卖出  
                a = df[df["stock_code"]==doc["code"]]["enable_amount"].values[0]
                if a > 0 :
                   doc["volume"] = a 
                   pages += 1 # 可买份数加1
                   recs.append(doc)
                   
        # 买de1, 按values倒序 
        docs.reverse()
        for doc in docs:
            if doc["type"] == 1:
                # 买入基金，份额低于0则退出
                if doc["code"] not in exists:
                    pages = pages - 1
                    doc["amount"] = realAmount
                    recs.append(doc)
                    if pages ==0 :
                        break
        return recs        

    # 按orderbook 进行拆单交易
    def Vwap(self,docs):
        v_inter, v_rate = int(self.iniKey('vwap_interval')), float(self.iniKey('vwap_volume_rate'))
        
        No = 0
        # 根据深度数据，自动批量下单
        while True:
          # 总共需要卖出的手数 用于卖出
          total_v = sum([doc["volume"] for doc in docs if doc["type"]==-1])
          
          # 总共需要买入金额 用于买入
          total_a = sum([doc["amount"] for doc in docs if doc["type"]== 1])
              
          if total_v <= 0  and total_a <= 0:  
                break 
          cur_docs, docs = self.vwap_volume(docs, v_rate) 
          # 下单
          self.Matic.order(cur_docs) 
          No += 1             
          logger.info("批量下单批次号: %s" % No) 
          # 下单间隔时间
          time.sleep(5) 
    
    # 查询当前时间点的level,并选择1/2档的一半进行下单操作       
    def vwap_volume(self,docs, v_rate):
        rec, recs = {}, []
        # 深度层次
        dep_length = int(self.iniKey("vwap_depth_length")) 
        
        # 查询深度数据，并计算每次交易的量
        for doc in docs:
            rec = {}
            rec.update(doc)
            
            # 获得深度数据
            book = self.FT.get_order_book(doc["code"])
            if doc["type"] == -1: 
                # 卖出的量
                rec = self.v_depth(rec, book['Ask'][:dep_length], v_rate, doc["volume"])                 
                # 递减初始交易量
                doc["volume"] =  doc["volume"] - rec ["volume"] 
            else:
                # 买入的量价
                rec = self.v_depth(rec, book['Bid'][:dep_length], v_rate, doc["amount"]) 
                # 递减初始交易量
                doc["amount"] =  doc["amount"] - rec ["amount"] 
            
            if rec["amount"]> 0: 
                recs.append(rec) 
            
        return recs, docs
    
    # 深度数据计算
    def v_depth(self, rec, dep, v_rate, cur_amount):
        if len(dep)==0: 
            return rec
        
        p = sum ([n[0] for n in dep]) / len(dep) 
        v = sum([n[1] * v_rate  for n in dep]) 
        a = sum([n[0] * n[1] * v_rate * 1.0 for n in dep]) 
       
        # 卖出通过量计算a      
        if rec["type"] == -1:
            v = v if cur_amount > v else cur_amount 
            a = v * p 
        
        # 买入通过金额进行手数
        else:    
            a = a if cur_amount > a else cur_amount
            v = int(round( a / p / 100,0) * 100)  
            
        rec.update({
                "price": p, 
                "volume": v,
                "amount": a,
                })
        return rec 
    
    # 检查最新报价， inform提示价格和交易量 
    def priceCheck(self, docs):
        # 更新price 和 volume 
        df_prices = self.FT.get_cur_quote([doc["code"] for doc in docs]) 
        for doc in docs:
           p , v = doc["price"] , doc["volume"] 
           p1 = df_prices.loc[(df_prices["code"] == doc["code"]), "last_price"].values[0]
           doc["price"] = p1   
           doc["volume"] = int(round(p*v/p1/100,0)*100) 
        
        return docs    
    
    def resultCheck(self):
        pass
    
    
def main():
    actionMap = {
            "new":1, 
            "price":0, # 
            "update":0,
            "inform":0
            }
    
    obj  = etf4_compare()
    if actionMap["new"]==1:
        obj.create(0)

    if actionMap["price"]==1:
        obj.price()
    
    if actionMap["update"]==1:
        obj.update()
        
    if actionMap["inform"]==1:
        obj.inform(type=0)
    
    
if __name__ == '__main__':
    main()
