# -*- coding: utf-8 -*-

""" 
Created on  2018-01-04 
@author: lbrein
      从各网站读取symbol信息  
 ----- 
"""
 
from com.object.obj_entity import warInfo, warrant_his, wave , warrant_online
from com.data.data_base import fytuObject
from com.data.data_interface import TensentInfo
import time 
from com.base.public import public , logger

import pandas as pd
import math 

def erfcc(x):
        """Complementary error function."""
        z = abs(x)
        t = 1. / (1. + 0.5*z)
        r = t * math.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
            t*(.09678418+t*(-.18628806+t*(.27886807+
            t*(-1.13520398+t*(1.48851587+t*(-.82215223+
            t*.17087277)))))))))
        
        if (x >= 0.):
            return r
        else:
            return 2. - r

def Normsdist(x):
        return 1. - 0.5*erfcc(x/(2**0.5))


# 港股涡轮对比模型 -  正股每日波动率计算   
class warModel:
    """
       涡轮计算模型，
       1、获取涡轮基础信息
       2、统计正股波动率 和收盘价 （在data_waveCompare)
       3、查询涡轮历史数据，并根据s_d公式计算每日的预测价格

    """    
    None_list = ['lot_size','stockid']
    WarInfo_map = {} 
    
    def __init__(self):
        self.Futu = fytuObject() # 
        self.Tensent = TensentInfo() #  
        self.War = warInfo() # 涡轮基础信息表 
        self.Wh = warrant_his() # 
        self.Wave = wave()  # 波动率结果表 
        
    # 查询并存储war基础信息
    def initWar(self):
        df = self.Futu.getStockInfo("HK","WARRANT")
        self.add(df)    
    
    #  添加到warInfo表基础数据
    def add(self,df):
        time0 = time.time()
        pc = 50  #页码
        c = df["code"].count()
        pgs = c//pc +1
        for i in range(pgs):
            s, e = i*pc, (i+1)*pc                  
            if e > c: e = c 
            # 截取pandas
            docs = df[s:e].to_dict(orient='records')
                 
            # 查询获得腾讯其他信息
            maps  = self.Tensent.getWar(df[s:e]["code"].tolist())
            for doc in docs:
                if doc["code"] in maps.keys():
                    doc.update(maps[doc["code"]])
                else:
                    doc.update({'exercise_price':0,'exchange_ratio':0,'due_date': ''})
                doc = self.alter(doc)
            
            res = self.War.insertAll(docs)
            print(i, time.time()-time0)
            if not res :
                break 
            
        self.Futu.Quant.close()    
        
    def check(self):
        # 检查并更新 
        df = self.Futu.getStockInfo("HK","WARRANT")
        # 查询已有code
        sql = "select DISTINCT 'HK.'+code from warInfo"
        rs = self.War.execSql(sql,isJson=False)
        es = [r[0] for r in rs]
        
        # 过滤
        nf = df[-(df["code"].isin(es))]
        print(nf["code"].count())
        
        self.add(nf)     
    
    # 调整     
    def alter(self,doc):
        for key in self.None_list:
            del doc[key]
        if doc['due_date'].find("-")==-1:
            doc['due_date']= '2000-01-01'
            
        s = doc["code"].split(".")
        doc["code"] = s[1]
        doc["market"] = s[0]
        return doc
    
     #每日更新涡轮股价计算
    def update(self):
        # 甲岸村最新日 
        res = self.Wh.execOne("select max(sdate) from warrant_his", isJson=False)
        start = public.getDate(1,res[0])
        self.price(start)

    # 计算价格     
    def price(self,start = None):
        # 正股中间表
        owner_maps = {}
        time0 = time.time()
        # 查询已有code
        k = 1 
        for doc in self.War.getWars():
            # 查询正股波动率
            owner = doc["owner_stock_code"][3:]
            if not owner in owner_maps.keys():
                Ow = [n for n in self.Wave.getByCode(owner)]       
                if len(Ow)==0: continue 
                owner_maps[owner] = Ow 
            else:
                Ow = owner_maps[owner]
                
            df = pd.DataFrame(Ow)
            
            if not start:
                start = doc["listing_date"] 
            end = df.values[df["sdate"].count()-1, 3] 
            end = str(end)[:10]
            
            # 查询涡轮历史每日价格 
            df_w = self.Futu.get_history_kline( "HK."+doc["code"],start=start, end=end)
            
            if type(df_w) == str :
                continue  
            
            if df_w["code"].count()>0:
                recs = []    
                for index, row in df_w.iterrows():
                    doc1 = row.to_dict()
                    rec ,doc2 = None, {} 
                    try:
                        # 计算结果
                        doc2 = df[(df['sdate']== doc1["time_key"][:10])].to_dict(orient='records')[0]
                        rec = self.calcPrice(doc, doc1, doc2)
                    except:
                        continue  
             
                    if rec:
                        recs.append(rec)
                # 保存         
                if len(recs) > 0 :
                    self.Wh.insertAll(recs)
                    
                    if k % 50==0:
                        print(k, doc["code"],time.time()-time0)
                        #break 
                    k+=1 
        
        self.Futu.Quant.close() 
        
    def calcPrice(self, doc_info, doc_war, doc_wave):
        # doc_info 涡轮基本信息 包括代码-
        # doc_war 涡轮K线数据
        # doc_wave 正股波动率
        #print(doc_info)
        res = {
               'code':  doc_info["code"], # 
               'sdate': doc_wave["sdate"], # 
               'sclose': doc_war["close"] , # 当期收盘价
               'owner': doc_wave["sclose"], # 正股价
               'ep': doc_info["exercise_price"],  # 行权价
               'sd_Price':0.0,
               'd1':0.0,
               'd2':0.0,
               }    
        
        params_d = {
                "type": doc_info["stock_child_type"],
                "S": doc_wave["sclose"], # 正股价 
                "L": doc_info["exercise_price"], # 行权价   
                "E": 1 if doc_info["exchange_ratio"]== 0  else doc_info["exchange_ratio"], # 换股比例
                "T": self.timeDiff(doc_info["due_date"], doc_wave["sdate"]), # 到期日
                "b": doc_wave["wave"] * math.sqrt(250), # 波动率
                "r": 0.01
                }
        # 计算 d1, d2 
        #print(code, params_d)
        d1, d2 = self.calc_d(params_d)
        
        # 计算实际价格  
        params_d.update({"d1":d1,"d2":d2}) 
        
        # 计算目标价
        c = self.calc_c(params_d)
        
        res.update({"d1":d1,"d2":d2,"sd_Price":c})
        return res 
    
    def calc_d(self, pm):
        s1 =  math.log(pm['S']/pm['L'])+ pm['r'] * pm["T"] 
        s2 = 0.5 * pm["T"] * pm['b']**2 
        s3 = pm['b'] * math.sqrt(pm ["T"])

        d1, d2 = (s1+s2)/s3, (s1- s2)/s3
        return d1, d2 


    def calc_c(self,pm):
        s1 = pm["S"] * Normsdist(pm["d1"])
        s2 = pm["L"] * math.exp((-1) * pm["r"] * pm["T"]) * Normsdist( pm["d2"])
        s3 = 0 
        
        if pm["type"]=="PUT":
             s3 = pm["L"] * math.exp((-1) * pm["r"] * pm["T"]) - pm["S"]


        c = (s1-s2+s3) / float(pm["E"])  
        #print(c) 
        return c
    
    # 计算时间间隔    
    def timeDiff(self,t1,t2):
        diff = public.timeDiff(t1 + " 00:00:00", str(t2) + " 00:00:00") / 60.0 / 60.0 / 24.0 / 365.0  
        return 1 if diff==0 else diff
    
    """
       在线监控程序
       
    """
    # 查询正股收盘价和前一日波动率 
    def getOwnerMap(self, owners = None):
        owners_map = {}
        res = self.Wave.getLastWave()
        # 存储对象
          # 查询获得正股波动率
        oDf = pd.DataFrame([doc for doc in res if (owners is None or doc['code'] in owners)])
        # 正股列表
        lists = ('HK.'+oDf['code']).tolist()
        
        # 查询获得正股当前股价
        closes, volumes, rates = [],[],[]
        for ls in public.eachPage(lists,pc=200):
            df = self.Futu.getSnap(ls)
            #print(df)
            if not df.empty:
                closes +=  df['last_price'].tolist()
                volumes += df['volume'].tolist()
                rates += df['turnover_rate'].tolist()
                
            time.sleep(5)
             
        #添加正股收盘价
        oDf['sclose'] = closes
        oDf['volume'] = volumes
        oDf['turnover_rate'] = rates
        oDf['sdate'] = public.getDate()
        
        docs = oDf.to_dict(orient='records')
        for doc in docs:
            owners_map[doc['code']] = doc
              
        return owners_map 
        
    # 在线监控
    def online(self):
        # 查询正股波动率和当前正股价
        time0 =time.time()
        Obj =  warrant_online() # sql数据表
        Obj.empty()
        
        # 检查链接状态


        # 查询已有code
        wars ,owners = [],[] 
        for doc in self.War.getOnline():
             wars.append(doc)   
             if doc['owner_stock_code'][3:] not in owners:
                  owners.append(doc['owner_stock_code'][3:]) # 正股ID
            
        owner_maps = self.getOwnerMap(owners)
        
        # 查询涡轮历史每日价格 
        k ,total = 0, 0  
        for docs in public.eachPage(wars,pc=200):
            #每批次200个读取当前价格
            res = [] #结果集
            lst = ['HK.'+doc['code'] for doc in docs] 
            df = self.Futu.getSnap(lst)
            
            i = 0 
            for index,row in  df.iterrows():
                doc = {"code": row['code'] ,
                       "close": row['last_price']
                       }
                
                pcode =  docs[i]['owner_stock_code'][3:]
                if pcode in owner_maps.keys(): 
                    try:
                        rec = self.calcPrice(docs[i], doc, owner_maps[pcode])
                        #更改时间为当前时间
                        if rec:
                            # 添加交易量
                            rec.update({
                                'sdate':public.getDatetime(),        
                                'w_volume':row['volume'],
                                'w_turnover_rate':row['turnover_rate'],
                                'o_volume':owner_maps[pcode]['volume'],
                                'o_turnover_rate':owner_maps[pcode]['turnover_rate'],
                             })
                            
                            res.append(rec)
                    except:
                        i+=1
                        continue
                i += 1 
            k+=1
            total += len(res)
            print(k,len(res), time.time()-time0)
            Obj.insertAll(res) 
            time.sleep(5) 
            #break 
        
        logger.info("窝轮波动率检测结束,共 %s" % total)    
        self.Futu.Quant.close() 

def main():
    actionMap = {
            "new":0, #历史数据初始化
            "check":0, 
            "price":0, # 计算每日涡轮实际价和预测价格
            "update":0,
            "online":1,
            }
    
    obj  = warModel()
    if actionMap["new"]==1:
        obj.initWar()
   
    if actionMap["check"]==1:
        obj.check()
        
    if actionMap["price"]==1:
        obj.price()
    
    if actionMap["update"]==1:
        obj.update()
        
    if actionMap["online"]==1:
        obj.online()
        
        
   
if __name__ == '__main__':
    main()
