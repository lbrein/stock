# -*- coding: utf-8 -*-

""" 
Created on  2018-01-04 
@author: lbrein



 ----- 
"""
 
from com.object.obj_entity import sh50_daily_1, sh50_daily_tmp
import time 
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
class sh50_Model:
    """
       sh50 计算期权价格
       params_d = {
                "type": doc_info["stock_child_type"],
                "S": doc_wave["sclose"], # 正股价
                "L": doc_info["exercise_price"], # 行权价
                "E": 1 if doc_info["exchange_ratio"]== 0  else doc_info["exchange_ratio"], # 换股比例
                "T": self.timeDiff(doc_info["due_date"], doc_wave["sdate"]), # 到期日
                "b": doc_wave["wave"] * math.sqrt(250), # 波动率
                "r": 0.01
                }
       ['code', 'date', 'close', 'expireDay', 'owner_price', 'exe_price',  'mode', 'range']

    """    
    None_list = ['lot_size','stockid']
    WarInfo_map = {} 
    
    def __init__(self):
        self.Data = sh50_daily_1()#
        self.rate = 0.05

    def start(self):
        Tmp = sh50_daily_tmp()
        for df in self.Data.getAll():
            df["expireDay"] =  df["expireDay"] / 365

            print(df["code"].values[0])
            docs = []
            Tmp.empty()
            for pm in df.to_dict(orient="records"):
                pm["d1"], pm["d2"] = self.calc_d(pm)
                if pm["d1"]:
                    pm["bsPrice"] = self.calc_c(pm)
                    docs.append(pm)
            Tmp.insertAll(docs)
            Tmp.updateBs()


    def calc_d(self, pm):
        s1 =  math.log(pm['owner_price']/pm['exe_price'])+ self.rate * pm["expireDay"]
        s2 = 0.5 * pm["expireDay"] * pm['range']**2

        s3 = pm['range'] * math.sqrt(pm ["expireDay"])

        if s3 == 0 :
            return None, None

        d1, d2 = (s1+s2)/s3, (s1- s2)/s3
        return d1, d2 


    def calc_c(self,pm):
        s1 = pm["owner_price"] * Normsdist(pm["d1"])
        s2 = pm["exe_price"] * math.exp((-1) * self.rate * pm["expireDay"]) * Normsdist( pm["d2"])
        s3 = 0 
        
        if pm["mode"]==-1:
             s3 = pm["exe_price"] * math.exp((-1) * self.rate * pm["expireDay"]) - pm["owner_price"]

        return  s1-s2+s3

    sql_group = """
    
    
    
    """


def main():
    actionMap = {
            "start":1, #历史数据初始化
            }
    
    obj  = sh50_Model()
    if actionMap["start"]==1:
        obj.start()
   

   
if __name__ == '__main__':
    main()
