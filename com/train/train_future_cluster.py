# -*- coding: utf-8 -*-
"""
Created on 2017-10-09 
@author: lbrein
   使用聚类自动进行联赛分类：  
"""

from com.base.public import logger, public
from sklearn.cluster import KMeans
from com.base.stat_fun import per
import numpy as np
from com.object.obj_entity import train_total, future_baseInfo, stock_uniform
from com.data.interface_Rice import interface_Rice
import itertools
from multiprocessing import Pool, Queue
import statsmodels.api as sm  # 协整
from scipy import sparse 
import time 
import talib as ta


class cluster_data(object):

    def __init__(self):
        self.isEmptyUse = True   # 是否情况记录
        # k线时间
        self.klineTypeList = ['1d', '5m', '15m', '60m']

        # 起始时间
        #self.startDate = public.getDate(diff=-self.testDays)  # 60天数据回测

        self.startDate = '2013-01-01'

        self.endDate = public.getDate()
        self.total_tablename = 'future_uniform'
        self.method = 'cluster_diff'
        self.uidKey = "%s_%s_" + self.method

    def empty(self):
        if self.isEmptyUse:
            Total = stock_uniform()
            Total.tablename = self.total_tablename
            Total.empty()

    def Pool(self):
            time0 = time.time()
            pool = Pool(processes=5)
            self.empty()

            Base = future_baseInfo()
            # 交易量大的，按价格排序, 类型:turple,第二位为夜盘收盘时间
            lists = Base.all(vol=120)

            pool.apply_async(self.start, (codes, time0, kt))

            for rs in list(itertools.combinations(lists, 2)):
                # 检查时间匹配
                codes = [rs[0][0], rs[1][0]]
                #if 'SC' not in codes: continue

                for kt in self.klineTypeList:
                    #self.start(codes, time0, kt)
                    try:
                        pool.apply_async(self.start, (codes, time0, kt))
                        pass

                    except Exception as e:
                        print(e)
                        continue
                #break
            pool.close()
            pool.join()

    cindex = 0
    def start(self, codes, time0, kt):
        print("子进程启动:", self.cindex, codes, kt, time.time() - time0)

        self.klineType = kt
        # 主力合约
        self.codes = codes
        self.mCodes = mCodes = [n + '88' for n in codes]
        self.baseInfo = {}
        self.Rice = interface_Rice()

        # 查询获得配置 - 费率和每手单量
        self.Base = future_baseInfo()
        for doc in self.Base.getInfo(codes):
            self.baseInfo[doc["code"] + '88'] = doc

        cs0, cs1 = self.baseInfo[mCodes[0]], self.baseInfo[mCodes[1]]

        # 子进程共享类
        self.Rice.setTimeArea(cs0["nightEnd"])
        self.Total = stock_uniform()
        self.Total.tablename = self.total_tablename

        if kt[-1]=='m':
            self.startDate = public.getDate(diff=-200)

        # 查询获得N分钟K线
        dfs = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=60)

        doc = self.total(dfs, kt)
        if doc is not None :
             self.Total.insert(doc)

    def total(self, dfs, kt):
        #uid = self.uidKey % ( '_'.join(self.codes), self.klineType[:-1])

        df0, df1 = dfs[self.mCodes[0]], dfs[self.mCodes[1]]

        df0 = df0.dropna(axis=0)
        df1 = df1.dropna(axis=0)
        if len(df0) != len(df1):
            #print('------------------change', kt, len(df0), len(df1))
            if kt[-1] == 'm':
                if len(df0) > len(df1):
                    df0 = df0[df0.index.isin(df1.index)]
                    df1 = df1[df1.index.isin(df0.index)]
                else:
                    df1 = df1[df1.index.isin(df0.index)]
                    df0 = df0[df0.index.isin(df1.index)]

            elif len(df0) > len(df1):
                df0 = df0[df1.index[0]:]
            else:
                df1 = df1[df0.index[0]:]

        print(len(df0), len(df1))

        c0, c1 = df0['close'], df1['close']
        # 交易量差距
        volrate = df0['volume'].mean() / df1['volume'].mean()

        # 涨跌一致性
        diff0 = c0.diff().apply(lambda x: 1 if x > 0  else -1 if x<0 else 0)
        diff1 = c1.diff().apply(lambda x: 1 if x > 0  else -1 if x<0 else 0)

        df0 = df0.dropna(axis=0)
        df1 = df1.dropna(axis=0)
        ss = ((diff0+diff1)).apply(lambda x: 1 if x!=0 else 0)
        delta = ss.sum() / len(ss)

        # 波动比
        s0 = ta.MA(c0,timeperiod=15).mean()
        sd0 = ta.STDDEV(c0, timeperiod=15, nbdev=1).mean()

        s1 = ta.MA(c1, timeperiod=15).mean()
        sd1 = ta.STDDEV(c1, timeperiod=15, nbdev=1).mean()
        std = sd0 * s1 / s0 / sd1
        #
        # 跌涨标准差
        dd0 = ta.STDDEV(c0.diff(), timeperiod=15, nbdev=1).mean()
        dd1 = ta.STDDEV(c1.diff(), timeperiod=15, nbdev=1).mean()
        diffstd = dd0 * s1 / s0 / dd1

        # 协整
        coint = sm.tsa.stattools.coint(c0, c1)

        # x相关性
        relative = per(c0, c1)

        doc = {
            "code": self.codes[0],
            "code1": self.codes[1],
            "kline": self.klineType,
            "relative": relative,
            "samecount": len(df0),
            "samerate": delta,
            "diffstd": diffstd,
            "coint_1": 0 if np.isnan(coint[0]) else coint[0],
            "coint": 0 if np.isnan(coint[1]) else coint[1],
            "std": std,
            "vol": volrate,
            "type": "future"
        }

        print(doc)
        # 按时间截取并调整
        print('kline load:', kt, self.codes, len(df0), len(df1))
        return doc

    def kmeans_test(self):
        """"
         执行聚类操作，并将结果保存回文档
       """
        time0 = time.time()
        obj = cluster_data()
        #return
        items = obj.get()
        logger.info("获取数据, 耗时: %s " % str(time.time()-time0))

        num = len(items)
        width = max(items[0].keys())+1

        ss = sparse.lil_matrix((num, width))
        i=0
        for item in items:
           for key in item.keys():
              if key!=0: ss[i,key] = item[key]
           i+=1

        random_state = 170
        c = KMeans(n_clusters= 4 , random_state=random_state)
        c.fit_predict(ss)
        clur = c.labels_.tolist()

        j=0
        for item in items:
            sql = obj.sql_update % (clur[j] , item[0])
            obj.update(sql)
            j += 1

        # 更新ad数据
        time.sleep(2)
        #obj.refresh()
        logger.info("聚类和更新数据, 耗时: %s" % str(time.time()-time0))


def refresh():
    obj = cluster_data()
    obj.refresh()   
    
def main():
    obj = cluster_data()
    obj.Pool()

    
if __name__=="__main__": 
        main()  