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
from multiprocessing import Pool
import statsmodels.api as sm  # 协整
from scipy import sparse 
import time 
import talib as ta
from multiprocessing.connection import Listener,  Client
from multiprocessing.managers import BaseManager

# 回归方法
class MyManager(BaseManager):
    pass

MyManager.register('interface_Rice', interface_Rice)
MyManager.register('future_baseInfo', future_baseInfo)

def Manager2():
    m = MyManager()
    m.start()
    return m

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

        self.address = ('localhost', 6000)
        self.authkey = b'ctpConnect'

    def empty(self):
        if self.isEmptyUse:
            Total = stock_uniform()
            Total.tablename = self.total_tablename
            Total.empty()

    def listen(self, main):
        listener = Listener(self.address, authkey=self.authkey)
        Total = stock_uniform()
        Total.tablename = self.total_tablename

        while True:
            conn = listener.accept()
            data = conn.recv()

            if 'end' in data and data['end']==1:
                print('------------end-------------')
                conn.close()
                break

            result=b'0'
            try:
                Total.insert(data)
                result = b'1'
                conn.send_bytes(result)
            except Exception as e:
                print(e)
                conn.send_bytes(result)

            finally:
                conn.close()

    def closeListen(self, tt):
        conn = Client(self.address, authkey=self.authkey)
        conn.send({'end':1, 'a':tt})
        conn.close()

    def Pool(self):
            time0 = time.time()
            pool = Pool(processes=5)
            self.empty()
            share = Manager2()

            Base = share.future_baseInfo()
            # 交易量大的，按价格排序, 类型:turple,第二位为夜盘收盘时间
            lists = Base.all(vol=120)
            print(len(lists))
            Rice = share.interface_Rice()
            #Base1 = share.future_baseInfo()

            pool.apply_async(self.listen, (1,))

            for rs in list(itertools.combinations(lists, 2)):
                # 检查时间匹配
                codes = [rs[0][0], rs[1][0]]
                #if 'SC' not in codes: continue
                for kt in self.klineTypeList:
                    #self.start(codes, time0, kt, False, Rice, None)
                    try:
                        pool.apply_async(self.start, (codes, time0, kt, True, Rice, None))
                        pass

                    except Exception as e:
                        print(e)
                        continue

            while True:
                num = len(pool._cache)
                if num < 2:
                    pool.apply_async(self.closeListen, (1,))
                    break
                time.sleep(1)

            pool.close()
            pool.join()


    cindex = 0
    def start(self, codes, time0, kt, isPool=True, Rice=None, Base = None):
        self.klineType = kt
        # 主力合约
        self.codes = codes
        self.mCodes = mCodes = [n + '88' for n in codes]
        self.baseInfo = {}
        self.Rice = interface_Rice() if Rice is None else Rice

        # 查询获得配置 - 费率和每手单量
        self.Base = future_baseInfo() if Base is None else Base
        #self.Base = Base
        #print(codes, self.Base.tablename)

        for doc in self.Base.getInfo(codes):
            self.baseInfo[doc["code"] + '88'] = doc

        cs0, cs1 = self.baseInfo[mCodes[0]], self.baseInfo[mCodes[1]]

        # 子进程共享类
        self.Rice.setTimeArea(cs0["nightEnd"])

        if kt[-1]=='m':
            self.startDate = public.getDate(diff=-200)

        # 查询获得N分钟K线
        dfs = self.Rice.kline(mCodes, period=self.klineType, start=self.startDate, end=self.endDate, pre=60)

        doc = self.total(dfs, kt)
        if doc is not None:
            if isPool:
                conn = Client(self.address, authkey=self.authkey)
                conn.send(doc)
                #print(conn.recv_bytes())
                conn.close()
                #Total.insert(doc)
            else:
                Total = stock_uniform()
                Total.tablename = self.total_tablename
                Total.insert(doc)

        print("子进程启动:", self.cindex, codes, kt, time.time() - time0)

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

        # 3根一致性
        diff0_3, diff1_3 = ta.SUM(diff0, timeperiod=3), ta.SUM(diff1, timeperiod=3)
        ss3 = ((diff0_3 - diff1_3)).apply(lambda x: 1 if x == 0 else 0)
        delta3= ss3.sum()/len(ss3)

        # 波动比
        s0 = ta.MA(c0,timeperiod=15).mean()
        s1 = ta.MA(c1, timeperiod=15).mean()

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
            "samerate3": delta3,
            "diffstd": diffstd,
            "coint_1": 0 if np.isnan(coint[0]) else coint[0],
            "coint": 0 if np.isnan(coint[1]) else coint[1],
            "std": 0,
            "vol": volrate
        }

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