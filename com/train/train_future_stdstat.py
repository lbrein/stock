# -*- coding: utf-8 -*-
"""
Created on 2019-3-11
@author: lbrein

    计算单品种的 各种kline 下的 std 均值， 评估合理的 k线区间匹配

"""

from com.base.public import public, logger
import pandas as pd
import talib as ta
import numpy as np

from com.ctp.interface_pyctp import BaseInfo
from com.object.obj_entity import future_baseInfo
from com.object.mon_entity import mon_tick
from com.data.interface_Rice import interface_Rice
import time

# 回归方法
class future_stdstat(object):
    """

    """
    def __init__(self):
        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
        self.quickPeriodList = ['1m', '3m', '5m', '10m']
        self.periodList = ['15m', '30m', '60m']

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    def getStart(self, nt, ktype):
        key = nt + ' ' + ktype
        if nt in self.preDict:
            start = self.preDict[key]
        else:
            start = self.preDict[key] = self.Rice.getPreDate(period=ktype, num=self.count)
        return start


    def create(self):
        self.Rice = Rice = interface_Rice()
        self.Rice.setIndexList(self.indexCodeList)
        self.Base = Base =  future_baseInfo()
        self.Tick = mon_tick()

        codes = Base.getUsedMap(hasIndex=True)
        BI = BaseInfo(codes)
        mCodes = Rice.getMain(codes)
        period = 14
        self.count = count = 300
        self.preDict = {}
        self.exKtype = self.getKtype()
        self.renewQuicktype()

        docs, ex, trends = [],[], []
        for mcode in mCodes:
            code = BI.parseCode(mcode)
            nt = BI.att(code, 'nightEnd')
            Rice.setTimeArea(BI.att(code, 'nightEnd'))

            #if code in self.banCodeList: continue
            doc = {'code': code, 'qtype':''}
            std0v,maxDiss = 0, 0
            ptick = self.getTick(code)
            doc['qtype'] = self.quickPeriodList[-1]
            for ktype in self.quickPeriodList + self.periodList:
                start = self.getStart(nt, ktype)
                df0 = Rice.kline([mcode], period=ktype, start=start, pre=0)[mcode]

                close = df0["close"]
                # bull
                ma = ta.MA(close, timeperiod=period)
                std = ta.STDDEV(close, timeperiod=period, nbdev=1)
                std0 = std / ma * 100
                std0v = std0.mean()
                # ss = std0v if 0.3 < std0v < 0.5 else -1
                doc[ktype] = std0v
                tick = BI.att(code, 'tick_size') / ma.mean() * 100
                tick0 = ptick / ma.mean() * 100
                tick = tick0 if tick0 > tick else tick

                # 快速线大于5个tick
                if ktype in self.quickPeriodList:
                    sub = 3 * tick
                    if std0v > sub and doc['qtype']=='':
                        doc['qtype'] = ktype
                        doc['qtick'] = std0v

                    continue

                # sub = 0.2 if code in ['AU', 'AL', 'AG', 'I'] else 0.28
                sub = 8 * tick
                if sub > 0.4:  sub = 0.4
                if sub < 0.2:  sub = 0.2

                if sub < std0v and not code in ex:
                    doc['ktype'] = ktype
                    doc['stdv'] = std0v
                    doc['tick'] = tick
                    ex.append(code)

                # kdj顶点
                kd = 5
                kdjK, kdjD = ta.STOCH(df0["high"], df0["low"], close,
                                      fastk_period=kd, slowk_period=3, slowk_matype=1, slowd_period=3,
                                      slowd_matype=1)
                df0["kdj_d2"] = kdj_d2 = kdjK - kdjD
                df0["kdjm"] = kdj_d2 * kdj_d2.shift(1)
                df0["kdjm"] = df0.apply(lambda row: self.turn(row['kdjm'], row['kdj_d2'], 1), axis=1)

                ah= df0[df0["kdjm"]==1]
                diss = count / len(ah)
                doc['d'+ktype] = diss
                if diss > maxDiss:
                     doc['diss'] = diss
                     doc['dtype'] = ktype
                     maxDiss = diss

            # 最小trend
            if code not in ex:
                doc['ktype'] = '60m'
                doc['stdv'] = std0v

            minx = 100
            #I = self.periodList.index('30m')
            #line = 0.20 if doc['ktype'] in ['30m', '60m'] else 0.40 if doc['ktype'] in ['5m'] else 0.30
            line = 0.35

            #line = 0.3
            for k in  self.periodList:
                v  =  (doc['diss'] - doc['d'+k]) / doc['diss'] + abs(doc[k] - line) / 0.2
                if v < minx:
                    doc['mtype'] = k
                    minx = v

            if code in self.exKtype:
                if not doc['mtype'] == self.exKtype[code]:
                    print(code, doc['mtype'])
                    self.updateKtype(code, doc['mtype'])

                if doc['qtype']!='':
                    kc = self.klineCompare(doc['mtype'], doc['qtype'])
                    if kc > 4:
                        print(code, doc['qtype'])
                        self.updateQuicktype(code, doc['qtype'])

            docs.append(doc)
            #break

        df = pd.DataFrame(docs)
        file = Rice.basePath + 'std_%s_.csv' % (public.getDatetime(style='%Y%m%d_%H%M%S'))
        df.to_csv(file, columns=['code', 'mtype','qtype','qtick','ktype', 'stdv', 'tick']+

                                self.quickPeriodList + self.periodList + ['dtype', 'diss']+['d'+c for c in self.periodList] , index=0)

    def pool(self, func = None):
        print(1222)
        self.Rice = Rice = interface_Rice()
        self.Rice.setIndexList(self.indexCodeList)
        self.Base = Base = future_baseInfo()
        self.Tick = mon_tick()

        codes = Base.getUsedMap(hasIndex=True)
        BI = BaseInfo(codes)
        mCodes = Rice.getMain(codes)
        period = 14
        self.count = count = 300
        self.preDict = {}
        self.exKtype = self.getKtype()
        self.renewQuicktype()

        self.klines = self.quickPeriodList + self.periodList

        docs, ex, trends = [], [], []
        for mcode in mCodes:
            print(mcode)
            code = BI.parseCode(mcode)
            nt = BI.att(code, 'nightEnd')
            Rice.setTimeArea(BI.att(code, 'nightEnd'))

            doc, base = {"code": code},  1
            for ktype in self.klines:
                start = self.getStart(nt, ktype)
                df0 = Rice.kline([mcode], period=ktype, start=start, pre=0)[mcode]

                if func is not None:
                    res = func(df0, period)
                    if isinstance(res, dict):
                        doc.update({'kline': res})
                    else:
                        doc.update({'k_'+ ktype: res, 'r_'+ ktype: res/base})

                    if ktype == self.quickPeriodList[0]:
                        doc.update({'r_' + ktype: 1})
                        base = res

            docs.append(doc)

        if len(docs)>0:
           #columns = [ ('k_'+ k, 'r_'+ k) for k in (self.klines)]
           self.saveCsv(docs, 'atrCompare_%s_.csv')

    def stat_atr(self, df0, period):
        close = df0['close']
        df0['atr'] = ta.ATR(df0['high'], df0['low'], close, timeperiod=period)
        return df0['atr'].mean()


    def saveCsv(self, docs , fileStyle):
        df = pd.DataFrame(docs)
        file = self.Rice.basePath + fileStyle % (public.getDatetime(style='%Y%m%d_%H%M%S'))
        df.to_csv(file, index=0)
        print('csv saved:', file)

    def getTick(self, code):
        docs = self.Tick.getTick(code, count=1000)
        df = pd.DataFrame(docs)
        if 'p_l' in df.columns:
            return (df['p_l'] - df['p_h']).mean()
        else:
            return 0

    def klineCompare(self, a, b):
        return int(a.replace('m',''))/int(b.replace('m',''))


    def updateKtype(self,code,ktype):
        sql= "update %s set klinetype='%s' where code='%s'" % (self.Base.tablename, ktype, code)
        #print(sql)
        self.Base.update(sql)

    def renewQuicktype(self):
        sql = "update %s set quickkline ='' where code <> ''" % (self.Base.tablename)
        # print(sql)
        self.Base.update(sql)

    def updateQuicktype(self, code, ktype):
        sql = "update %s set quickkline ='%s' where code='%s'" % (self.Base.tablename, ktype, code)
        # print(sql)
        self.Base.update(sql)

    def getKtype(self):
        d = {}
        sql = "select code, klinetype from future_baseInfo where  isUsed = 1"
        for doc in self.Base.execSql(sql,isJson=False):
            d[doc[0]] = doc[1]
        return d

def main():
    action = {
        "stat": 0,
        "atrcompare": 1,
    }

    obj = future_stdstat()
    if action['stat']==1:
         obj.create()

    if action['atrcompare']==1:
         obj.pool(obj.stat_atr)


if __name__ == '__main__':
    main()



