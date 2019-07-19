# -*- coding: utf-8 -*-
"""
Created on  2018-01-04 
@author:
      T+0

-
"""

from com.base.public import public, public_basePath
import pandas as pd
from com.data.interface_Rice import interface_Rice
from com.ctp.interface_pyctp import BaseInfo
import numpy as np
import talib as ta
import copy
import uuid
from com.object.obj_entity import future_orderForm, future_baseInfo
from multiprocessing import Pool
import json
import itertools

class data_future_T0(object):
    #
    filepath = "E:\lbrein\python\stock/file/"
    times = 15

    def __init__(self):
        self.period = '1m'
        self.startDate = '2019-02-11'
        self.endDate = '2019-06-12'
        self.timeperiod = 14
        self.T0 = future_orderForm()
        self.T0.tablename = 'future_orderForm_sg'
        self.indexList = ['IH','IF','IC']

    def parseCode(self, code):
        res = str(code)
        tmp = copy.copy(res)
        if len(res) < 6:
           for i in range(6-len(res)):
                tmp = '0' + tmp
        return tmp

    def csv(self):
        dates = ['226']
        BI = BaseInfo([])
        self.T0.empty()

        for d in dates:
            file = self.filepath + 'future_sg.csv'
            # 去除
            df = pd.read_csv(file, encoding='gb2312')
            df['createdate'] = df['date'] + ' ' + df['time']
            df['name'] = df['code'].apply(lambda x: BI.parseCode(x))

            df['vol'] = df.apply(lambda row: (BI.att(row['name'], 'contract_multiplier') * row['hands']), axis=1)

            df.sort_values(['code', 'createdate'], ascending=[True, True], inplace=True)
            df = df.reset_index(drop=True)
            df = self.setIncome1(df)

            print(df['income'].sum())

            pc = 1500
            df.sort_values(['createdate'], ascending=[True], inplace=True)
            df = df.reset_index(drop=True)

            if len(df) > pc:
                pages = len(df) // pc + 1
                for p in range(pages):
                    s, e = p * pc , (p+1) * pc
                    if e > len(df): e - len(df)
                    docs = df.iloc[s:e].to_dict(orient='records')
                    print(s, e)
                    self.T0.insertAll(docs)

            else:
                docs = df.to_dict(orient='records')
                self.T0.insertAll(docs)


    def setIncome1(self, df):
        preCode, batchid, total , prePrice = '', '', 0, 0

        for i in range(len(df)):
            row = df.iloc[i]
            if preCode != row['code']:
                total, batchid = 0, ''

            if row['isopen'] == 1 and total == 0:
                # 第一次开仓
                total = row['mode'] * row['hands']
                df.loc[i, 'batchid'] = batchid = uuid.uuid1()

            elif batchid != '':
                total += row['mode'] * row['hands']
                df.loc[i, 'batchid'] = batchid

            df.loc[i, 'ini_hands'] = total
            preCode = row['code']

        return df

    def category(self):
        cfg = public_basePath + "/com/data/future_category.json"
        with open(cfg, 'r', encoding='utf-8') as f:
            params = json.load(f)

        keys = copy.deepcopy(params).keys()
        d = {"indexList": self.indexList}
        for cat in keys:
            d[cat + 'List'] = []
            for s in params[cat]:
                if s == 'index': continue
                d[cat + 'List'] += params[cat][s]

        return d

    def combin_date(self):
        self.Rice = interface_Rice()

        docs = self.T0.sg_get(diss='30m')
        df = pd.DataFrame(docs)

        df['detail'] = df.apply(lambda r :'(%s,%s,%s)' % (str(r['mode']),r['isopen'], int(r['vol'])), axis=1)
        df['detail_1'] = df['detail'].shift(1)
        df['date_1'] = df['date'].shift(1)
        df['date_-1'] = df['date'].shift(-1)
        df['dup'] = df.apply(lambda r:  1 if r['date'] == r['date_-1'] else 0, axis=1)
        df['detail'] = df.apply(lambda r:  r['detail_1'] +'\n'+ r['detail'] if r['date'] == r['date_1'] else r['detail'], axis=1)

        df0 = df[df['dup']==0].loc[:, ["code", "name", "mode", "date", "detail"]]

        #按日期-code
        #self.byCode(df0)

        # 按code类别
        #self.byCat(df0)

        # 按组合类别
        self.byPair(df0)

    def toCvs(self, df, key):
        file = self.Rice.basePath + 'sg_%s_%s.csv' % (key, public.getDate())
        print('to cvs:', file)
        df.to_csv(file, index=1)

    def byPair(self, df0):
        lists = df0.loc[:, 'name'].unique()

        i = 0
        df2 = pd.DataFrame()

        for rs in list(itertools.combinations(lists, 2)):
            df2.loc[i, 'pair'] = '%s_%s' % (rs[0],rs[1])

            df2.loc[i, 'count'] = len(df0[df0['name'].isin(rs)])

        pass

    def byPeriod(self, df0):

        pass

    # 按日期-code分类报表
    def byCode(self, df0):
        df1 = pd.DataFrame(index=df0['date'].unique())
        for a in df0.loc[:, 'code'].unique():
            sub = copy.deepcopy(df0[df0['code'] == a]).set_index('date')
            for d in sub.index:
                df1.loc[d, a] = sub.loc[d, 'detail']

        df1.sort_index(inplace=True)
        self.toCvs(df1, 'combinDate')

    # 按类别组合
    def byCat(self,df0):
        # 分类, 添加未归类分类
        cats = self.category()
        cats['otherList'] = []
        for a in df0.loc[:, 'name'].unique():
            isIn = False
            for cat in cats:
                if a in cats[cat]:
                    isIn = True
                    break
            if not isIn:
                cats['otherList'].append(a)

        df2 = pd.DataFrame(index=df0['date'].unique())
        for d in df0.loc[:, 'date'].unique():
            sub = copy.deepcopy(df0[df0['date'] == d])
            sub['com'] = sub.apply(lambda r: (str(r['detail'])+' '+r['code']), axis=1)
            for cat in cats:
                df2.loc[d, cat] = "\n".join(sub[sub['name'].isin(cats[cat])].loc[:, 'com'])

        df2.sort_index(inplace=True)
        self.toCvs(df2, 'combinCode')


def main():
    action = {
        "csv": 0,
        "date":1
    }

    obj = data_future_T0()
    if action["csv"] == 1:
       obj.csv()

    if action["date"] == 1:
       obj.combin_date()


if __name__ == '__main__':
    main()
    #test()
