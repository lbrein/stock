# -*- coding: utf-8 -*-
"""
Created on  2018-01-04 
@author:
      T+0

-
"""

from com.base.public import public
import pandas as pd
from com.data.interface_Rice import interface_Rice
import numpy as np
import talib as ta
import copy
import uuid
from com.object.obj_entity import stock_orderForm, stock_baseInfo, stock_record_t0, stock_t0_param
from multiprocessing import Pool

class data_stock_Rice(object):
    #
    filepath = "E:\lbrein\python\stock/file/"
    times = 15

    def __init__(self):
        self.period = '1m'
        self.startDate = '2019-02-11'
        self.endDate = '2019-06-12'
        self.timeperiod = 14

    def parseCode(self, code):
        res = str(code)
        tmp = copy.copy(res)
        if len(res) < 6:
           for i in range(6-len(res)):
                tmp = '0' + tmp
        return tmp

    def csv(self):
        dates = ['226', '319', '612']
        #dates = ['612']
        T0 = stock_record_t0()
        T0.empty()
        for d in dates:
            file = self.filepath + 'record_%s.csv' % d
            # 去除
            df1 = pd.read_csv(file, encoding='gb2312')

            df = df1[((df1['memo'] == '证券买入') | (df1['memo'] == '证券卖出')) & (abs(df1['volume']) >=100)]

            df['date'] = df['date'].apply(lambda d: public.parseTime(str(d),'%Y%m%d','%Y-%m-%d'))
            df['time'] = df['time'].apply(lambda d: public.parseTime(str(d), '%H:%M:%S', '%H:%M:%S'))
            df['datetime'] = df['date'] + ' ' + df['time']
            df['code'] = df['code'].apply(lambda d: self.parseCode(d))
            df['volume'] = df['volume'].apply(lambda x: round(x/100,0)*100)
            df['mode'] = df['volume'].apply(lambda d: 1 if d > 0 else -1)

            df['orderid'].fillna(0, inplace=True)
            df['orderid'] = df['orderid'].astype('int')

            df['batchid'] = ''
            df['income'] = 0.0

            df.sort_values(['code', 'datetime'], ascending=[True, True], inplace=True)
            df = df.reset_index(drop=True)

            df = self.setIncome1(df)
            df['isopen'].fillna(-4, inplace=True)
            df['isopen'] = df['isopen'].astype('int')


            file = self.filepath + 'filter_date_%s.csv' % (d)
            df.to_csv(file, index=0,columns=['code','name','datetime', 'mode', 'price','amount','volume', 'isopen',
                                             'income', 'cost','amount_c', 'count', 'vc'])
            #df.to_csv(file, index=0)

            print(df['income'].sum())
            pc = 1500
            if len(df) > pc:
                pages = len(df) // pc + 1
                for p in range(pages):
                    s, e = p * pc , (p+1) * pc
                    if e > len(df): e - len(df)
                    docs = df.iloc[s:e].to_dict(orient='records')
                    #docs = [d for d in docs if str(d['batchid']).strip()!='']
                    print(s, e)
                    T0.insertAll(docs)
            else:
                docs = df.to_dict(orient='records')
                T0.insertAll(docs)

    def setIncome1(self, df):
        i0, skip = 0, 0
        for i in range(len(df)):
            if i0 < i < (i0 + skip): continue
            row = df.iloc[i]
            if (i+1 < len(df) and row['code']!= df.loc[i+1,'code']) or ('2019-03-27 09:29:00' < str(row['datetime']) < '2019-03-27 09:39:55'):
                continue

            i0, skip =i, 1
            for p in range(2, self.times):
                if i+p >len(df): break
                # 时差超过24小时
                if public.timeDiff(str(df.loc[(i + p - 1), 'datetime']), str(row['datetime']))/3600 > 24:
                    break

                vc = df.loc[i:(i + p - 1), 'volume'].sum()
                vc1 = df.loc[i:(i + p ), 'volume'].sum()

                if abs(vc) <=100 and vc1!=0 and abs(row['volume']) > 100:

                    batchid = uuid.uuid1()
                    mode0, amount, vol = row['mode'], 0, 0
                    df.loc[i, 'count'] = p
                    df.loc[i, 'vc'] = vc

                    for j in range(p):
                        rowp = df.iloc[i + j]
                        #print(rowp)
                        cost = rowp['fee'] + rowp['tax'] + rowp['export']
                        df.loc[(i+j), 'batchid'] = batchid

                        # fanzhuan
                        if (vol + rowp['volume']) * mode0 >= 0:
                            if rowp['mode'] == mode0:
                                df.loc[i + j, 'income'] = - cost
                                df.loc[i + j, 'isopen'] = 1
                                amount += rowp['amount']

                            else:
                                #测
                                delta = 0 if ((vol + rowp['volume']) == vc and vc!=0) else 0
                                py = abs(amount * (rowp['volume'] - delta)/vol)
                                df.loc[i + j, 'income'] = -rowp['mode'] * (rowp['amount'] - py) - cost
                                df.loc[i + j, 'isopen'] = 0
                                amount += -py

                        else:
                            df.loc[i + j, 'isopen'] = 1
                            py = abs(rowp['amount'] * vol /rowp['volume'])
                            df.loc[i + j, 'income'] = -rowp['mode'] * (py - amount) - cost
                            amount = rowp['amount'] * (1 + vol / rowp['volume'])
                            mode0 = -mode0

                        df.loc[i + j, 'cost'] = cost
                        df.loc[i + j, 'amount_c'] = amount
                        vol += rowp['volume']

                    # 跳跃过P
                    i0 = i
                    skip = p
                    break
        return df

    # 分段布林策略
    def empty(self):
        Record = stock_t0_param()
        #Record.tablename = self.recordTableName
        Record.empty()

    # 查询参数写入到param表
    def pool(self):
        self.empty()

        pool = Pool(processes=5)
        T0 = stock_record_t0()
        lists = T0.getCodes1()
        print(len(lists))

        for k in range(0, len(lists)):
            c = lists[k]
            #if '600230.XSHG'!=c[0]:continue

            #self.start(c)
            #break
            try:
                pool.apply_async(self.start, (c,))
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

    def start(self, cs):
        self.Rice = interface_Rice()
        self.Record = stock_record_t0()
        self.Param = stock_t0_param()
        code, self.startDate,self.endDate = cs[0], str(cs[1]), str(cs[2])
        self.code= code
        print(code, self.startDate, self.endDate)
        res = self.Rice.kline([self.code], period=self.period, start=self.startDate, end=self.endDate, pre=1, type='stock')
        df = res[code]
        #self.klineColumns = df.columns

        tks = self.Rice.kline([code], period='tick', start=self.startDate, end=self.endDate, pre=2, type='stock')
        tk = tks[code]
        #print(tk)

        tk['datetime'] = tk.index
        df['datetime'] = df.index


        df, tk = self.total(df, tk)
        docs = self.saveStage(df, tk)
        print('process %s end: %s ' % (code, len(docs)))
        if len(docs)>0:
            self.Param.insertAll(docs)

    def curMA(self, row, df0):
        e = row['datetime']
        #print(str(e)[:10])
        s = public.str_date(public.getDate(diff=-1, start=str(e)[:10]) + ' 09:30:00')
        df = df0[(df0['datetime'] >= s) & (df0['datetime'] <=e)]

        return (df['close'] * df['volume']).sum() / df['volume'].sum() if df['volume'].sum() != 0 else 0

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

    def tickMode(self, tick):
        if np.isnan(tick['raise']): return 0
        mode = 0
        if tick['raise'] != 0:
            mode = int(np.sign(tick['raise']))
        elif tick['a1'] != tick['pa1']:
            mode = 1 if tick['last'] == tick['pa1'] else -1 if tick['last'] == tick['pb1'] else 0
        else:
            mode = 1 if tick['last'] == tick['a1'] else -1 if tick['last'] == tick['b1'] else 0
        return mode

    def total(self, df0, tk, period=14):
        # 计算参数
        close = df0["close"]

        df0["Curma"] = df0.apply(lambda row: self.curMA(row, df0), axis=1)

        df0["ma"] = ma = ta.MA(close, timeperiod=period)
        df0["std"] = std = ta.STDDEV(close, timeperiod=period, nbdev=1)
        df0['mv'] = ta.MA(df0['volume'], timeperiod=period)
        df0['volc120'] = df0['volume'] / ta.MA(df0['volume'], timeperiod=120)
        df0['volc'] = df0['volume'] / df0['mv']
        df0['bias'] = (close - ma) / ma * 1000
        df0['min5'], df0['max5'] = ta.MINMAX(close, timeperiod=10)

        # tick
        tk['vol'] = tk['volume'] - tk['volume'].shift(1)
        tk = tk[tk['vol'] > 0]

        # print(tk)
        tk['raise'] = tk['last'] - tk['last'].shift(1)
        tk['sb_diff'] = tk['a1'] - tk['b1']

        #print(tk)

        for k in ['a', 'b']:
            tk['r' + k] = tk[k + '1'] - tk[k + '1'].shift(1)
            tk['p%s1' % k] = tk[k + '1'].shift(1)

            tk['v%s_5' % k ] = sum([tk[k + str(i)+'_v'] for i in range(1, 6)])
            #tk['p%s_diff' % k] = tk[k+'5']-tk[k+'1']
            rrr = tk['r' + k].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            for n in [3]:
                key = 'sbm%s%s' % (k, str(n))
                tk[key + '_v'] = ta.SUM(tk['r' + k], timeperiod=n)
                tk[key] = ta.SUM(rrr, timeperiod=n)

        # print(tk.iloc[-1])

        tk['mode'] = tk.apply(lambda row: self.tickMode(row), axis=1)
        tk['datetime'] = tk['datetime'].apply(lambda x: str(x)[:str(x).find(".")] if str(x).find(".") > -1 else x)

        #tk['dd5'] = tk['datetime'].shift(5).fillna('2019-01-01 00:00:00')
        #tk['interval'] = tk.apply(lambda row: public.timeDiff(str(row['datetime']), str(row['dd5'])), axis=1)
        tk['modem5'] = ta.SUM(tk['mode'], timeperiod=5)
        tk['vol_t5'] = ta.SUM(tk['vol'] * tk['mode'], timeperiod=5)

        #tk['modem3'] = ta.SUM(tk['mode'], timeperiod=3)
        #tk['vol_t3'] = ta.SUM(tk['vol'] * tk['mode'], timeperiod=3)

        return df0, tk

    def saveStage(self, df, tk):
        print(self.code[:6])
        records = self.Record.getRecord(self.code[:6])
        for doc in records:
            tt = doc['datetime']
            row = df[:tt].iloc[-1]
            cur = tk[:tt].iloc[-1]
            #print(tt, row['datetime'], cur['datetime'])
            doc.update(row)
            doc.update(cur)
            doc['price'] = doc['last']
            doc['parentid'] = doc['id']
            doc['interval'] = int(doc['interval'])
            doc['modem3'] = int(doc['modem3'])
            doc['modem5'] = int(doc['modem5'])

        return records

    def getTick(self, codes, period='1d'):
        pass

    def update_stockBase(self):
        # 更新stockBase
        Base = stock_baseInfo()
        Rice = interface_Rice()

        codes = Base.getAllCodes()
        #Base.empty()

        df0 = Rice.allCodes(type='CS')
        df = df0[~df0['order_book_id'].isin(codes)]
        #print(df)

        df['code'] = df['order_book_id'].apply(lambda x: x[:x.find('.')])
        s = Rice.index_compose('000980.XSHG')
        df['is50'] = df['order_book_id'].apply(lambda x: 1 if x in s else 0)
        df['isST'] = df['symbol'].apply(lambda x: 1 if x.find('*ST') > -1 else 0)

        print(len(df))
        Base.insertAll(df.to_dict('records'))


def main():
    action = {
        "csv": 1,
        "update": 0,
        "param": 0
    }

    obj = data_stock_Rice()
    if action["csv"] == 1:
       obj.csv()

    if action["update"] == 1:
       obj.update_stockBase()

    if action["param"] == 1:
       obj.pool()

if __name__ == '__main__':
    main()
    #test()
