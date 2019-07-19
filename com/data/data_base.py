# -*- coding: utf-8 -*-

""" 
Created on  2018-01-04 
@author: lbrein
       ---  FUTu 信息接口 ---      
 
"""

from com.base.public import public, config_ini, logger
#from futuquant import *
from com.object.obj_entity import plate, stockdetail, tradeDate

#import tushare as TS
import time
import sys

class Stock:

    def isTradeDate(self):
        Ta = tradeDate()
        return Ta.isTradeDate()

    def isTradeTime(self, market='SH'):
        d = public.getDate()
        t = public.getDatetime()

        timeMap = [['09:30', '11:30'], ['13:00', '15:00']]

        if market == 'HK':
            timeMap = [['09:30', '12:00'], ['13:00', '16:00']]

        for lim in timeMap:
            if t >= d + " " + lim[0] and t <= d + " " + lim[1]:
                return [True, True]

        # over
        if t > d+" "+ timeMap[1][1]:
            return [False, False]
        else:
            return [False, True]


# futuAPI接口
"""
class fytuObject(Stock):

    def __init__(self):
        self.host = config_ini.get("futu.host")
        print(self.host)

        self.port = int(config_ini.get("futu.port"))
        self.Quant = None
        self.reconnect()

    def reconnect(self):
        if not self.Quant:
            try:
                self.Quant = OpenQuoteContext(host=self.host,  port=self.port)
                print(self.host + "已连接")
            except:
                print("链接错误")

    def close(self):
        self.Quant.close()
        self.Quant = None

    def isConnect(self):
        return self.Quant

    def isTradeDate(self, market):
        # 日期匹配
        d = public.getDate()
        start, end = public.getDate(diff=-3), public.getDate(diff=3)
        days = self.get_trading_days(market, start_date=start, end_date=end)
        if not d in days: return False
        return True

    def tradeDate(self, market='SH'):
        T = tradeDate()
        start, end = public.getDate(diff=-3 * 365), public.getDate(diff=3 * 365)
        days = self.get_trading_days(market, start_date=start, end_date=end)
        docs = [{"sdate": d} for d in days]
        T.insertAll(docs)

    # 检查是否在交易时间
    def isTradeDateTime(self, market):
        # 日期匹配
        d = public.getDate()
        start, end = public.getDate(diff=-3), public.getDate(diff=3)
        days = self.get_trading_days(market, start_date=start, end_date=end)
        if not d in days: return False

        # 时间匹配
        t = public.getDatetime()
        timeMap = [['09:30', '11:30'], ['13:00', '15:00']]
        if market == 'HK':
            timeMap = [['09:30', '12:00'], ['13:00', '16:00']]

        for lim in timeMap:
            if t >= d + " " + lim[0] and t <= d + " " + lim[1]:
                return True

        return False

    def getStockInfo(self, market="HK", stock_type="STOCK"):
        ret, df = self.Quant.get_stock_basicinfo(market=market, stock_type=stock_type)
        return df

        # 获得历史K线

    def get_history_kline(self, code, start=None, end=None, year=1, ktype='K_DAY'):
        if start is None:
            start = public.getDate(diff=-365 * year)

        ret, df = self.Quant.get_history_kline(code, start=start, end=end, ktype=ktype, autype='qfq',
                                               fields=[KL_FIELD.ALL])
        return df

    def get_history_kline_m(self, code, start=None, end=None, days=1, ktype='K_15M'):
        if start is None:
            start = public.getDatetime(diff=-1 * days)

        ret, df = self.Quant.get_history_kline(code, start=start, end=end, ktype=ktype, autype='qfq',
                                               fields=[KL_FIELD.ALL])
        return df

    def get_cur_kline(self, code, num=1000, ktype='K_DAY', autype='qfq'):
        self.subscribe([code], ktype)
        ret, df = self.Quant.get_cur_kline(code, num, ktype=ktype, autype=autype)
        return df

    def get_trading_days(self, market, start_date=None, end_date=None):
        ret_code, ret_data = self.Quant.get_trading_days(market, start_date=start_date, end_date=end_date)
        return ret_data


    def getLastWarren(self):
        df = self.getStockInfo("HK", "WARRANT")
        list_date = df["listing_date"].tolist()
        nd = public.getDate()
        for d in list_date:
            if d < nd:
                nd = d
        return nd

    def getWarrantOwners(self):
        df = self.getStockInfo("HK", "WARRANT")
        res = []
        owners_code = df["owner_stock_code"].tolist()
        for o in owners_code:
            if o and not o in res:
                res.append(o)
        return res

        # 深度数据

    def get_order_book(self, code):
        # 订阅
        self.reconnect()
        self.subscribe([code], "ORDER_BOOK")
        # 获得最新列表
        ret_status, ret_data = self.Quant.get_order_book(code)
        if ret_status == RET_ERROR:
            print(ret_data)
            return None
        return ret_data

        # 分时数据

    def get_rt_data(self, code):
        # 订阅
        self.subscribe([code], "RT_DATA")
        # 获得最新列表
        ret_status, ret_data = self.Quant.get_rt_data(code)
        if ret_status == RET_ERROR:
            print(ret_data)
            return None
        return ret_data

        # 及时报价
    def get_cur_quote(self, codelist):
        # 订阅
        self.subscribe(codelist, "QUOTE")
        # 获得最新列表
        ret_status, ret_data = self.Quant.get_stock_quote(codelist)
        if ret_status == RET_ERROR:
            print(ret_data)
            return None
        return ret_data

        # 订阅数据

    def subscribe(self, codelist, stype):
        ret_status, ret_data = self.Quant.query_subscription()
        if ret_status == RET_ERROR:
            exit()

        lst = []
        if stype in ret_data:
            lst = ret_data[stype]

            # 添加订阅
        for code in codelist:
            if code in lst: continue
            self.Quant.subscribe(code, stype)

    # 清除订阅
    def unsubscribe(self, codelist, stype):
        # 延时1分钟执行
        time.delay(60)

        # 获得所有订阅列表
        ret_status, ret_data = self.Quant.query_subscription()
        if ret_status == RET_ERROR:
            exit()

        lst = []
        if stype in ret_data:
            lst = ret_data[stype]

            # 添加订阅
        for code in codelist:
            if code in lst: continue
            self.Quant.unsubscribe(code, stype)

    # 订阅并获取及时行情数据
    def getSnap(self, stock_code_list):
        ret_code, ret_data = self.Quant.get_market_snapshot(stock_code_list)

        if ret_code != 0:
            print(ret_data)

        return ret_data

class stockBase:
    def __init__(self):
        self.Futu = fytuObject()

    # 基础类别表
    def getClass(self):
        Sc = self.Futu.Quant
        Plate = plate()
        ms = ['HK', 'SH', 'SZ']
        ps = ["INDUSTRY", "REGION", "CONCEPT"]
        for m in ms:
            for p in ps:
                ret, df = Sc.get_plate_list(m, p)
                df['market'], df['class'] = m, p
                docs = df.to_dict(orient='records')
                for doc in docs:
                    doc["code"] = doc["code"][3:]
                # print(docs)
                Plate.insertAll(docs)
        self.Futu.close()

    # 沪深资产信息表
    def getBasic(self):
        df = TS.get_stock_basics()
        df['code'] = df.index
        df['timeToMarket'] = df['timeToMarket'].replace('0', '19000101')
        ST = stockdetail()
        docs = df.to_dict(orient='records')
        t = 0
        for pgs in public.eachPage(docs, 1000):
            t += len(pgs)
            if not ST.insertAll(pgs):
                break
        print(t)

    # 保持链接
    def keepLink(self):
        try:
            df = self.Futu.getStockInfo()
            if not df.empty:
                logger.info("FutuLink true")
        except:
            logger.info("FutuLink false")
            sys.exit()

        self.Futu.close()
"""
def main():
    pass

def test():
    obj = Stock()
    print(obj.isTradeTime())

if __name__ == '__main__':
    main()
