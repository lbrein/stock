# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

        sql数据库映射类
"""

from com.base.public import baseSql, public
import time
import re
import pandas as pd


# stock基础类
class baseInfo(baseSql):
    tablename = "baseinfo"
    up_struct = keylist = ['code', 'name', 'market', 'listing_date', 'stocktype']
    Chinese_List = ['name']


# 基金交易记录
class ETFRecord(baseSql):
    tablename = "ETFRecord"
    up_struct = keylist = ['code', 'returnid', 'type', 'price', 'volume', 'value', 'createtime']


# maticIndex
class maticIndex(baseSql):
    tablename = "maticIndex"
    up_struct = keylist = ['date', 'indexNo']

    def getIndex(self):
        unit = 3600 * 24
        date = int(round(time.time() / unit, 0) * unit)

        # 查询
        sql = "select date, indexNo from %s where date = %s " % (self.tablename, str(date))
        res = self.execOne(sql)
        if res:
            return date + int(res["indexNo"])

        else:
            # 无日期则添加新日期
            sql = "insert into %s (%s) values(%s, 0) " % (self.tablename, ",".join(self.keylist), str(date))
            self.update(sql)
            return date

    def setIndex(self, index):

        unit = 3600 * 24
        date = int(round(time.time() / unit, 0) * unit)
        # 更新
        sql = "update %s set indexNo = %s  where date = %s " % (self.tablename, str(index - date), str(date))
        self.update(sql)

    # 股票账户明细


class account(baseSql):
    tablename = "account"
    up_struct = ['account', 'code', 'price', 'volume', 'total', 'modifytime', 'stage', 'status']

    sql_get = """
         select code, volume, total from account where account = '%s' 
     """

    def get(self, account):
        self.keylist = ['code', 'volume', 'total']
        return self.execSql(self.sql_get % account)


class stockdetail(baseSql):
    tablename = "stockdetail"
    up_struct = ['code', 'name', 'industry', 'area', 'pe', 'outstanding', 'totals',
                 'totalAssets', 'liquidAssets', 'fixedAssets', 'reserved',
                 'reservedPerShare', 'esp', 'bvps', 'pb', 'timeToMarket', 'undp',
                 'perundp', 'rev', 'profit', 'gpr', 'npr', 'holders']
    Chinese_List = ['name', 'industry', 'area']

    sql_ind = """
        select *   
         from 
         (SELECT a.code , a.industry, a.timeToMarket,
          row_number() over ( partition by industry  order by a.totals * b.sclose desc) rn 
          from stockDetail a  
          LEFT JOIN history b on a.code = b.code and b.sdate = '2018-04-13') aa 
          where aa.rn < 3 
          order by industry
       """

    def getIndu2(self):
        self.keylist = ['code', 'industry', 'startdate', 'rn']
        for doc in self.execSql(self.sql_ind):
            yield doc


# stock基础类
class plate(baseSql):
    tablename = "plate"
    up_struct = keylist = ['code', 'plate_name', 'plate_id', 'market', 'class']
    Chinese_List = ['plate_name']


# stock基础类
class tradeDate(baseSql):
    tablename = "tradeDate"
    up_struct = keylist = ['sdate']

    def isTradeDate(self):
        d = public.getDate()
        sql = "select sdate from %s where sdate = '%s'" % (self.tablename, d)
        res = self.execOne(sql)
        return res != {}


# 新浪每3秒一次的510050数据记录
class sh510050(baseSql):
    tablename = "sh510050"
    # https://hq.sinajs.cn/list=s_sh510050
    up_struct = keylist = ['code', 'price', 'range', 'rate', 'amount', 'volume', 'datetime']


# sh50_daily 完整历史数据
class sh50_daily(baseSql):
    tablename = "sh50_daily"
    up_struct = keylist = ["code", "date", "pre_close", "open", "high", "low", "close", "volume", "amt", "dealnum",
                           "chg", "pct_chg", "swing", "vwap", "oi", "oi_chg", "adjfactor"]


# sh50_daily  只包含close的历史数据
class sh50_daily_1(baseSql):
    tablename = "sh50_daily_1"
    up_struct = ["code", "date", "close"]
    keylist = ['code', 'date', 'close', 'expireDay', 'owner_price', 'exe_price', 'mode', 'range']
    sql = """
            select  a.code , a.[date], a.[close], a.expireDay, a.owner_price, b.exe_price, b.exe_mode , c.range_244  
            from sh50_daily_1 a 
            left join sh50_extend_1 b on a.code = b.code 
            left join s_510050 c on a.date = c.date  
            where 1=1 and a.code = '%s'  
          """

    sql1 = """
            select  %s      
            from sh50_daily_1  
            where 1=1 and code = '%s'  
     
    """

    def getByCode(self, code=None):
        if code:
            return self.df(self.sql % code)

    def getAll(self):
        obj = sh50_extend_1()
        for code in obj.getCodes():
            yield self.df(self.sql % code)

    def getData(self, code=None, keylist=None):
        if keylist:
            self.keylist = keylist
        else:
            self.keylist = ['code', 'date', 'close', 'expireDay', 'owner_price', 'bs_price', 'calc_diss']

        strTitles = ",".join(self.keylist)
        return self.df(self.sql1 % (strTitles, code))


# sh50_daily 中间表，用于程序计算结果的更新
# 因正太分布函数无 sqlserver 对应函数，只能通过python程序计算
class sh50_daily_tmp(baseSql):
    tablename = "sh50_daily_tmp"
    up_struct = keylist = ["code", "date", "bsPrice"]

    sql = """
        UPDATE sh50_daily_1 
        set bsPrice = b.bsPrice 
        from sh50_daily_1 a 
        LEFT JOIN sh50_daily_tmp b 
        on a.code = b.code and a.date= b.date 
        where a.code = b.code 
        
    """

    def updateBs(self):
        self.update(self.sql)


class sh50_extend(baseSql):
    tablename = "sh50_extend"
    up_struct = keylist = ['code', 'us_code', 'underlyingwindcode', 'us_name', 'us_type', 'exe_mode', 'exe_type',
                           'exe_price'
        , 'maint_margin', 'exe_ratio', 'totaltm', 'startdate', 'lasttradingdate', 'exe_enddate',
                           'settlementmethod']

    Chinese_List = ['exe_mode', 'exe_type', 'settlementmethod']


# sh50 期权普通数据导入
class sh50_extend_1(baseSql):
    tablename = "sh50_extend_1"
    up_struct = ['code', 'us_name', 'exe_mode', 'exe_price', 'exe_ratio', 'exe_enddate']
    Chinese_List = ['us_name']
    sql = "select code from %s"

    def getCodes(self):
        for res in self.execSql(self.sql % self.tablename, isJson=False):
            yield res[0]


class s_510050(baseSql):
    tablename = "s_510050"
    up_struct = keylist = ['date', 'open', 'high', 'low', 'close', 'volume', 'fq_ratio',
                           'qfq_ratio', 'nfq', 'qfq', 'price_2016', 'price_2017', 'range',
                           'range_244']
    Chinese_List = ['us_name']


#
class sh50_spy(baseSql):
    tablename = "sh50_spy"
    up_struct = keylist = ['code', 'code_1', 'price_bid', 'price_ask', 'price_owner', 'remainDay', 'power',
                           'currentTime', 'result', 'mode', 'result1']

    sql_last = """
        select %s from
        (select *,
    	row_number() over (partition by code order by currentTime desc) rn
        from sh50_spy) a
        where a.rn = 1
        order by a.code
    """

# option
class option_tmp(baseSql):
    tablename = "option_tmp"
    up_struct = keylist = ['code','datetime','close','open','volume','strike_price','d_date','monthdiff','type','ownerprice']

    def getOrders(self, type='C'):
        self.keylist = ['datetime', 'code',  'close']
        sql = """
            select datetime , code, [close]  from 
                (select datetime , [close], code,
                row_number() over(PARTITION by datetime ORDER BY [close]) as rn 
                from option_tmp 
                where [close] <> 0 and type='%s') a 
                where rn=1 
                ORDER BY datetime 
            ;
        """ % type
        return [d for d in self.execSql(sql)]


# stock sh50_baseinfo
class sh50_baseinfo(baseSql):
    tablename = "sh50_baseinfo"
    up_struct = keylist = ['code', 'code_down', 'contractMonth', 'stockId', 'cateId', 'cateList', 'expireDay']

    # Chinese_List = ['plate_name']

    def getCodes(self):
        self.keylist = ['code', 'code_down', 'expireDay']
        for doc in self.execSql("select code,code_down, expireDay from %s  where status=1" % self.tablename):
            yield doc

    def findCodes(self, sign=1, start=30, end=60):
        c = 'code' if sign == 1 else 'code_down'
        sql = """
              select %s from sh50_baseInfo
              where datediff(day, GETDATE(),expireDay) BETWEEN %s and %s   
          """ % (c, str(start), str(end))
        return [d[0] for d in self.execSql(sql, isJson=False)]


class sh50_baseinfo_tmp(sh50_baseinfo):
    tablename = "sh50_baseinfo_tmp"
    # sql_update - 每天检查期权更新

    sql_update = """
            update sh50_baseInfo 
            set status = 0
            where code not in (select code from sh50_baseInfo_tmp) 
            ;
            
            INSERT into sh50_baseInfo 
            (code, code_down, contractMonth, stockId, cateId,cateList, expireDay,[status]) 
            select code, code_down, contractMonth, stockId, cateId, cateList,expireDay, [status] from sh50_baseInfo_tmp 
            where code not in (select code from sh50_baseInfo) 
          """

    def updateBase(self):
        self.update(self.sql_update)


class sh50_price(baseSql):
    tablename = "sh50_price"
    up_struct = keylist = ["code", "createTime", "bid_vol", "bid", "price", "ask", "ask_vol", "keep", "increase",
                           "power",
                           "lastclose", "sopen",
                           "overprice", "overprice11", "ask5", "ask5_vol", "ask4", "ask4_vol", "ask3", "ask3_vol",
                           "ask2", "ask2_vol",
                           "ask1", "ask1_vol", "bid_1", "bid1_vol", "bid_2", "bid2_vol", "bid_3", "bid3_vol", "bid_4",
                           "bid4_vol",
                           "bid_5", "bid5_vol", "currentTime", "sign", "state", "ownertype", "ownercode", "name",
                           "wave", "high",
                           "low", "volume", "amount", "m"]

    Chinese_List = ['name']

    """
       买量(0)，买价，最新价，卖价，卖量，持仓量，涨幅，行权价，昨收价，开盘价，
       涨停价，跌停价(11), 申卖价五，申卖量五，申卖价四，申卖量四，申卖价三，申卖量三，申卖价二，申卖量二，
       申卖价一，申卖量一，申买价一，申买量一 ，申买价二，申买量二，申买价三，申买量三，申买价四，申买量四，
       申买价五，申买量五，行情时间，主力合约标识，状态码， 标的证券类型，标的股票，期权合约简称，振幅(38)，最高价，
       最低价，成交量，成交额,M
    """


class sh50_price_s(baseSql):
    tablename = "sh50_price_s"

    keylist = ['code', 'name', 'power', 'bid', 'ask']
    sql_etf = """
    
         select top(1) code,name, power, bid, ask 
             from %s
             where datediff(mi, createTime, '%s') BETWEEN 0 and 5  
             and expireDay BETWEEN 30 and 60
             and sign * ( power - %s) BETWEEN 0 and 0.05 
             and sign = %s 
             order by expireDay 
    """

    sql_code = """

             select top(1) code, name, power, bid, ask 
                 from %s
                 where datediff(mi, createTime, '%s') BETWEEN 0 and 5  
                 and code = '%s' 
        """

    def getETF(self, sign=1, price=2.30, currentTime=None, code=None):
        if currentTime is None:
            currentTime = public.getDatetime()

        if code is None:
            sql = self.sql_etf % (self.tablename, str(currentTime), str(price), str(sign))
        else:
            sql = self.sql_code % (self.tablename, str(currentTime), code)

        # print(sql)
        return self.execOne(sql)


# 期货基础信息
class stock_baseInfo(baseSql):
    tablename = "stock_baseInfo"
    up_struct = keylist = ['code', 'abbrev_symbol', 'board_type', 'exchange', 'industry_code', 'industry_name',
                           'listed_date', 'market_tplus',
                           'order_book_id', 'round_lot', 'sector_code', 'sector_code_name',
                           'special_type', 'status', 'symbol', 'trading_hours', 'type', 'is50', 'isST']

    Chinese_List = ['industry_name', 'sector_code_name', 'symbol']

    def getCodes(self, isBound=0):
        self.keylist = ['order_book_id']
        f = ' isBound=1' if isBound == 1 else '1=1'

        sql = "select order_book_id from %s  where isST<>1 and %s" % (self.tablename, f)
        return [doc[0] for doc in self.execSql(sql, isJson=False)]

    def getAllCodes(self):
        sql = "select order_book_id from %s " % (self.tablename)
        return [doc[0] for doc in self.execSql(sql, isJson=False)]

    def getDict(self, isBound=0):
        f = ' isBound=1' if isBound == 1 else '1=1'
        d = {}
        sql = "select order_book_id, symbol from %s  where %s" % (self.tablename, f)
        for doc in self.execSql(sql, isJson=False):
            d[doc[0]] = doc[1]
        return d

    def updateBound(self, codes):
        sql = """
                  update stock_baseInfo 
                  set isBound = 1
                  where order_book_id in ('%s')  
               """ % "','".join(codes)
        self.update(sql)

    def iniBound(self):
        sql = """
                    update stock_baseInfo 
                    set isBound = 0
                    where id > 0 
                 """
        self.update(sql)

    def getSameIndustry(self, code):
        sql = """
            select order_book_id, industry_code from stock_baseInfo 
            where industry_code in
             (select industry_code
              from stock_baseInfo 
                where code = '%s')
        """
        key = code[:code.find('.')] if code.find('.')>-1 else code
        #print(key)
        sql =sql % key
        docs = [doc for doc in self.execSql(sql, isJson=False)]
        if len(docs)==0: return None,None
        return docs[0][1], [doc[0] for doc in docs]

# 股票相关一致性计算
class stock_uniform(baseSql):
    tablename = "stock_uniform"
    up_struct = keylist = ['code','code1','kline','relative','coint','coint_1','samecount',
                           'samerate','samerate3','diffstd','std','vol','score']

    def top(self, score = 90):
        sql = """
            	select code, code1, 
		        avg(score) as score
		        from %s
	            GROUP BY code, code1 
		        having avg(score)>%s        
        """ % (self.tablename, str(score))
        return [(doc[0],doc[1]) for doc in self.execSql(sql, isJson=False)]

# 期货基础信息
class stock_PE_year(baseSql):
    tablename = "stock_PE_year"
    up_struct = keylist = ['code', 'year', 'pe_ratio']

    sql_pe = """
        with ttt as (
            select 
            a.industry_code,
            b.year,
            avg(b.pe_ratio) as ratio 
            from stock_baseInfo a 
            LEFT JOIN stock_PE_year b 
            on a.order_book_id = b.code
            GROUP BY a.industry_code, b.year) 
            
            select ratio 
            from ttt a 
            LEFT JOIN stock_baseInfo b on a.industry_code = b.industry_code
            where (b.code = '%s' or b.order_book_id  = '%s') and a.year = %s
    """

    def getPE(self, code, year):
        sql = self.sql_pe % (code, code, str(year))
        res = self.execOne(sql, isJson=False)
        return res[0]


class stock_record_t0(baseSql):
    tablename = "stock_record_t0"
    up_struct = keylist = ['datetime', 'code', 'name', 'mode', 'price', 'volume', 'amount', 'isopen',
                           'income', 'orderid', 'InvestorID', 'entrustid', 'fee', 'tax', 'export', 'fee_other', 'memo',
                           'batchid']
    Chinese_List = ['name', 'memo']

    def getCodes(self):
        sql = """
            select 
                DISTINCT
                b.order_book_id 
                from stock_record_t0 a 
                LEFT JOIN stock_baseInfo b 
                on a.code = b.code 
                where b.order_book_id is not Null  
 
        """
        return [res[0] for res in self.execSql(sql, isJson=False)]

    def getCodes1(self):
        sql = """
           select 
                b.order_book_id, min(datetime) as start, max(datetime) as [end]
    			from stock_record_t0 a 
                LEFT JOIN stock_baseInfo b 
                on a.code = b.code 
                where b.order_book_id is not Null  
								GROUP BY b.order_book_id 
								ORDER BY b.order_book_id 

        """
        return [[res[i] for i in range(3)] for res in self.execSql(sql, isJson=False)]

    def getRecord(self, code):
        self.keylist = ['id', 'datetime', 'code', 'isopen']
        sql = """
          select id, datetime, code, isopen 
          from stock_record_t0
          where trim(batchid)<>'' and code = '%s'                
          ORDER BY datetime ;
           """ % code

        return [res for res in self.execSql(sql, isJson=True)]


class stock_t0_param(baseSql):
    tablename = "stock_t0_param"
    up_struct = keylist = ['parentid', 'code', 'price', 'vol', 'mode', 'isopen', 'batchid', 'interval', 'raise',
                           'Curma', 'mv', 'bias', 'volc', 'volc120', 'vol_t3', 'vol_t5', 'modem3', 'modem5', 'sb_diff',
                           'pa_diff', 'pb_diff', 'va_5', 'vb_5', 'sbma2', 'sbmb2', 'sbma3', 'sbmb3', 'min5', 'max5']


class stock_orderForm(baseSql):
    tablename = "stock_orderForm"
    up_struct = keylist = ['code', 'name', 'createdate', 'price', 'vol', 'mode', 'isBuy', 'fee',
                           'amount', 'income', 'batchid', 'uid', 'interval', 'orderID', 'method', 'status', 'ini_price',
                           'ini_vol', 'pow', 'width', 'last', 'vol_3', 'vol_120', 'diss_a', 'reportdate']

    Chinese_List = ['name']

    # 用于初始化节点
    def getPosition(self, method='ma60', codes=None):
        self.keylist = ['code', 'vol', 'mode', 'price', 'batchid']
        filter = '1=1'
        if codes is not None:
            filter = "code in ('%s')" % ("','").join(codes)
        sql = """
            select  code, sum(vol * isBuy) as vol,
             max(mode) as mode, max(price) as price, 
             max(batchid) as batchid
            from %s  
            where method = '%s' and %s  
            GROUP BY code 
            having sum(vol*isBuy) > 0 
       """ % (self.tablename, method, filter)

        d = {}
        for doc in self.execSql(sql):
            d[doc['code']] = doc

        return d

    # 更新状态
    def confirm(self, docs):
        sql = """
            select orderID from %s where status<> 6 
        """ % self.tablename
        orders = [d[0] for d in self.execSql(sql, isJson=False)]

        sql_u = """
            update %s 
            set status = 6 ,
                price = %s,
                amount = %s,
                vol = %s 
                where orderID = %s and status <> 6 
        """
        for doc in docs:
            if doc['local_report_no'] in orders:
                p = doc['matchprice'] / doc['matchqty']
                sql0 = sql_u % (
                    self.tablename, str(p), str(doc['matchprice']), str(doc['matchqty']), doc['local_report_no'])
                self.update(sql0)
        return True

    def update_last(self, code, last, batchid=None):
        f = '1=1'
        if batchid is not None: f = "batchid='%s'" % batchid
        sql = """ 
             update %s 
             set last = %s
                 where code = '%s' and isBuy= 1 and status in (0,1, 6) and %s
        
        """ % (self.tablename, str(last), code, f)
        self.update(sql)

    # 用于提示
    def getRecords(self, method='ma60'):
        self.keylist = ['createdate', 'code', 'name', 'isBuy', 'vol', 'price', 'ownerPrice']
        sql = """
            select 
                a.createdate, a.code, a.name, a.isBuy, a.vol, round(a.price,3) , round(a.ownerPrice,3)
                from %s a 
                where DATEDIFF(dd, createdate, GETDATE()) between 0 and 2 
                and method = '%s'
                ORDER BY createdate;
        """ % (self.tablename, method)
        return [doc for doc in self.execSql(sql)]


class sh50_orderForm(baseSql):
    tablename = "sh50_orderForm"
    up_struct = keylist = ['code', 'name', 'createdate', 'price', 'vol', 'mode', 'isBuy', 'pos', 'ownerPrice', 'fee',
                           'amount', 'income', 'batchid', 'uid', 'interval', 'orderID', 'method', 'status', 'ini_price',
                           'ini_hands', 'pow', 'width']

    Chinese_List = ['name']

    def getPosition(self, method='ma60'):
        self.keylist = ['code', 'vol', 'pos', 'mode', 'price']
        sql = """
            select  code, sum(vol*isBuy) as vol, max(pos) as pos, max(mode) as mode, max(price) as price from %s  
            where code in 
            (select top(1) code from sh50_orderForm
             where method = '%s'
            order by createdate desc) 
            GROUP BY code 
            having sum(vol*isBuy) > 0 
       """ % (self.tablename, method)
        return self.execOne(sql)

    def getRecords(self, method='ma60'):
        self.keylist = ['createdate', 'code', 'name', 'mode', 'isBuy', 'vol', 'price', 'ownerPrice']
        sql = """
            select 
                a.createdate, a.code, a.name, a.mode, a.isBuy, a.vol, round(a.price,3) , round(a.ownerPrice,3)
                from %s a 
                where DATEDIFF(dd, createdate, GETDATE()) between 0 and 2 
                and method = '%s'
                ORDER BY createdate;
        """ % (self.tablename, method)
        return [doc for doc in self.execSql(sql)]


class history(baseSql):
    tablename = "history"
    up_struct = keylist = ['code', 'area', 'sdate', 'sopen', 'high', 'low', 'sclose', 'volumn']

    def getByCode(self, code, date=0):
        sql = """
          select sdate, sclose 
          from history 
          where code = '%s'
          and DATEDIFF(day , sdate, getDate()) between 0 and  365 
          ORDER BY sdate  
          """ % code
        return self.execSql(sql, isJson=False)

    def getCompare(self, code1, code2, start=None):
        self.keylist = ['date', 'sclose', 'rate']
        sql = """
         select  
              a.sdate, a.sclose, case when b.sclose = 0 then 0 else a.sclose/b.sclose end as rate
              from history a 
              LEFT JOIN history b on a.sdate = b.sdate 
              where  a.code = '%s' and b.code = '%s' and a.sdate >= '%s' and b.sdate > '%s'
              ORDER BY a.sdate  
          """ % (code1, code2, start, start)
        for doc in self.execSql(sql):
            yield doc

    def getCodes(self):
        sql = """select a.code, b.name from
                history a 
                LEFT JOIN baseInfo b on a.code = b.code 
                where SUBSTRING(a.code,1,2)  in ('00','30','60') and b.stocktype='STOCK'
                
                GROUP BY a.code,b.name 
                ORDER BY a.code 
                  """
        return self.execSql(sql, isJson=False)


class history_qfq(baseSql):
    tablename = "history_qfq"
    up_struct = keylist = ['code', 'order_book_id', 'sdate', 'sopen', 'high', 'low', 'sclose', 'volume',
                           'total_turnover']


class warInfo(baseSql):
    tablename = "warInfo"
    up_struct = ['code', 'name', 'market', 'stock_type', 'stock_child_type',
                 'owner_stock_code', 'listing_date', 'due_date', 'exercise_price', 'exchange_ratio']

    keylist = ['code', 'owner_stock_code', 'listing_date', 'stock_child_type', 'due_date', 'exercise_price',
               'exchange_ratio']
    Chinese_List = ['name']
    sql = """
                 select %s   
                 from warinfo
                 where stock_child_type  in ('PUT','CALL') and exercise_price <> 0  and owner_stock_code <> ''
                """
    sql_online = """
           select %s  
           from warinfo a 
           LEFT JOIN  warrant_his b on a.code = b.code and b.sdate in (select max(sdate)from warrant_his) 
           
           where a.stock_child_type  in ('PUT','CALL') and a.exercise_price <> 0  and a.owner_stock_code <> '' 
                 and DATEDIFF(day, getDate(), a.due_date) > 90
        			   and b.sd_Price <> 0 
	       and b.sclose / b.sd_Price < 1.2
    """

    def getWars(self):
        keys = ",".join(self.keylist)
        sql = self.sql % keys
        for doc in self.execSql(sql):
            yield doc

    def getOnline(self):
        keys = "a." + ",a.".join(self.keylist)
        sql = self.sql_online % keys
        for doc in self.execSql(sql):
            yield doc


class warrant(baseSql):
    tablename = "warrant"
    up_struct = keylist = ['code', 'sdate', 'sclose', 'owner', 'sd_Price', 'd1', 'd2', 'ep']


class warrant_his(warrant):
    tablename = "warrant_his"


class warrant_online(warrant):
    tablename = "warrant_online"
    up_struct = keylist = ['code', 'sdate', 'sclose', 'owner', 'sd_Price', 'd1', 'd2', 'ep', 'w_volume',
                           'w_turnover_rate', 'o_volume', 'o_turnover_rate']


# stock基础类
class wave(baseSql):
    tablename = "wave"
    up_struct = keylist = ['code', 'sdate', 'wave', 'count', 'sclose']

    sql = " select %s from wave where code='%s'"

    sql_last = """
         select code , sdate,wave
         from wave 
           where sdate in (
         select max(sdate) from wave )
         ORDER BY code 
         """

    def getByCode(self, code):
        keys = ",".join(self.keylist)
        sql = self.sql % (keys, code)
        for doc in self.execSql(sql):
            yield doc

    def getLastWave(self):
        self.keylist = ['code', 'sdate', 'wave']
        return self.execSql(self.sql_last)


class orderRecord(baseSql):
    tablename = "orderRecord"
    up_struct = keylist = ['code', 'market', 'mode', 'type', 'price', 'volume', 'amount', 'account', 'company',
                           'createTime', 'status', 'returnid']


class bitmex(baseSql):
    tablename = "bitmex"
    up_struct = keylist = ['cointype', 'currentprice', 'futureprice', 'difference', 'recordtime']

    def getLast(self, code=None):
        if code is None or code == 'all':
            sql = """
                select cointype, currentprice, futureprice, difference, recordtime 
                from 
                (select * , row_number() over(PARTITION by cointype  ORDER BY recordtime desc) as rn 
                 from bitmex 
                  where datediff(mi,recordtime,GETDATE()) BETWEEN 0 and 60 ) a 
                where a.rn = 1  
            """
        else:
            sql = """
                       select cointype, currentprice, futureprice, difference, recordtime 
                       from 
                       (select * , row_number() over(PARTITION by cointype  ORDER BY recordtime desc) as rn 
                        from bitmex 
                         where datediff(mi,recordtime,GETDATE()) BETWEEN 0 and 60 and cointype like '%""" + code + """%') a 
                       where a.rn = 1  
                   """

        return [doc for doc in self.execSql(sql, isJson=False)]


# 期货基础信息
class future_baseInfo(baseSql):
    tablename = "future_baseInfo"
    up_struct = keylist = ['code', 'ratio', 'symbol', 'exchange', 'margin_rate', 'contract_multiplier', 'product',
                           'tick_size', 'nightEnd']
    Chinese_List = ['symbol']

    def getInfo(self, codes=[]):
        self.keylist = ['code', 'ratio', 'margin_rate', 'exchange', 'contract_multiplier', 'nightEnd',
                        'lastPrice', 'tick_size', 'ctp_symbol', 'bulltype', 'klinetype', 'quickkline']

        if len(codes) == 0:
            sql = "select %s from %s where product in ('Commodity','Index') and ctp_symbol is not Null"
            sql = sql % (",".join(self.keylist), self.tablename)

        else:
            sql = "select %s from %s where code in ('%s')"
            sql = sql % (",".join(self.keylist), self.tablename, "','".join(codes))


        res = [doc for doc in self.execSql(sql)]
        print(res)
        return res

    def getUsedMap(self, hasIndex=False, isquick=False):
        filter = " and product<>'Index'"
        if hasIndex:
            filter = " "

        if isquick:
            filter += " and quickkline <> ''"

        sql = "select code from %s where isUsed=1 %s  order by code" % (self.tablename, filter)
        # print(sql)
        return [d[0] for d in self.execSql(sql, isJson=False)]

    sql_all = """
        select code, nightEnd from %s 
        where product='Commodity' and lastVolume> %s 
        order by lastPrice desc 
    """

    def all(self, vol=700):
        sql = self.sql_all % (self.tablename, str(vol))
        return [(d[0], d[1]) for d in self.execSql(sql, isJson=False)]

    def exists(self):
        sql = "select code, nightEnd from %s" % (self.tablename)
        return [(d[0], d[1]) for d in self.execSql(sql, isJson=False)]

    # 更新ctp当前主code
    def setCtpCode(self, key, ctp, main=''):
        sql = "update %s set ctp_symbol='%s', ctp_main='%s' " \
              "where code = '%s'" % (self.tablename, ctp, main, key)
        self.update(sql)

    # 更新ctp当前主code
    def setMulti(self, doc):
        c, m, r, e = (doc[k] for k in ['code', 'contract_multiplier', 'margin_rate', 'nightEnd'])
        sql = """
                update %s set  contract_multiplier=%s, margin_rate=%s, nightEnd= '%s' 
                where code = '%s'
            """ % (self.tablename, str(m), str(r), e, c)
        self.update(sql)


# 下单记录
class future_orderForm(baseSql):
    tablename = "future_orderForm"
    up_struct = keylist = ['code', 'name', 'createdate', 'price', 'vol', 'mode', 'isopen', 'fee', 'uid', 'income',
                           'rel_price',
                           'rel_std', 'batchid', 'bullwidth', 'delta', 'widthdelta', 'hands', 'isstop', 'interval',
                           'method', 'orderId', 'status',  'ini_price', 'ini_hands', 'symbol', 'highIncome',
                           'lowIncome', 'stop_price', 'memo']

    Chinese_List = ['memo']
    preColumns = "id,uid,code,mode,isopen,isstop,batchid,price,vol,hands,fee,ini_price,createdate,symbol,stop_price,method,name"

    def getOpenMap(self, method='1m', codes=[], batchNum=2):
        sql = """
            select %s
                FROM %s 
                where batchid in 
                (select batchid 
                 FROM %s 
                 where method ='%s' 
                 GROUP BY batchid
                 having count(*)=%s)
                 and isopen = 1 and (status is null or status <> -1)  and hands > 0     
                 ORDER BY createdate desc 
                            
        """ % (self.preColumns, self.tablename, self.tablename, method, str(batchNum))
        # print(sql)
        map = {}
        self.keylist = self.preColumns.split(",")
        for doc in self.execSql(sql):
            keys = doc['uid'].split("_")
            if len(codes) == 0 or (keys[0] in codes and keys[batchNum - 1] in codes):
                key = "_".join(keys[0:batchNum])
                if not key in map:
                    map[key] = []
                map[key].append(doc)

        # print(map)
        return map

    def getById(self, id):
        sql = """  select %s from %s where id = '%s'""" % (self.preColumns, self.tablename, id)
        self.keylist = self.preColumns.split(",")
        return self.execOne(sql)

    def getRecent(self, method=[]):
        methods = "'%s'" % ("','").join(method)
        sql = """
            select %s from %s
            where method in (%s)
            and DATEDIFF(mi, createdate,GETDATE()) BETWEEN 0 and 10
            order by createdate desc 
        """ % (self.preColumns, self.tablename, methods)
        map = {}
        self.keylist = self.preColumns.split(",")
        for doc in self.execSql(sql):
            map[doc['id']] = doc
        return map

    # 更新pStatus
    def setPStatus(self, id, status):
        sql = """
            update %s 
            set pStatus = %s 
            where id = %s 
        """ % (self.tablename, str(status), str(id))
        self.update(sql)

    # 标记结束
    def disuse(self, ids):
        ss = ",".join(ids)
        sql = "update %s set status=-1  where id in (%s)" % (self.tablename, ss)
        self.update(sql)

    def exitCodes(self, long=18):
        sql = """
          select
          DISTINCT
          SUBSTRING(uid, 0, len(uid) - %s)
          from %s
         """ % (str(long), self.tablename)
        return [r[0] for r in self.execSql(sql, isJson=False)]

    # 查询上一次交易的方向，用于趋势策略
    def lastMode(self, method=['mZhao'], code='ZC1905'):
        methods = "'%s'" % ("','").join(method)
        sql = """
            select top(1) mode  
                FROM %s 
                where 
					name = '%s' and method in (%s) 
					and isopen = 1 and (status is null or status <> -1) and hands > 0
					and name  not in 
                (select max(name)  
                 FROM future_orderForm 
                 where method in (%s)  
					 and (status is null or status <> -1) and hands > 0 
                 GROUP BY batchid
                 having count(*)=1)
                 ORDER BY createdate desc  
         """ % (self.tablename, code, methods, methods)
        doc = self.execOne(sql, isJson=False)
        if doc is None:
            return 0
        else:
            return doc[0]

    # 计算趋势
    def trendMap(self, methods=['']):
        sql = """
             select name,
			  (case when (count(*)=1 and max(method)='%s') then 1 else -1 end) * (case when mode > 0 then 1 else -1 end)  as trend,
				 count(*) as num
			          FROM %s
                where batchid in 
                (select batchid 
                 FROM future_orderForm 
                 where method in ('%s') 
                 GROUP BY batchid
                 having count(*)=1)
                  and isopen = 1 and (status is null or status <> -1) and hands > 0 
								 GROUP BY name, case when mode > 0 then 1 else -1 end 
								 order BY name, case when mode > 0 then 1 else -1 end 
	    """
        if len(methods) < 2: return None
        sql = sql % (methods[0], self.tablename, "','".join(methods))
        map = {}
        self.keylist = ['code', 'trend', 'num']
        for doc in self.execSql(sql):
            map[doc['code']] = doc
        return map

    # 趋势策略用于检查是否存在同向的ma10策略交易
    def isOpenExists(self, method='mZhao', code='ZC1905'):
        sql = """
            select count(*) 
                FROM %s 
                where batchid in 
                (select batchid 
                 FROM future_orderForm 
                 where method = '%s' and code = '%s'
                 GROUP BY batchid
                 having count(*)=1)
                 and isopen = 1 and (status is null or status <> -1) and hands > 0 
        """ % (self.tablename, method, code)
        res = [d[0] for d in self.execSql(sql, isJson=False)]
        return res[0] == 1

    # 上一次平仓类型为isstop=2的品种
    def lastStop(self, method=['mZhao', 'mZhao55'], code='MA'):
        methods = "'%s'" % ("','").join(method)
        sql = """
              select isstop, createdate from(
                    select
                        max(name) as code,
                        max(createdate) as createdate,
                        max(case when isopen=1 then -1 else isstop end) as isstop,
                        row_number() over(PARTITION by  max(name) ORDER BY max(createdate) desc) as rn  	
                    from %s  
                    where status <> -1 and method in (%s) and name  = '%s'
                    GROUP BY batchid 
                    having count(*)= 2 
                    ) a where a.rn=1 
        """ % (self.tablename, methods, code)

        doc = self.execOne(sql, isJson=False)
        if doc is not None:
            return (doc[0], doc[1])
        else:
            return (0, None)

    def openMode(self, method=['mZhao', 'mZhao55'], code='MA'):
        methods = "'%s'" % ("','").join(method)
        sql = """
               select max(method) as method, count(*), max(mode) as count 
                    from 
                    (select  
                    max(name) as code, 
                    max(method) as method,
                    max(mode) as mode  
                    from %s
                    where method in (%s)  and status <> -1 and hands> 0 and name = '%s'
                    GROUP BY batchid
                    having count(*)=1) b 
                    GROUP BY code 
                """ % (self.tablename, methods, code)

        doc = self.execOne(sql, isJson=False)
        if doc is not None:
            return (doc[0], doc[1], doc[2])
        else:
            return ('', 0, 0)

    # 所有未平仓的交易对
    def currentOpenCodes(self, method='1m', batchNum=2):
        sql = """
               select max(uid), batchid, sum(hands), max(mode) 
               from %s
               where method = '%s' and (status is Null or status<>-1)   
               GROUP BY batchid  
               having count(*) = %s
           """ % (self.tablename, method, str(batchNum))
        # print(sql)
        return [r[0].split('_') + [r[1], r[2], r[3]] for r in self.execSql(sql, isJson=False)]

    # 收益好的币对
    def topCodes(self, method='mul', toptable='train_total_s1', batchNum=2, topNum=10):

        sql = """
          select top(%s) b.uid from 
          (select  dbo.uidWhich(uid,0) as code,  dbo.uidWhich(uid,$num) as code1, sum(income) as income from 
                 %s 					
                 where method = '%s' 
                 GROUP BY dbo.uidWhich(uid,0) , dbo.uidWhich(uid,$num)  
                 having sum(income) > 0 
           ) a 
			LEFT JOIN 
			(select uid ,code, code1, income/amount as rate, 
				row_number() over(PARTITION by a.code, a.code1 ORDER BY income/amount desc) as rn
				from %s a 
				where count >10 
	    ) b on a.code = b.code and (b.code1 is null or a.code1=b.code1) 
			where b.rn = 1 
			order by a.income desc 
 
        """ % (str(topNum), self.tablename, method, toptable)

        sql = sql.replace('$num', str(batchNum - 1))
        # print(sql)
        return [r[0].split('_') for r in self.execSql(sql, isJson=False)]

    def posByCode(self, methods=['zhao', 'zhao55']):
        sql = """
                select name, case when mode>0 then 0 else 1 end  as mode, sum(hands) as hands, max(code)  
                from %s 
					where batchid  in 
					(select  
					  batchid
                from %s
                where method in ('%s')  and status <>-1   and hands> 0
                GROUP BY batchid
                having count(*)=1) 
				GROUP BY name, case when mode>0 then 0 else 1 end  
                ORDER BY name, case when mode>0 then 0 else 1 end  
        """
        sql = sql % (self.tablename, self.tablename, "','".join(methods))
        for doc in self.execSql(sql, isJson=False):
            yield (doc[0] + "_" + str(doc[1]), doc[2], doc[3])


    # 用于期货SG历史数据的回测整理
    def sg_get(self, diss= '1d'):
        self.keylist = ['code', 'name', 'date','mode','isopen','vol']

        dd =  'convert(VARCHAR,createdate,112)' if diss[-1]=='d'\
            else ('dbo.calcDateArea(createdate,%s)' % diss[:-1])

        sql = """
            select code, max(name), $dd,  mode, isopen, sum(hands) as vol 
            from %s 
            where batchid<>'0'
            GROUP BY  $dd, code, mode ,isopen
            ORDER BY code, $dd;
        """ % self.tablename
        sql = sql.replace('$dd', dd)
        return [doc for doc in self.execSql(sql)]

    # 期货SG历史数据 - 品种、时间区间，用于对比
    def sg_codes(self):
        self.keylist = ['code', 'name', 'startdate', 'enddate', 'iniVolume']
        sql = """
            select a.code, a.name,
            min(convert(VARCHAR,createdate,120)) as startdate,
            max(convert(VARCHAR,createdate,120)) as enddate,
            max(abs(ini_hands)) as maxKeepVolume 
            from %s a
            where name not in ('IC','IH','IF') 
            GROUP BY a.code, a.name   
            having datediff(day,min(createdate), max(createdate))>0 
            ORDER BY a.code 
        """ % self.tablename
        return [doc for doc in self.execSql(sql)]

class future_status(baseSql):
    """
        期货过程状态类，用于隔日恢复
    """
    tablename = "future_status"
    up_struct = keylist = ['code', 'createdate', 'status', 'method']

    # 更新pStatus
    def setStatus(self, code, method, status):
        sql = """
            IF EXISTS(SELECT 1 FROM $table where code='$code' and method='$method')
             BEGIN
              update $table
              set status = $status,
                 createdate = getDate()
             where code='$code' and method='$method'
            END
             --不存在就插入
            ELSE
             BEGIN
              insert into $table (code,method,createdate,status)
              VALUES('$code','$method',GETDATE(),$status) 
            END 
           """
        self.paramMap = {
            "$table": self.tablename,
            "$code": code,
            "$method": method,
            "$status": str(int(status))
        }
        sql = self.parseSql(sql)
        self.update(sql)

    def getStatus(self, method):
        self.keylist = ['code', 'status']
        sql = """
            select code,status from %s 
            where method = '%s'
        """ % (self.tablename, method)
        res = {}
        for doc in self.execSql(sql):
            res[doc['code']] = doc['status']
        return res


class train_future(baseSql):
    tablename = "train_future"
    up_struct = keylist = ['code', 'createdate', 'price', 'vol', 'mode', 'isopen', 'fee', 'uid', 'income', 'rel_price',
                           'rel_std', 'batchid', 'bullwidth', 'delta', 'isstop', 'shift', 'widthDelta', 'atr',
                           'price_diff', 'p_l', 'p_h', 'hands', 'result', 'macd', 'options', 'highIncome', 'lowIncome',
                           'mastd', 'macd2d', 'stop_price', 'amount', 'pid']


class train_total(baseSql):
    tablename = "train_total"
    up_struct = keylist = ['code', 'createdate', 'code1', 'count', 'amount', 'period', 'scale', 'income', 'uid',
                           'relative', 'std', 'delta', 'shift', 'coint', 'timediff', 'price', 'maxdown', 'sharprate']

    def exits(self):
        sql = """
            select DISTINCT code, code1 from %s
              """ % self.tablename
        res = self.execSql(sql, isJson=False)
        return [[r[0], r[1]] for r in res]

    def top(self):
        sql = """
           select DISTINCT code+'_' +code1 
            from %s
            where income > 0 
         """ % self.tablename
        res = self.execSql(sql, isJson=False)
        return [r[0].split("_") for r in res]

    # 最新的最佳对
    def last_top(self, num=20, maxsames=3, filter='1=1', minRate=0.035, mode='income', ban=[]):
        # key = 'sharprate' if  mode=='sharp' else 'income/amount'
        bFilter = '1=1'
        if len(ban) > 0:
            bFilter = "aa.code not in ('%s')" % "','".join(ban)

        sql = """
              select aa.uid,aa.code,aa.code1, aa.rate * bb.winrate as rate
                from 
                ( select uid ,code, code1, income/amount as rate, 
                  row_number() over(PARTITION by a.code ORDER BY %s desc) as rn
                  from %s a 
                  where %s and income/amount > %s 
                ) aa left join 
                
                ( select code, code1, 
                  sum(case when income > 0 then 1 else 0 end ) * 1.0 / count(*) * 1.0 as winrate
                  from %s 
                  -- where  dbo.uidwhich(uid,3) <> 5
                  GROUP BY code, code1) bb 
                on aa.code = bb.code and (aa.code1 = bb.code1 or aa.code1 is null)  
                   
                where aa.rn = 1  and aa.rate * bb.winrate > %s and %s  
                
                order by aa.rate * bb.winrate  desc  
            
         """ % (mode, self.tablename, filter, str(minRate), self.tablename, str(minRate), bFilter)

        # print(sql)
        total = {}
        res = []
        for doc in self.execSql(sql, isJson=False):
            valid = True
            # 重复次数不超过3次
            for j in [1, 2]:
                if not doc[j] in total.keys():
                    total[doc[j]] = 1

                elif doc[j]:
                    total[doc[j]] += 1

                if total[doc[j]] > maxsames:
                    valid = False

            if valid:
                # uid = re.sub('_last$|_0.628|88|_kline_0.527$', '', doc[0])
                uid = doc[0]
                l = uid.split("_")
                res.append(l)
                if len(res) >= num:  break
        return res

    def last_top1(self, num=20, minVol=100, mode='income', method='zhao', ban=[]):
        bFilter = '1=1'
        if len(ban) > 0:
            bFilter = "a.code not in ('%s')" % "','".join(ban)

        sql = """
                  select 
                    max(case when dbo.uidWhich(uid,5)='%s' then uid else '' end) as uid
                    ,a.code
                    from %s a 
                    LEFT JOIN future_baseInfo b 
                    on a.code = b.code 
                    where b.lastVolume > %s and %s
                    group by a.code 
                    having sum(income)> -2000000
                    order by sum(income) desc; 
                    
                 """ % (method, self.tablename, str(minVol), bFilter)

        total = {}
        res = []
        for doc in self.execSql(sql, isJson=False):
            valid = True
            if valid:
                uid = doc[0]
                l = uid.split("_")
                res.append(l)
                if len(res) >= num: break
        return res


class action_stock:
    @staticmethod
    def exec_creat():
        pass


def main():
    obj = future_orderForm()
    res = obj.isOpenExists(method='mZhao55', code='ZC1905')
    print(res)


if __name__ == '__main__':
    main()
