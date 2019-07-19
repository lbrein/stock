# -*- coding: utf-8 -*-
"""
Created on  2018-01-04 
@author: lbrein
       ---  通用接口 ---      

   主要接口：
          米框期货行情接口

-
"""

import rqdatac as rq

import datetime
import time
from com.base.public import public, logger
from com.object.obj_entity import future_baseInfo, stock_baseInfo, stock_PE_year
from rqdatac.services import future
from rqdatac.services.financial import Fundamentals as fds
from rqdatac.services.financial import query_entities as query
from rqdatac.services.financial import Financials as fns

import pandas as pd
import re
import os
import copy
import numpy as np
import talib as ta



class interface_Rice(object):
    # 读取url
    #basePath = 'C:/stock/csv/' if os.name == 'nt' else "/root/home/stock/csv/"
    basePath = 'C:/lbrein/csv/' if os.name == 'nt' else "/root/home/stock/csv/"
    iniTimeArea = [('09:00:00', "11:30:00"), ('13:00:00', '15:00:00')]

    iniTimeDiffArea = [('12:00:00', 120), ('16:00', 360), ('07:00:00', 480)]
    financialFields = ["pe_ratio"]

    def __init__(self, isSub=False):
        if not isSub:
            rq.init()
        else:
            rq.reset()

        self.columns = ["close", "open", "high", "low", "volume"]
        self.tickColumns = ["last", "open", "high", "low", "volume", "asks", "ask_vols", "bids", "bid_vols"]

        self.HisKline = None
        self.HisKlineTime = None
        self.currentKline = {}
        self.preKline = None
        self.validDate = None
        self.kPeriod = 1
        self.TimeArea = []
        self.TimeDiffArea = []
        self.indexCodeList, self.indexMcodes = [], [] # 指数期货对应的行情代码
        self.indexMap = {}

        self.HisKlineTime = {}
        self.HisKline = {}
        # tick 触发持续状态，进程结束时返回false，用于终止ctp窗口
        self.tickStatus = True

    def setIndexList(self, lists):
        self.indexCodeList = lists
        self.indexMcodes = self.getMain([l[0] for l in lists])
        i = 0
        for l in lists:
           self.indexMap[l[0]] = l[1]
           self.indexMap[self.indexMcodes[i]] = l[1]
           self.indexMap[l[1]] = self.indexMcodes[i]
           i += 1
        return self.indexMap

    def allCodes(self, type='CS'):
        return rq.all_instruments(type=type, date=public.getDate())

    def detail(self, code):
        return rq.instruments(code)

    def id_convert(self,ctpCode):
        return rq.id_convert(ctpCode)

    def get_financials(self, codes, years=10, type='y'):
          q = query(fds.financial_indicator.adjusted_return_on_equity_diluted,
                    fds.announce_date
                  ).filter(fds.stockcode.in_(codes))
          S = str(int(public.getDate().split("-")[0])-1)+'q4'

          Y = str(years)+type
          res = rq.get_financials(q, S, interval=Y)
          d = {}
          for c in codes:
              try:
                  d[c] = res.minor_xs(c)
              except:
                  continue
          return d

    def get_fundamentals(self, codes, start_date=None, years=10, type='y'):
         """
         q = query(fds.eod_derivative_indicator.pe_ratio,
                   fds.balance_sheet.total_assets,
                   fds.balance_sheet.total_liabilities,
                   fds.balance_sheet.total_equity_and_liabilities
                   ).filter(fds.stockcode.in_(codes))
         """

         q = query(fds.financial_indicator.adjusted_return_on_equity_diluted
                   ).filter(fds.stockcode.in_(codes))

         if start_date is None:
             start_date = public.getDate()

         print('-----', start_date)

         Y = str(years *4) + 'q'
         res = rq.get_fundamentals(q, start_date, interval=Y, report_quarter=True)
         d = {}
         for c in codes:
               try:
                  d[c] = res.minor_xs(c)
               except:
                  continue
         return d

    PE = None
    def getOpen(self, code, date, type='future', rate = 0.5):
        if type!='future': self.indexMap ={}

        end = public.getDate()
        req = self.kline([code], period='1d', start=date, end=end, pre=1, type=type)
        if req and req[code] is not None:
            if self.PE is None:
                self.PE = stock_PE_year()

            df = req[code]
            # 查询每日PE值
            df["pe"] = self.get_factor([code], start=date, end=end)
            # 最低值PE最小
            df["pe_min"] = df.apply(lambda row: row['low'] * row['pe']/row['close'], axis=1)
            df["pe_open"] = df.apply(lambda row: row['open'] * row['pe'] / row['close'], axis=1)

            # 查询行业均值
            pem = self.PE.getPE(code, int(str(date)[:4])-1)

            pl = pem * (1 + rate)
            sub = df[df['pe_min'] <= pl]

            if len(sub) > 0:
                ss = sub.ix[0, ['close', 'open', 'pe', 'pe_min', 'pe_open']]
                if ss['pe_open'] <= pl:
                    return sub.index[0], ss['open']
                else:
                    return sub.index[0], round(ss['close'] * ss['pe'] / pl, 2)
            else:
                return None, None

        else:
            return None


    def getLastVaildDay(self, date):
        pass

    def get_factor(self, codes, start='1999-01-01', end='2019-01-01', factor='pe_ratio'):
        try:
            df = rq.get_factor(codes, factor, start_date=start, end_date= end)
        except:
            df  = None

        return df

    def int_stockBase(self):
        # 更新stockBase
        Base = stock_baseInfo()
        Base.empty()
        df = self.allCodes(type='CS')
        df['code'] = df['order_book_id'].apply(lambda x: x[:x.find('.')])
        s = self.index_compose('000980.XSHG')
        df['is50'] = df['order_book_id'].apply(lambda x: 1 if x in s else 0)
        df['isST'] = df['symbol'].apply(lambda x: 1 if x.find('*ST') > -1 else 0)

        Base.insertAll(df.to_dict('records'))

    def index_compose(self, code):
        return rq.index_components(code)

    def allFuture(self, isSave=False):
        df = rq.all_instruments(type='Future', date=public.getDate())
        u = df["underlying_symbol"].unique()
        return u

    def allHisFuture(self, start=None):
        columns = ['maturity_date', 'order_book_id', 'listed_date', 'underlying_symbol']
        if start is  None:
            df = rq.all_instruments(type='Future')
        else:
            df = rq.all_instruments(type='Future' , date = public.getDate())
        #print(df)
        return df[df['maturity_date']!='0000-00-00'].loc[:, columns]

    def getMainMap(self):
        #  查询获得code活动代码及映射表
        lists = future_baseInfo().getUsedMap(hasIndex=False) + ['PB', 'SN']
        self.mCodes = self.getMain(lists)
        i, cmap = 0, {}
        for m in self.mCodes:
            cmap[lists[i]] = m
            i += 1

        df = self.allCodes('Future')
        df = df[df['underlying_symbol'].isin(lists) & (df['listed_date']!='0000-00-00')]
        df['mcode'] = df['underlying_symbol'].apply(lambda x: cmap[x])
        return df

    def getActive(self):
        # 获得所有当前活动清单
        df = self.getMainMap()
        codes = df['order_book_id'].tolist()

        # 查询获得日线数据
        dd = public.getDate(diff=-1)

        dfs = self.kline(codes, period='1d', start=dd, pre=0)
        res = {}
        for c in dfs:
            m = df[df['order_book_id']==c]['mcode'].values[0]
            r, rm = dfs[c].ix[dd,'volume'], dfs[m].ix[dd, 'volume']

            # 满足条件，即输出
            if (c in self.mCodes) or r/rm > 0.7 or (r > 20000 and r/rm > 0.1) or r > 50000:
                res[c] = (c in self.mCodes, r/rm, r)

        return res

    def baseUpdate(self):
        df = rq.all_instruments(type='Future', date=public.getDate())
        u = df["underlying_symbol"].unique()

        FT = future_baseInfo()
        # 已有code及结束时间
        exists = FT.exists()
        exCodes = [c[0] for c in exists]
        endMap =  {}
        for c in exists: endMap[c[0]] = c[1]

        docs = []
        mu = self.getMain(u)
        i = 0

        for d in u:
            n = df[df["underlying_symbol"] == d].loc[:, FT.keylist[2:]].values[-1].tolist()
            doc = FT.set([d, 0] + n )

            detais = rq.instruments(mu[i])
            # tick_size
            doc["tick_size"] = detais.tick_size()
            doc['contract_multiplier'] = detais.contract_multiplier
            doc['margin_rate'] = detais.margin_rate

            # 结束时间
            hh = detais.trading_hours.split(',')
            hs = hh[0].split('-')
            if hs[0][0:2] != '09':
                doc["nightEnd"] = hs[1]
            else:
                doc["nightEnd"] = hh[-1].split('-')[1]

            if d not in exCodes:
                doc['product'] = 'Commodity'
                docs.append(doc)
            else:
                # 更新结束时间
                if doc["nightEnd"]!= endMap[d][0:len(doc["nightEnd"])]:
                    print(d, endMap[d][0:len(doc["nightEnd"])], doc["nightEnd"])

                FT.setMulti(doc)
            i += 1

        if len(docs) > 0:
            FT.insertAll(docs)
            logger.info(('future base update finished, ', docs))


    def getTicks(self, mcode):
        return rq.get_ticks(mcode)

    def getAdjustDate(self, code, diff=120, start=None):
        if start is None:
            start = public.getDate(diff=-diff)

        res = rq.get_dominant_future(code, start_date= start, end_date=public.getDate(diff=0))
        pre = res[0]
        l = []
        for i in range(len(res)):
            if pre != res[i]:
                l.append((res.index[i - 1], res.index[i], pre, res[i]))
                pre = res[i]
        return l

    # 启动线程,触发tick事件
    def startTick(self, codes, kPeriod='1m', callback=None, timesleep=None, source='snap', kmap=None):
        self.codes = codes
        self.kPeriod = kPeriod
        self.curKline(codes, callback=callback, source=source, kPeriod=kPeriod, kmap=kmap, timesleep=timesleep)

    # 获取主力合约
    def getMain(self, codes, start=None, end=None):
        if start is None:
            start = public.getDate(diff=-5)
        if end is None:
            end = public.getDate()

        rs = []
        for c in codes:

            res = rq.get_dominant_future(c, start_date=start, end_date=end)
            if res is not None:
                rs.append(res.values[-1])
            else:
                rs.append(c + '88')
        return rs


    def getAllMain(self, code, start = None):
        if start is None:
            start = public.getDate(diff=-2)

        res = rq.get_dominant_future(code, start_date= start, end_date=public.getDate(diff=1))
        pre = ''
        docs = []
        for i in range(len(res)):
           if pre != res[i]: docs.append((res.index[i], res[i]))
           pre = res[i]

        return docs

    # 获得最新的K线
    def getKline(self, codes, ktype=1, key='_0', num=300):
        # 每分钟更新K线
        mins, unit = ktype, 'm'
        if str(key).find('d') > -1:
            unit = 'd'
            mins = ktype * 24 * 60

        now = int(time.time() / 60 / mins)
        #
        time0 = time.time()
        if (key not in self.HisKline.keys()) or (self.HisKlineTime[key] != now):
            self.HisKline[key] = self.kline(codes, period=str(ktype) + unit, pre=num)
            self.HisKlineTime[key] = now
            logger.info(("%s alter new %s kline" % (key, str(ktype) + unit), now, codes, ' 耗时:', time.time()-time0))
        return self.HisKline[key]

    def parseCode(self, mCode=None):
        t = ''.join(re.findall(r'[0-9]', mCode))
        return mCode.replace(t, '').upper()


    def ticks(self):
        pass


    # 获取最新的1分钟线，返回df 字典
    # pre 提前根数，用于计算k线参数
    def kline(self, codes, period='1m', start=None, end=None, pre=200, type='future'):
        d = {}
        # 指数清单
        # 提前日期

        #start = self.getPreDate(period=period, num=pre, start=start)
        preDay = 1 if (pre == 0 or period == 'tick') else self.getPreDays(period, pre)
        # 开始日期
        if start is None:
            start = public.getDate(diff=-preDay)
        else:
            start = public.getDate(diff=-preDay, start=start.split(' ')[0])

        if end is None:
            end = public.getDate(diff=1)

        # 查询K线
        if period.find('d') > -1 and len(codes) > 1:
            # 替换查询线
            newCodes = self.alterCode(codes)
            res = rq.get_price(newCodes, frequency=period,  start_date=start,
                               end_date=end, adjust_type='pre')

            for c in newCodes:
                mc = c if not c in self.indexMap else self.indexMap[c]
                d[mc] = res.minor_xs(c)

        else:

            iMap  = self.indexMap
            for c in codes:
                c0 = c1 = c
                if type == 'future':
                     c0 = self.parseCode(c)
                     c1 = iMap[c0] if  c0 in iMap else c
                #print(c1)
                d[c] = d[c0] = rq.get_price(c1, frequency=period,  start_date=start,
                                                           end_date=end, adjust_type='pre')

        return d

    def alterCode(self, codes):
        iMap, newCodes = self.indexMap, []
        if len(self.indexCodeList) > 0 and self.indexMcodes[0] in codes:
           for code in codes:
                if code in self.indexMcodes:
                    newCodes.append(iMap[code])
                else:
                    newCodes.append(code)
           return newCodes

        else:
            return copy.deepcopy(codes)

    # 获取最新快照，返回 tick 字典
    def snap(self, codes=[]):
        d = {}
        #
        newCodes = self.alterCode(codes)

        rs = rq.current_snapshot(newCodes)
        if isinstance(rs, list):
            for r in rs:
                s = {}
                for item in self.tickColumns:
                    s[item] = r[item]
                oid = r["order_book_id"]
                s["code"] = oid if not oid in self.indexMap else self.indexMap[oid]
                d[s["code"]] = s
        else:
            s = {}
            for item in self.tickColumns:
                s[item] = rs[item]
            oid = rs["order_book_id"]
            s["code"] = oid if not oid in self.indexMap else self.indexMap[oid]
            d[s["code"]] = s
        return d

    # 一分钟及时K线, 回调返回
    def curKline(self, codes, callback=None, source='snap', timesleep=None, kPeriod='1m', kmap=None):
        # 是否是整点秒的第一个
        if timesleep is None:
            timesleep = 30 if source == 'kline' else 0.1

        # print('source:', source, timesleep )
        while True:
            # 满足时间
            valid = self.isValidTime()
            if valid[0]:
                # 快照
                if source == 'snap':
                    callback(self.snap(codes))

                # 快照合成
                elif source == 'combin':
                    dk = self.snap(codes)
                    self.combin(dk, kmap=kmap)
                    callback(self.currentKline)

                # 即时K线
                elif source == 'kline':
                    d = {}
                    map = self.kline(codes, period=kPeriod, pre=1)
                    for c in map.keys():
                        if len(map[c]) > 0: d[c] = map[c].iloc[-1]

                    callback(d)

            # 非交易日和收盘则结束
            elif not valid[1]:
                break

            time.sleep(timesleep)

        # 晚盘结束，终止进程
        self.tickStatus = False

    def getValidDate(self, start=-3, end=20):
        start, end = public.getDate(diff=start), public.getDate(diff=end)

        return rq.get_trading_dates(start_date=start, end_date=end)


    def getValidDate1(self, start='19990-01-01', end='2002-01-01'):
        #start, end = public.getDate(diff=start), public.getDate(diff=end)
        return rq.get_trading_dates(start_date=start, end_date=end)

    def getLastDay(self):
        dd = str(public.getTime(style='%H:%M:%S'))
        valids = self.getValidDate(start=-15, end =0)
        if ('18:15:00' < dd < '23:59:59'):
            date = public.getDate()
        else:
            date = str(valids[-2])
        return date

    # 计算实际交易间隔时间
    def timeDiff(self, preDate, curDate=None, quick=0):
        #
        if curDate is None:
            curDate = public.getDatetime(style='%Y-%m-%d %H:%M:%S')

        preDate = str(preDate)
        if preDate.find('.') > -1:
            preDate = preDate[:preDate.find(".")]
        elif len(preDate) == 8:
            preDate = public.parseTime(preDate, format='%Y%m%d', style='%Y-%m-%d %H:%M:%S')

        diff = public.timeDiff(str(curDate), preDate) / 60
        # 快速比较
        if quick > 0 and diff < quick: return 1

        # 间隔小于120，为非跨区
        if diff < 120: return int(diff)
        # 间隔周末
        if diff > (24 * 60 * 2): diff -= 24 * 60 * 2

        # 日内跨区
        m0, m1 = (public.parseTime(d, style='%H:%M:%S') for d in [preDate, curDate])
        for ta in self.TimeDiffArea[:-1]:
            if m0 < ta[0] < m1:
                diff -= ta[1]

        if diff < 120: return int(diff)

        # 夜盘隔日
        date = public.parseTime(curDate, style='%Y-%m-%d')
        ta = self.TimeDiffArea[-1]
        taTime = str(date) + ' ' + ta[0]
        if preDate < taTime < curDate:
            diff -= ta[1]
        return int(diff)

    # 计算k线开始日期
    preDates = None
    def getPreDate(self, period='', num=60, start = None):
        # 查询结束时间和每日运行总分钟
        if num==0 : num = 1
        mins = 240
        if len(self.TimeArea) == 3: mins += self.TimeArea[2][2]
        p = int(period.replace('m', '').replace('d', ''))
        # 返回天数
        days = int(round(num * p / mins, 0)) if period.find('m') > -1 else num * p
        if self.preDates is None:
            self.preDates = dates = self.getValidDate(start= -2 * num, end=0)
        else:
            dates = self.preDates

        if start is None:
            return str(dates[-days])
        else:
            return str(dates[[str(c) for c in dates].index(start)-days])

    # 计算k线提前天数
    def getPreDays(self, period='', num=60):
        # 查询结束时间和每日运行总分钟
        mins = 240
        if len(self.TimeArea) == 3: mins += self.TimeArea[2][2]

        p = int(period.replace('m', '').replace('d', ''))

        # 返回天数
        days = int(round(num * p / mins, 0)) if period.find('m') > -1 else num * p

        # 跨周末+2
        weekDay = public.getWeekDay()
        if (days > weekDay) or (days > 4 and weekDay > 4): days += 2
        return days + 14

    # 设置结束时间和时限计算时间
    def setTimeArea(self, nightEnd):
        # 设置交易对的收盘时间和间隔区间
        self.TimeArea, self.TimeDiffArea = [], []
        for n in self.iniTimeArea: self.TimeArea.append(n)
        for n in self.iniTimeDiffArea: self.TimeDiffArea.append(n)

        date = public.getDate()
        if ("21:00:00" < nightEnd < "23:59:00") or ("00:00:00" < nightEnd < "05:00:00"):
            diff = int(public.timeDiff(date + ' ' + '09:00:00', date + ' ' + nightEnd[0:8]) / 60)
            # 夜盘隔日
            if diff < 0: diff = 24 * 60 + diff
            self.TimeArea.append(("21:00:00", nightEnd[0:8], 720 - diff))
            self.TimeDiffArea.append(("07:00:00", diff))
        else:
            self.TimeDiffArea[1] = ('16:00', 18 * 60)

    # 满足的时间段
    def isValidTime(self):
        if self.validDate is None:
            # 非交易日
            self.validDate = self.getValidDate()

        if not datetime.date.today() in self.validDate:
            return (False, False)

        # 交易时间
        ct = time.strftime('%H:%M:%S')
        for t in self.TimeArea[0:2]:
            if t[0] <= ct <= t[1]:
                return (True, True)

        # 下午收市
        if len(self.TimeArea) > 2:
            t = self.TimeArea[-1]
            if t[1] > t[0]:
                if ct >= t[0] and ct <= t[1]:
                    return (True, True)
                # 晚盘非隔日
                elif ct > t[1]:
                    return (False, False)
            else:
                # 晚盘隔日
                if ct >= t[0] or ct <= t[1]:
                    return (True, True)

                elif ct > t[1] and ct < '05:00:00':
                    return (False, False)

        # 下午盘收盘
        elif ct > t[1]:
            return (False, False)

        if int(time.time() * 10) / 18000 == 0:
            print("wait", ct, self.codes)
        return (False, True)

    # 回测获得tick数据，模拟生成tick真实交易数据
    def tickSim(self, td, e, ktype='1m'):
        s = e
        if ktype[-1]=='m':
           s = public.getDatetime(minutes=-int(ktype[:-1]), start=str(s))

        sub0 = td[s:e].iloc[:-1]

        sub = sub0[sub0['a1']!= 0.0]

        p = td[:s].iloc[-2]
        tick = {}
        for i in range(len(sub)):
            #if sub.ix[i,'a1']==0 or sub.ix[i,'b1']==0: continue
            if i == 0:
                tick = copy.deepcopy(sub.iloc[i])

                for item in ["open", "high", "low", "close"]:
                    tick[item] = tick["last"]

                p['start_volume'] = p['volume']
                #for key in ['a', 'b']:
                    #tick['raise_' + key] = []
                    #for j in range(-3, 0):
                       #p0, p1 = td[:s].iloc[j-1], td[:s].iloc[j]
                       #tick['raise_' + key].append(p1[key+'1']-p0[key+'1'])

            else:
                p = copy.deepcopy(tick)
                tick = copy.deepcopy(sub.iloc[i])
                close = tick['close'] = tick['last']

                tick['open'] = p['open']
                tick["high"] = close if p["high"] < close else p["high"]
                tick["low"] = close if p["low"] > close else p["low"]
                """
                # pop
                for key in ['a', 'b']:
                    tick['raise_' + key] = p['raise_' + key]
                    tick['raise_' + key].append(tick[key+'1']-p[key+'1'])
                    if len(tick['raise_' + key]) > 3:
                        tick['raise_' + key].pop(0)
                """
            # 交易量变化
            tick['start_volume'] = p['start_volume']
            tick['volume_m'] = tick['volume'] - p['start_volume'] if 'start_volume' in p else 0
            """
            #tick['volume_c'] = tick['volume'] - p['volume']
            #tick['raise'] = round(tick['last'] - p['last'], 2)

            tick['volume_mode'] = 0

            if tick['raise'] != 0:
                tick['volume_mode'] = int(np.sign(tick['raise']))
            elif tick['a1'] != p['a1']:
                 tick['volume_mode'] = 1 if tick['last'] == p['a1'] else -1 if tick['last'] == p['b1'] else 0
            else:
                 tick['volume_mode'] = 1 if tick['last'] == tick['a1'] else -1 if tick['last'] == tick['b1'] else 0
            """

            yield tick

    # 合成一分钟及时k线
    def combin(self, dk, kmap=None):
        if kmap is None:  return None
        # 按不同K线区间合并元素
        for code in dk.keys():
            ktype = kmap[code] if code in kmap else kmap[self.parseCode(code)]

            now = int((time.time()) / 60 / ktype)
            if code not in self.currentKline or self.currentKline[code]['now'] != now:
                # 第一根线
                #self.currentKline[code] = {}

                self.currentKline[code] = nd =  copy.deepcopy(dk[code])
                self.currentKline[code]['now'] = now
                #self.currentKline[code]['start_volume'] = nd['volume']
                for item in ["open", "high", "low", "close"]:
                    self.currentKline[code][item] = dk[code]["last"]
            else:
                p = copy.deepcopy(self.currentKline[code])
                close = dk[code]["last"]
                # 更新currentKline
                self.currentKline[code].update(dk[code])
                self.currentKline[code]["close"] = close
                #self.currentKline[code]['volumed'] += dk[code]["volume"] - p["volume"]
                self.currentKline[code]["now"] = now
                self.currentKline[code]["open"] = p['open']
                self.currentKline[code]["high"] = close if p["high"] < close else p["high"]
                self.currentKline[code]["low"] = close if p["low"] > close else p["low"]

    # 检查是否更新
    def isNewTick(self, dk):
        if self.preKline is None:
            self.preKline = dk
            return True
        else:
            for code in dk.keys():
                if self.preKline[code] != dk[code]:
                    self.preKline = dk
                    return True
        return False

class tick_csv_Rice(interface_Rice):
    # 合成一分钟及时k线
    def combin1(self, dk, isNew=False):
        c0, c1 = dk[self.codes[0]], dk[self.codes[1]]
        if isNew:
            self.currentKline = {}
            self.setNormal(c0, c1)
            for item in ["open", "high", "low"]:
                self.currentKline[item] = self.currentKline["close"]

        else:
            self.setNormal(c0, c1)
            # 更新currentKline
            p = self.currentKline
            close = p["close"]
            self.currentKline["high"] = close if p["high"] < close else p["high"]
            self.currentKline["low"] = close if p["low"] > close else p["low"]

    def setNormal(self, c0, c1):
        self.currentKline.update(c0)
        for key in c0.keys():
            self.currentKline['n_' + key] = c1[key]
        self.currentKline["close"] = c0["last"] / c1["last"]

#
def getAllMain():
    Rice = interface_Rice()
    codes = Rice.allFuture(isSave=True)
    print(codes)

# 更新夜盘收盘时间和最新价格
sql_nightEnd = """
    update future_baseInfo 
    set lastPrice = %s,
        lastVolume = %s
    where code = '%s' 
"""

def stand(ser):
    return ser / ser.abs().mean()

def updateNightEnd():
    Rice = interface_Rice()
    codes = Rice.allFuture(isSave=True)
    Base = future_baseInfo()
    mCodes = Rice.getMain(codes)

    # 更新一分钟成交量
    klines = Rice.kline(mCodes, start=public.getDate(diff=-5))
    for i in range(len(codes)):
        df = klines[mCodes[i]].dropna().reindex()
        print(i, codes[i], mCodes[i], len(df))
        if len(df) > 0:
            # night = public.parseTime(str(df.index[-1]), style='%H:%M:%S')
            lastPrice = df["close"].values[-1]
            lastVolume = df["volume"].mean()
            sql = sql_nightEnd % (str(lastPrice), str(lastVolume), codes[i])
            Base.update(sql)
            # break
        else:
            print('None')

def test():
    obj = interface_Rice()
    df = obj.allHisFuture(start='2010-01-01')

    print(df)

def getAllCodes(type):

    Rice = interface_Rice()
    res =  Rice.allCodes(type)
    print(res[res['order_book_id'].str.contains('000016') | res['order_book_id'].str.contains('399905') | res['order_book_id'].str.contains('399300')])


def main():
    action = {
        "main": 1,
        "update": 0,
        "tick":0
    }
    W = interface_Rice()
    if action["main"] == 1:
        Base = future_baseInfo()
        codes = Base.getUsedMap(hasIndex=1)
        for code in codes:
        #for code in ['NI']:
            print('---------', code)
            #res = rq.get_dominant_future(code, start_date=public.getDate(diff=-100), end_date=public.getDate(diff=1))
            res = rq.get_dominant_future(code, start_date='2019-03-30', end_date='2019-05-15')
            if res is None: continue
            pre = ''
            for i in range(len(res)):
                if pre != res[i]:
                    print(res.index[i], res[i])
                    pre = res[i]

    if action["update"] == 1:
        updateNightEnd()

    if action["tick"] == 1:
        c1 = 'SR1909'
        period = 'tick'
        start ='2019-05-14'
        end = '2019-05-14'

        res = rq.get_price(c1, frequency=period, fields=W.columns, start_date=start,
                     end_date=end, adjust_type='pre')

        file = W.basePath + '%s_tick.csv' % (c1)
        print(res)
        print(c1, '---------------------------- to_cvs', file )
        res['datetime'] = res.index
        res.to_csv(file, columns =['datetime', 'last'] + W.columns, index=0)

if __name__ == '__main__':
    test()
    #main()
