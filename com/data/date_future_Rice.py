# -*- coding: utf-8 -*-
"""
Created on  2018-01-04 
@author: lbrein
       ---  通用接口 ---      

   主要接口：
          米框期货行情接口
      122321312
-
"""

from com.base.public import public, logger
from com.object.obj_entity import future_baseInfo, future_orderForm
import pandas as pd
from com.data.interface_Rice import interface_Rice
from com.ctp.interface_pyctp import BaseInfo, interface_pyctp
from com.model.model_future_zhao_ctp_v1 import model_future_zhao_v1

from com.object.mon_entity import mon_tick
import numpy as np
import talib as ta
import copy
import uuid

from com.model.model_future_zhao_ctp_v1 import model_future_zhao_v1


class data_future_Rice(object):
    # 更新夜盘收盘时间和最新价格
    sql_nightEnd = """
        update future_baseInfo 
        set lastPrice = %s,
            lastVolume = %s
        where code = '%s' 
    """

    # 更新夜盘收盘时间和最新价格
    sql_width15 = """
        update future_baseInfo 
        set width15 = %s, slope=%s, range = %s , atr=%s 
        where code = '%s' 
        """

    ch_index = ['零','一','二','三','四','五','三特','无']
    trend_index= ['无','多', '空']

    def __init__(self):
        self.indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]

        mZhao = model_future_zhao_v1()
        self.banCodeList = mZhao.banCodeList  # 暂时不操作的code（不列交易量低的)
        self.longCodeList = mZhao.longCodeList  # 只做多仓的list
        self.shortCodeList = mZhao.shortCodeList # 只做空仓的list

        self.oneCodeList = ['SC', 'IH', 'IF', 'IC']  # 最低为1手单的
        self.columns_posiDetail = ['InstrumentID', 'Direction', 'OpenPrice', 'Volume', 'CloseProfitByTrade',
                          'PositionProfitByTrade',
                          'ExchangeID', 'OpenDate']
        #
    def getAllMain(self):
        Rice = interface_Rice()
        codes = Rice.allFuture(isSave=True)
        print(codes)

    def k_fix(self, row, mode):
        close, open, high, low = (row[key] for key in ['close', 'open', 'high', 'low'])
        d0 = abs(close - open) / open
        lim, rate = 0.003, 0.075
        if mode == 1:
            if close > open and open != 0:
                trend = 1 if high == open else abs(close - open) / (high - open)
                opt = (d0 < lim and trend < rate) or (d0 > lim and trend < rate * 2)
                return close if opt else high
            else:
                return high

        elif mode == -1:
            if close < open and open != 0:
                trend = 1 if low == open else abs(open - close) / (open - low)
                opt = (d0 < lim and trend < rate) or (d0 > lim and trend < rate * 2)
                return close if opt else low
            else:
                return low

    # 买入后的最高/最低价
    def getMax(self, df0, s, e, mode):
            s = str(s)[:10]
            if mode > 0:
                ss = df0[(df0['datetime'] >= s) & (df0['datetime'] <= e)]
                mm = ss.ix[:-1, 'close'].max()
                if np.isnan(mm): print(s, e,  ss)

                return df0[(df0['datetime'] >= s) & (df0['datetime'] <= e)].ix[:-1, 'close'].max()
            else:
                return df0[(df0['datetime'] >= s) & (df0['datetime'] <= e)].ix[:-1, 'close'].min()

    def isout0(self, row):
        close, ma20, ma55 = (row[key] for key in "close,ma20,ma55".split(","))
        return 0 if np.isnan(ma55) else 1 if close > ma55 else -1

    def isout(self, row, pos):
        close, ma20, ma55, trend = (row[key] for key in "close,ma20,ma55,trend".split(","))
        pt = pos if pos != 0 else trend if trend != 0 else 1 if ma20 < ma55 else -1
        return 1 if close > ma55 and pt > 0 else -1 if close < ma55 and pt < 0 else 0

    def isMam(self, row):
        #
        mad, mac, mac2, mac3 = (row[key] for key in "mad,mac,mac2,mac3".split(","))
        opt = np.isnan(mad) or mad > 0 or mac == 0
        # 为零时同向偏转
        opt1 = (mad ==0 and (mac * mac2 > 0 or (mac2==0 and mac * mac3) > 0))
        return 0 if (opt or opt1) else 1 if mac > 0 else -1

    # 每日自动计算ATR和最新
    def autoCreateAtr(self, type=0):
        Rice = interface_Rice()
        Rice.setIndexList(self.indexCodeList)
        Base = future_baseInfo()
        Record = self.Record = future_orderForm()
        Model = model_future_zhao_v1()


        methods = ['mZhao', 'mZhao55']
        self.iniAmount, self.stopLine = 15400000, 0.0025
        if type == 1 :
            methods = ['zhao', 'zhao55']
            self.iniAmount, self.stopLine = 20000000, 0.0025

        codes = Base.getUsedMap(hasIndex=True)
        BI = BaseInfo(codes)
        mCodes = Rice.getMain(codes)
        #print(codes)

        end = None
        dd = str(public.getTime(style='%H:%M:%S'))
        valids = Rice.getValidDate(start=-15, end=0)

        if ('18:15:00' < dd < '23:59:59'):
            end = public.getDate(diff=0)
        else:
            end = str(valids[-2])

        dfs = Rice.kline(mCodes, period='1d', start=public.getDate(diff=-150), end = end, pre=20)
        docs = []

        Tmap = Record.trendMap(methods)

        Pos = []
        j = 0
        for m in methods:
            Pos.append(Record.getOpenMap(method=m,  batchNum=1))

        for mcode in mCodes:
            code = BI.parseCode(mcode)
            if code in self.banCodeList: continue

            df = dfs[mcode]

            close = df['close']
            df["datetime"] = df.index

            df["ma10"] = ma10 = ta.MA(close, timeperiod=10)
            df["ma20"] = ma20 = ta.MA(close, timeperiod=20)
            df["ma55"] = ta.MA(close, timeperiod=55)

            atr21 = ta.ATR(df['high'], df['low'], close, timeperiod=21)
            df['3atr'] = atr21 * 3

            # 计算ma10-ma20 穿越线间距
            df['mac'] = mac = ma10 - ma20
            df['mac2'] = mac.shift(2)
            df['mac3'] = mac.shift(3)

            # isPoint
            df['mad'] = mac * mac.shift(1)
            df['mam'] = mam = df.apply(lambda row: self.isMam(row), axis=1)
            minidx, maxidx = ta.MINMAXINDEX(mam, timeperiod=75)
            df['interval'] = abs(minidx - maxidx)

            # 修正不正常K线
            df['high'] = df.apply(lambda row: self.k_fix(row, 1), axis=1)
            df['low'] = df.apply(lambda row: self.k_fix(row, -1), axis=1)

            # 唐奇安线18日
            df['tu_s'] = ta.MAX(df['high'], timeperiod=18-1)
            df['td_s'] = ta.MIN(df['low'], timeperiod=18-1)

            # 唐奇安线27日
            df['tu_s1'] = ta.MAX(df['high'], timeperiod=27-1)
            df['td_s1'] = ta.MIN(df['low'], timeperiod=27-1)

            # 唐奇安线34日
            df['tu_34'] = ta.MAX(df['high'], timeperiod=33)
            df['td_34'] = ta.MIN(df['low'], timeperiod=33)

            # 40日低点
            ld = close[close.notnull()]
            p = 40 if len(ld) > 40 else len(ld) - 1
            df['tu_d'] = ta.MAX(df['high'], timeperiod=p-1)
            df['td_d'] = ta.MIN(df['low'], timeperiod=p-1)

            fp, fd = 27, 5

            # 计算穿越值
            out = df.apply(lambda row: self.isout0(row), axis=1)
            df['out_s'] = ta.SUM(out, timeperiod=fp)
            df['trend'] = df['out_s'].apply(lambda x: -1 if x>fd else 1 if x<-fd else 0)

            #posTrend = 0 if code not in trendMap else trendMap[code]['trend']
            df['isout'] = isout = df.apply(lambda row: self.isout(row,0), axis=1)
            df['isout3'] = ta.SUM(isout, timeperiod=3)
            df['isout5'] = ta.SUM(isout, timeperiod=5)

            param = copy.deepcopy(df.iloc[-1]).to_dict()
            isLong,type = 1, 0
            if code in Tmap:
                isLong = Tmap[code]['trend']
                type = 1
            elif param['trend']!=0:
                isLong = param['trend']
                type = 2
            else:
                isLong = -1 if param['ma20'] > param['ma55'] else 1
                type = 3

            j+=1
            isL = -1 if code in self.shortCodeList else 1 if code in self.longCodeList else 0

            if code in []:
                print(code, isLong, isL, param['trend'])
                file = Rice.basePath + '%s_%s_%s.csv' % (code, public.getDatetime(style='%Y%m%d_%H%M%S'),methods[0])
                df.to_csv(file, index=0)

            # 计算交易手数
            mul = BI.att(code, "contract_multiplier")
            dp = param['td_d'] if isLong > 0 else param['tu_d']
            p18 = param['tu_s'] if isLong > 0 else param['td_s']
            p27 = param['tu_s1'] if isLong > 0 else param['td_s1']

            if np.isnan(param['ma55']) or np.isnan(p18) :
                print('period no long:', code)
                continue

            ma20v_18 = (self.iniAmount * self.stopLine / abs(p18 - dp) / mul)
            ma20v_18 = int(ma20v_18 + 0.2)

            ma20v_27 = (self.iniAmount * self.stopLine / abs(p27 - dp) / mul)

            ma20v_27 = int(ma20v_27 + 0.2)
            #
            ma55v = (self.iniAmount * self.stopLine / param['3atr'] / mul)
            ma55v = int(ma55v + 0.2)

            # 固定一手交易
            if code in self.oneCodeList:
                ma55v = ma20v_18 = ma20v_27 = 1

            # 计算持仓和止损
            p=np.zeros((2,3))
            i = 0
            mp = 0

            for pos in Pos:
                if code in pos:
                   d = pos[code][0]

                   # 最近高点
                   sign = np.sign(int(d['mode']))
                   p[i][0] = sign * d['hands']
                   p[i][2] = mul * d['hands'] * (param['close'] - d['price']) * sign

                   if i==0:
                       p[i][1] =  param['td_d'] if sign > 0 else param['tu_d']
                   else:
                       mp = self.getMax(df, d['createdate'], public.getDate(diff=1), d['mode'])
                       if np.isnan(mp):
                           print('no max Price:', code, mp)

                       p[i][1] = round(mp, 1) - sign * round(param['3atr'], 1)
                i+=1

            param.update({
                    'code': mcode,
                    '方向': '多' if isLong==1 else '空',
                    'price_18':p18,
                    'vol_18': ma20v_18,
                    'price_27': p27,
                    'vol_27': ma20v_27,
                    'vol_55': ma55v,
                    '乘数': mul,
                    '3ATR': round(param['3atr'], 1),
                    '系统1持仓': p[0][0],
                    'price_40': dp,
                    '40日止损价': p[0][1],
                    '浮盈1': p[0][2],
                    '系统2持仓': p[1][0],
                    '最高点': mp,
                    '3ATR止损价': p[1][1],
                    '浮盈2': p[1][2],
                    '状态': Model.getStatus(methods, code),
                    '指定方向': Model.getTrend(code),
            })
            docs.append(param)

        res = pd.DataFrame(docs, columns=['code', 'close', '方向', 'price_40', 'price_18', 'vol_18', 'price_27' ,'vol_27', '3ATR',
                                          'vol_55', '乘数', '系统1持仓', '40日止损价', '浮盈1','系统2持仓','最高点', '3ATR止损价','浮盈2', '状态', '指定方向'])

        res = res.sort_values('code', ascending=True)
        file = Rice.basePath + 'future_%s_%s.csv' % (public.getDate(),methods[0])
        res.to_csv(file, index=0)
        logger.info(('autoCreateAtr finished:', len(docs)))
        return res

    def getStatus(self, methods, code):
        posMode = self.Record.openMode(methods, code)
        lastStop = self.Record.lastStop(methods, code)
        trend = self.getTrend(code)

        status = -1
        if posMode[1] == 0:
            if lastStop[0] == 6 or (lastStop[0] == 0 and trend == 0):
               status = 0

            elif lastStop[0] in [3, 5] or (lastStop[0] == 0 and trend != 0):
               status = 1

            elif lastStop[0]==2:
               status = 2

        elif posMode[1] == 1 :
            if posMode[0].find('55') > -1:
                status = 5

            elif lastStop[0] == 3:
                status = 3.5

            else:
                status = 3

        elif posMode[1] == 2:
             status = 4

        return status

    def getTrend(self, code):
        return 1 if code in self.longCodeList else -1 if code in self.shortCodeList else 0

    def getMaps(self):
        orderMap, codes, mCodes = {}, [], []
        b = BaseInfo([])
        obj = interface_pyctp(baseInfo=b, userkey='zhao')
        res = obj.qryPositionDetail()
        methodName = 'mZhao'
        codes55 = ['J1905', 'Y1905', 'P1905', 'NI1905', 'HC1905']
        for r in res:
            if r['Volume'] > 0:
                s = r['InstrumentID'].decode('gbk')
                if s not in orderMap:
                    m = b.parseMCode(s)
                    c = b.parseCode(m)
                    order = {
                        "symbol": s,
                        "code": m,
                        "mode": 1 if r['Direction'].decode('gbk') == '0' else -1,
                        "isopen": 1,
                        "hands": r['Volume'],
                        "price": r['OpenPrice'],
                        "method": methodName,
                        "createdate": public.parseTime(r['OpenDate'].decode('utf-8'), format='%Y%m%d',
                                                       style='%Y-%m-%d %H:%M:%S')
                    }
                    orderMap[s] = order
                else:
                    orderMap[s]["hands"] += r['Volume']

        orders = []
        for key in orderMap.keys():
            doc = orderMap[key]
            if doc['code'] in codes55:
                v = doc['hands']
                v0 = int(v / 2)
                doc['hands'] = v0
                orders.append(doc)

                # method55
                v1 = v - v0
                doc1 = copy.deepcopy(doc)
                doc1['hands'] = v1
                doc1['method'] = 'mZhao55'
                orders.append(doc1)
            else:
                orders.append(doc)
        return orders

    def getMaps2(self, mapList):
        codes = [c[0] for c in mapList]
        orders = []
        mCodes = self.Rice.getMain(codes)
        snaps = self.Rice.snap(mCodes)
        i = 0
        for m in mCodes:
            s = self.BI.ctpCode(m)
            order = {
                "symbol": s,
                "code": m,
                "name": mapList[i][0],
                "mode": mapList[i][1],
                "isopen": mapList[i][2],
                "hands": mapList[i][3],
                "price": snaps[m]['last'] if  len(mapList[i])<6 else mapList[i][5],
                "method": mapList[i][4] ,
                "createdate": public.getDatetime(diff=0),
            }
            orders.append(order)
            i += 1
        return orders

    # 平仓
    def closeFuture(self, id='278231', price=5059, isstop=2, date = None):
        self.Record = future_orderForm()
        self.BI = BaseInfo([])

        doc = self.Record.getById(id)
        docnew = copy.deepcopy(doc)
        base = self.BI.doc(doc['name'])
        amount = price * doc['hands'] * base['contract_multiplier']
        r = base ['ratio']
        fee =  doc['hands'] * r if r >= 0.5 else amount * r
        docnew.update({
           "isopen": 0 ,
           "isstop": isstop,
           "price": price,
           "mode": -doc["mode"],
           "vol": doc['hands'] * base['contract_multiplier'],
           "status": 6,
           "fee": fee,
           "createdate": public.getDatetime() if date is None else date,
           "income": np.sign(doc["mode"]) * (price - doc['price']) * doc['hands'] * base['contract_multiplier'] - fee,
            "memo": ''
        })
        print(docnew)
        self.Record.insert(docnew)

        # 平仓

    def newFuture(self, id='278231', price=5059, mode=2, hands=5, date = None,
                  test=False, new_method=None, mcode=None, memo=''):

        self.Record = future_orderForm()
        self.BI = BaseInfo([])

        doc = self.Record.getById(id)
        docnew = copy.deepcopy(doc)

        base = self.BI.doc(doc['name'])

        amount = price * hands * base['contract_multiplier']
        r = base['ratio'] * 1.1

        fee = hands * r  if r >= 0.5 else amount * r

        del docnew['id']
        name = doc['name']

        # 替换method
        if new_method is None:
            mode_old = doc['mode']
            uid, method = doc['uid'], doc['method']
            m0 = method[:-2]  if method.find('55')!=-1 else method+'55'
            print(mode_old, uid, m0)
            if (abs(mode_old) < 5 and abs(mode) > 4) or (abs(mode_old) > 4 and abs(mode) <5):
                uid = uid.replace(method, m0)
                method = method.replace(method, m0)

        else:
            uid, method = doc['uid'], doc['method']
            uid = uid.replace(method, new_method)
            method = new_method

        docnew.update({
            "code": self.BI.mCode(name) if mcode is None else mcode,
            "symbol": self.BI.ctpCode(name),
            "isopen": 1,
            "isstop": 0,
            "price": price,
            "mode": mode,
            "hands": hands,
            "ini_hands": hands,
            "ini_price": price,
            "vol": hands * base['contract_multiplier'],
            "status": 6,
            "fee": round(fee, 2),
            "batchid": str(uuid.uuid1()),
            "createdate": public.getDatetime() if date is None else date,
            "income": round(-fee,2),
            "method":method,
            "uid":uid,
            "memo": memo
        })
        print(docnew)
        if not test:
            self.Record.insert(docnew)


    def alterFuture(self, id='278231', price0=0, price1=0, mCode = None):
        # 平仓
        self.closeFuture(id=id, price=price0, isstop =4)
        # 开仓
        Rice = interface_Rice()
        doc = self.Record.getById(id)
        docnew = copy.deepcopy(doc)

        if mCode is None:
            m = Rice.getMain([doc['name']])[0]
        else:
            m = mCode

        docnew.update({
            "price": price1,
            "ini_price": price1,
            "code": m,
            "symbol":self.BI.parseCtpCode(m),
            "status":6,
            #"createdate": str(doc['createdate']),
            "batchid": str(uuid.uuid1())
        })
        print(docnew)
        self.Record.insert(docnew)

    def orderStart(self):
        self.BI =  BaseInfo([])
        self. Rice = interface_Rice()
        self.Record = future_orderForm()
        self.Record.tablename = 'future_orderForm_1'
        self.Rice.setIndexList([('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')])

        map =[
            ['AP', 1, 1, 5, 'mZhao', 11087],
            #['IH', 1, 1, 1, 'mZhao55']
              ]
        orderMap = self.getMaps2(map)
        self.addOrder(orderMap)

    #
    def addOrder(self, orderMap):
        res = []
        for order in orderMap:
            m = order['code']
            dfs = self.Rice.kline([m], period='1d', start=public.getDate(diff=-100), pre=10)
            df0 = dfs[m]

            # 计算40天最小值
            df0['high'] = df0.apply(lambda row: self.k_fix(row, 1), axis=1)
            df0['low'] = df0.apply(lambda row: self.k_fix(row, -1), axis=1)
            period = 40 if len(df0) >= 40 else len(df0)
            print(m, period)
            mx = ta.MAX(df0['high'], timeperiod=period).values[-1]
            mi = ta.MIN(df0['low'], timeperiod=period).values[-1]
            dp = mx if order['mode'] < 0 else mi

            ra, mul = self.BI.att(m, 'ratio'), self.BI.att(m, 'contract_multiplier')

            fee0 = (order['hands'] * ra) if ra > 0.5 else (mul * order['hands'] * order['price'] * ra)
            doc = copy.deepcopy(order)
            doc.update({
                "vol": order['hands'] * mul,
                "fee": fee0,
                "ini_hands": order['hands'],  # 提交单数
                "ini_price": order['price'],  # 提交价格
                "isstop": 0,
                "income": - fee0,
                'stop_price': dp,
                "batchid": uuid.uuid1(),
                "status": 6,  # 定单P执行CT返回状态
                "uid": '%s_40_2.0_1_0_%s' % (self.BI.parseCode(m), order['method'])
            })

            res.append(doc)

        self.Record.insertAll(res)

    def getMonTick(self, codes=None, method='dema5', num=4000):
            Tick = mon_tick()
            Rice = interface_Rice()
            if codes is None: codes = ['MA', 'A']
            diff = -1
            for c in codes:
                docs = Tick.getTick(c, count=num, method=method)
                print(c, len(docs))
                if len(docs) > 0:
                    f = Rice.basePath + "%s_%s.csv" % (c, public.getDate())
                    r = [d for d in docs]
                    r.reverse()
                    df = pd.DataFrame(r)
                    df.drop(['_id'], axis=1, inplace=True)
                    try:
                        df.to_csv(f, index=0)
                        print("%s  output" % f)
                    except:
                        continue

    def getDf(self, codes, period='1d'):
        Rice = interface_Rice()
        Rice.setIndexList(self.indexCodeList)
        Base = future_baseInfo()
        BI = BaseInfo([])
        if len(codes)==0:
            codes = Base.getUsedMap(hasIndex=True)

        mCodes = Rice.getMain(codes)
        dfs = Rice.kline(mCodes, period=period, start=public.getDate(diff=-10), pre=200)
        i = 0

        for mcode in mCodes:
            c = codes[i]
            last = BI.att(c, 'nightEnd')[0:6].replace(':', '')
            file = Rice.basePath + 'future_%s_%s_%s.csv' % (mcode, last, public.getDatetime(style='%Y%m%d_%H%M%S'))
            print(mcode,last)
            df = dfs[mcode]
            df['datetime'] = df.index

            df0 = df[df['datetime'] > '2019-01-17 13:47:40.000']
            print(df0.index, len(df0))

            #df.to_csv(file, index=0)
            i += 1
            break

    def compare(self, type=''):

        if type== 'm':
            user = 'zhao'
            methods = ['mZhao', 'mZhao55']
        else:
            user = 'fz'
            methods = ['zhao', 'zhao55']

        b = BaseInfo([])
        Ctp = interface_pyctp(baseInfo=b, userkey=user)
        map =Ctp.posMap

        Rice = interface_Rice()
        Orders = future_orderForm()
        posMap={}

        for pos in Orders.posByCode(methods):
            posMap[pos[0]] = pos[1]

            if pos[0] in map and pos[1] == map[pos[0]][0]:
                print('match ', pos, map[pos[0]])
            else:
                if pos[0] in map:
                    print('unmatch ', pos, map[pos[0]])
                else:
                    print('no purchase', pos)

            # 检查是否调仓
            pCode, name = pos[2], pos[0].split("_")[0]
            mCode = Rice.getMain([name])[0]
            if pCode != mCode:
                print(' --------- Need alter position:', pCode, mCode)

        for key in map:
            if len(key) < 6 and key not in posMap:
                print('no record', key, map[key])

        print(len(posMap.keys()))

def test():
    obj = interface_Rice()
    codes= ['CU1903', 'AU1906', 'IH1902', 'IF1902', 'IC1903']
    indexCodeList = [('IH', '000016.XSHG'), ('IF', '399300.XSHE'), ('IC', '399905.XSHE')]
    res = obj.setIndexList(indexCodeList)
    ss = obj.snap(codes)

    print(ss)

def getAllCodes(type):
    Rice = interface_Rice()
    res =  Rice.allCodes(type)
    print(res[res['order_book_id'].str.contains('000016') | res['order_book_id'].str.contains('399905') | res['order_book_id'].str.contains('399300')])


def main():
    action = {
        "update": 0,
        "tick": 0,
        "atr":0 ,
        "order":0,
        "close":0,
        "alter":0,
        "new":0,
        "kline":0,
        "compare":1,
    }

    obj = data_future_Rice()
    if action["tick"] == 1:
        obj.getMonTick(
            #codes=['A', 'SC', 'J', 'JD', 'SF', 'ZC', 'P', 'BU', 'L', 'PP', 'AL', 'RU', 'AU', 'AP', 'FG', 'CU', 'TA',
            #       'RB', 'ZN', 'HC', 'I', 'OI', 'CF', 'M', 'MA', 'SR', 'SF'], method='dema5')
            codes=['SC'], method="fellow")
        # getMonTick(codes=['510050.XSHG'])

    if action["atr"] == 1:
        for type in [0, 1]:
          obj.autoCreateAtr(type=type)

    if action["order"] == 1:
           obj.orderStart()

    if action["kline"] == 1:
           obj.getDf([])

    if action["new"] == 1:
           obj.newFuture(id='279749', price=2426, mode = -1 , hands= 19, date='2019-07-05 22:27:14',
                         test= False
                         #new_method = 'fx_dch', mcode='IF1909', memo='方兴空(方兴+产品)'
                         )

    if action["close"] == 1:
        obj.closeFuture(id='279714', price=4677, isstop=5, date=None)

    if action["alter"] == 1:
           obj.alterFuture(id='278298', price0=3023.4, price1=3033, mCode='IH1905')

    if action["compare"] == 1:
           obj.compare(type='m')

if __name__ == '__main__':
    main()
    #test()
