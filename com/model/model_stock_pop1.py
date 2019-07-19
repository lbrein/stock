# -*- coding: utf-8 -*-
"""
Created on 12 19 -2018
@author: lbrein

      zhao 基金长线策略： bias 》 1 或bias 《 -1
      日线，每分钟持续跟踪

"""

from com.base.public import public, logger
import numpy as np
import pandas as pd
import talib as ta
import uuid
from com.data.interface_Rice import interface_Rice
from com.object.obj_entity import stock_orderForm, stock_baseInfo
from com.object.mon_entity import mon_tick
from multiprocessing import Pool, Manager
import time
import traceback
import copy
from com.data.Interface_GuangFa import ProcessMap, GuangFa_interface
import os
from multiprocessing.managers import BaseManager

# 回归方法
class MyManager(BaseManager):
      pass

MyManager.register('interface_Rice', interface_Rice)

def Manager2():
    m = MyManager()
    m.start()
    return m



# 选股
class model_stock_pop(object):

    def __init__(self):

        self.isWorking = False # 是否正式运行
        self.banCodeList = []  # 暂时无权限操作的code
        self.isTickSave = True
        self.timeInterval = 0.01  # 间隔处理时间

        self.period = '1d'
        self.timePeriods = 60
        self.dropLine = 0.20

        self.sarStart = 0.035
        self.sarEnd = 0.035
        self.pageCount = 1

        self.iniAmount = 10000
        self.ratio = 0.0006

        self.TimeArea = [('09:30:00', "11:30:00"), ('13:00:00', '15:00:00')]

        self.orderFormTest = 'stock_orderForm'
        self.methodName = 'pop01'  # 策略名称
        self.sourceType = 'snap'
        self.iterCondList = ['timePeriods', 'dropLine', 'sarStart', 'sarEnd']

    def pool_filter(self):
        time0 = time.time()
        Base = stock_baseInfo()
        Base.iniBound()
        lists = Base.getCodes(isBound=0)
        pool = Pool(processes=4)
        for k in range(0, len(lists), self.pageCount):
            codes = lists[k: k + self.pageCount]
            self.subFilter(codes, k)
            #pool.apply_async(self.subFilter, (codes, k))

        pool.close()
        pool.join()

        # 添加持仓股票
        Base = stock_baseInfo()
        Record = stock_orderForm()
        pos = Record.getPosition(method=self.methodName, codes=None)
        Base.updateBound([pos[c]['code'] for c in pos.keys()])

        #logger.info(("model_stock_pop stock filter finished: ", len(lists), time.time() - time0))

    # 每天初选股票
    def subFilter(self, codes, k):
        print("stock filter subprocess start:", k/self.pageCount)
        Rice = interface_Rice()
        Base = stock_baseInfo()

        startDate = public.getDatetime(diff=-200)
        res = Rice.kline(codes, period=self.period, start=startDate, pre=90)

        period = 180
        line1 = 0.5
        codeList = []
        for code in codes:
            df = res[code]

            # 计算跌幅和回跌幅度
            close = df['close']
            mx0 = ta.MAX(close, timeperiod=60).values[-1]
            last = close.values[-1]

            opt0 = (mx0 - last) / mx0 > 0.2

            mx = close[-period:].max()
            mi = close[-period:].min()

            miw = ta.MININDEX(close, timeperiod=period).values[-1]
            mid = close[miw:].max()
            # 超过M5
            ma5 = ta.MA(close, timeperiod=5)

            opt1 = (mx - mi) / mx > line1 and (mid - mi) / (mx - mi) < 0.30 and (mx0 - last) / mx0 > 0.12
            opt2 = (last > ma5.values[-1] or last > ma5.values[-2])

            if (opt0) or (opt1 and opt2):
                codeList.append(code)

        print(k,len(codes),len(codeList))
        Base.updateBound(codeList)
        Base.closecur()

    # 每天初选股票
    def update_last(self):
        Rice = interface_Rice()
        Record = stock_orderForm()

        res = Record.getPosition(method=self.methodName, codes=None)
        codes = [res[c]['code'] for c in res.keys()]
        dfs = Rice.snap(codes)
        for code in codes:
            #print(code, dfs[code]['last'])
            Record.update_last(code, dfs[code]['last'], batchid = res[code]['batchid'])

        Record.closecur()

    def control(self):
        Rice = interface_Rice()
        Rice.TimeArea = self.TimeArea
        Base = stock_baseInfo()
        #lists = Base.getCodes(isBound=1)
        #print(lists[0], len(lists))
        lists = self.csvList
        logger.info(('model_stock_pop scan list:', len(lists)))

        while True:
            # 满足时间
            valid = Rice.isValidTime()
            tt = int(time.time()) // 3
            if valid[0]:
               self.pool(lists)

            # 非交易日和收盘则结束
            elif not valid[1]:
                break

            time.sleep(self.timeInterval)

    # 使用crontab的启动方式
    def crontab(self):
        Rice = interface_Rice()
        Rice.TimeArea = self.TimeArea
        valid = Rice.isValidTime()
        if valid[0]:
            self.pool()

    csvList = ['002907.XSHE', '000636.XSHE', '002137.XSHE', '600809.XSHG']
    def pool(self, lists):
        time0 = time.time()
        pool = Pool(processes=5)
        shareDict = Manager().dict({})

        for k in range(0, len(lists), self.pageCount):
            codes = lists[k:k+self.pageCount]
            In = False
            for c in codes:
                if c in self.csvList:
                    In = True
                    break
            if not In: continue

            self.start(codes, int(k/self.pageCount+1),shareDict)
            try:
                #pool.apply_async(self.start, (codes, int(k/self.pageCount+1)), shareDict)
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

        # 检查订单执行结果，更新status
        if self.isWorking:
             #print(("model_stock_pop confirm start: ", time.time() - time0))
             self.confirm()

        # 更新最新价格
        if int(time.time()/60) % 30 == 0:
            #print(("model_stock_pop update start: ", time.time() - time0))
            self.update_last()

        #logger.info(("model_stock_pop scan finished: 1111 ", len(lists), time.time()-time0))

    def confirm(self):
        G = GuangFa_interface()
        res = G.getMatch()
        Record = stock_orderForm()
        if res is not None and len(res) > 0:
            res = res[res['matchtype']==0]
            Record.confirm(res.to_dict(orient='records'))
    
    def start(self, full_codes, pid, shareDict={}):
        time0 = time.time()
        self.used_stocks = full_codes
        self.Record = stock_orderForm()

        if not self.isWorking:
           self.Record.tablename = self.orderFormTest

        if self.isTickSave:
           self.Tick = mon_tick()

        self.Rice = interface_Rice(isSub=True)
        # 进程控制类
        self.procMap = ProcessMap()

        self.shareDict = shareDict
        self.shareDict = self.iniShare(full_codes)

        # 初始化节点
        self.iniNode(full_codes)
        orders = self.onTick(self.Rice.snap(full_codes))

        if len(orders)>0:
            # 文件提交
            if self.isWorking:
                G = GuangFa_interface()
                G.stage = self.methodName
                res = G.order(orders)
                logger.info(([(d['code'], d['vol'],d['isBuy'], d['status']) for d in res]))

                self.Record.insertAll(res)

        # 子进程启动
        #logger.info(("model_stock_pop: subprocess finished", pid, " orders:", len(orders), " buy:", len([d['code'] for d in orders if d['isBuy']>0]), time.time()-time0))

        self.Record.closecur()
        self.Record = None

    def getUid(self, code):
        uid = code
        for key in self.iterCondList:
            uid +="_%s" % self.__getattribute__(key)
        return uid

    def iniShare(self, codes):
        if codes[0] not in self.shareDict:
            filename = self.Rice.basePath+'%s_%s.csv' % (codes[0], public.getDate())

            if os.path.exists(filename):
                for code in codes:
                    filename = self.Rice.basePath + '%s_%s.csv' % (code, public.getDate())
                    self.shareDict[code] = pd.read_csv(filename, index_col=0, encoding='gb2312')

            else:
                dfs = self.Rice.kline(codes, period=self.period, pre=200)
                for code in codes:
                    if code in dfs:
                        # 删除前一日文件
                        filename_1 = self.Rice.basePath + '%s_%s.csv' % (code, public.getDate(diff=-1))
                        if os.path.exists(filename_1):
                            os.remove(filename_1)
                        # 保存文件
                        filename = self.Rice.basePath + '%s_%s.csv' % (code, public.getDate())
                        dfs[code].to_csv(filename)
                        self.shareDict[code] = dfs[code]

        return self.shareDict

    # 初始化节点
    def iniNode(self, codes):
            openMap = self.Record.getPosition(method=self.methodName, codes=codes)
            for code in codes:
                self.procMap.new(code)
                if openMap is not None and code in openMap:
                     self.procMap.setIni(code, openMap[code])
                     #print('ini:', openMap[code])

    # Tick 响应
    def onTick(self, tick):
        # 计算参数
        orders = []
        self.cvsMap = []
        for code in self.used_stocks:
            if code in self.banCodeList: continue
            self.uid = self.getUid(code)
            #print(code)
            if code =='002124.XSHE':
                print(tick[code])
            self.procMap.setUid(self.uid, method=self.methodName)

            # 按时长读取k线数据
            if not code in self.shareDict:
                self.shareDict = self.iniShare(self.used_stocks)

            try:
                # 计算指标
                param = self.paramCalc(tick)
                if param is not None and not np.isnan(param["close"]):
                    # mongodb Record
                    if self.isTickSave:
                        self.debugT(param=param, method=self.methodName)

                    # 执行策略，并下单
                    doc = self.orderCheck(tick, param)
                    if doc is not None:
                        orders.append(doc)

            except Exception as e:
                print(traceback.format_exc())


        if self.isTickSave and len(self.cvsMap) > 0:
              #print(self.cvsMap)
              self.Tick.col.insert_many(self.cvsMap)

        return orders

    def point(self, row):
        drop, sm, sm5 = row['drop'], row['sarm'], row['sarm5']
        return 1 if (sm==1 and drop > 0.15) else -1 if sm5==-1 else 0

    def turn(self, mm, md, mode):
        return 0 if mm > 0 else 1 if mode * md > 0 else -1

     # 买入后的最高/最低价
    def getMax(self, df0, s, e, mode):
        if mode > 0:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'close'].max()
        else:
            return df0[(df0['datetime'] >= s) & (df0['datetime'] < e)].ix[:-1, 'close'].min()

    klinecolumns = ['high', 'open', 'volume', 'close', 'low']
    def paramCalc(self, tick):
        code = self.procMap.codes[0]
        c0 = tick[code] # 快照
        c0['close'] = c0['last']
        df0 = copy.deepcopy(self.shareDict[code])
        # 部分修正即时K线
        df0.loc[public.getDatetime()] = pd.Series([c0[key] for key in self.klinecolumns], index=self.klinecolumns)

        close = df0["close"]

        df0['max90'] = mx =  ta.MAX(close, timeperiod = 60)
        df0['drop'] = (mx - close) / mx

        sar = ta.SAR(df0['high'], df0['low'], acceleration=self.sarStart, maximum=0.2)
        df0['sard'] = sard = close - sar
        df0['sarm'] = sard * sard.shift(1)
        df0['sarm'] = df0.apply(lambda row: self.turn(row['sarm'], row['sard'], 1), axis=1)

        sar5 = ta.SAR(df0['high'], df0['low'], acceleration=self.sarEnd, maximum=0.2)
        df0['sard5'] = sard5 = close - sar5
        df0['sarm5'] = sard5 * sard5.shift(1)
        df0['sarm5'] = df0.apply(lambda row: self.turn(row['sarm5'], row['sard5'], 1), axis=1)

        df0['createTime'] = df0.index
        df0['mode'] = df0.apply(lambda row: self.point(row), axis=1)

        param = copy.deepcopy(df0.iloc[-1]).to_dict()
        param.update(c0)
        #if param['sarm']==1:  print(df0)
        #print(param)
        param.update({
            "code": code,
            "p_l": c0["asks"][0],
            "p_h": c0["bids"][0]
        })
        return param

    itemCount = 0
    cvsMap, cvsCodes = {}, []
    def debugT(self, param=None, method= None):
        if method is None:
            method = self.procMap.currentUid.split("_")[-1]

        code = self.procMap.codes[0]
        if param is not None:
            # param = param.to_dict()
            param['code'] = code
            param['method'] = method
            param['isOpen'] = self.procMap.isOpen

            for key in ['sarm', 'sarm5', 'mode']:
                if key in param: param[key] = int(param[key])
            self.cvsMap.append(param)
    #
    def orderCheck(self, tick, param):
            mode, close = (param[key] for key in "mode,close".split(","))
            isOpen, isRun = self.procMap.isOpen, False
            isBuy = 0
            pN = self.procMap.preNode
            # print(isOpen, pN)

            if isOpen ==0 and pN is None and mode == 1 :
                isBuy, isRun, mode = 1, True, mode

            elif isOpen ==1 and pN is not None and mode== -1:
                isBuy, isRun, mode = -1, True, mode

            if isRun:
                return self.order(param, isBuy, mode)

            return None

    def order(self, n0, isBuy, mode):
        pN = self.procMap.preNode
        if  pN is None:
            batchid = uuid.uuid1()
        else:
            batchid = pN['batchid']

        now = public.getDatetime()
        vol, fee, amount, income, p0 = 0, 0, 0, 0, 0
        price = n0["p_h"] if isBuy==1 else n0["p_l"]

        if isBuy > 0 :
            self.batchid = uuid.uuid1()
            p0 = price
            vol = int(self.iniAmount/p0/100) * 100
            amount = vol * p0
            fee = vol * p0 * self.ratio
            income = -fee

        elif isBuy < 0:
            p0 = price
            vol = pN['vol']
            amount = vol * p0
            fee = vol * p0 * self.ratio
            income = vol * (p0 - pN['price']) - fee

        doc = {
            "code": n0['code'],
            "name": n0['code'],
            "createdate": now,
            "price": p0,
            "vol": vol,
            "ini_price": p0,
            "ini_vol": vol,
            "mode": mode,
            "isBuy": isBuy,
            "fee": fee,
            "amount": amount,
            "income": income,
            "method": self.methodName,
            "batchid": batchid,
            "status": 1,
            "uid": self.procMap.currentUid
        }

        self.procMap.isBuy = isBuy
        # 设置上一个记录
        if isBuy > 0:
            self.procMap.preNode = doc
            self.procMap.batchid = uuid.uuid1()

        else:
            self.preNode = None

        #self.orders.append(doc)
        return doc

def main():
    actionMap = {
        "start": 1,  #
        "filter": 0,
        "confirm": 0,
        "last":0,
    }

    obj = model_stock_pop()

    if actionMap["start"] == 1:
        obj.control()

    if actionMap["filter"] == 1:
        obj.pool_filter()

    if actionMap["confirm"] == 1:
        obj.confirm()

    if actionMap["last"] == 1:
        obj.update_last()


if __name__ == '__main__':
    main()
