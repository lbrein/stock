# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 17:19:21 2016

EVENT_TIMER = 'eTimer'                  # 计时器事件，每隔1秒发送一次
EVENT_LOG = 'eLog'                      # 日志事件，全局通用

# Gateway相关
EVENT_TICK = 'eTick.'                   # TICK行情事件，可后接具体的vtSymbol
EVENT_TRADE = 'eTrade.'                 # 成交回报事件
EVENT_ORDER = 'eOrder.'                 # 报单回报事件
EVENT_POSITION = 'ePosition.'           # 持仓回报事件
EVENT_ACCOUNT = 'eAccount.'             # 账户回报事件
EVENT_CONTRACT = 'eContract.'           # 合约基础信息回报事件
EVENT_ERROR = 'eError.'
"""

from vnpy.trader.gateway.ctpGateway import CtpGateway
from vnpy.event.eventEngine import EventEngine
from vnpy.trader.vtEvent import *
from vnpy.trader.vtObject import VtOrderReq
from vnpy.trader.vtConstant import *

from com.object.obj_entity import future_baseInfo
from com.data.interface_Rice import interface_Rice

import re
import time


class interface_vnpy(object):
    order_map = {
        'price': 'price',
        'volume': 'vol',
        'exchange': 'exchange',
    }

    def __init__(self):
        self.ee = EventEngine()
        self.API = CtpGateway(self.ee)
        self.API.connect()
        self.ee.start()

        self.ee.register(EVENT_POSITION, self.onQryPosition)

        self.records = {}

    # ----------------------------------------------------------------------

    """发单"""

    def exit(self):
        self.API.close()
        self.ee.stop()

    def sendOrder(self, orderDoc, callback=None):

        orderReq = VtOrderReq()

        # 按匹配字段赋值
        for key in self.order_map.keys():
            orderReq.__setattr__(key, orderDoc[self.order_map[key]])

        # 查询获得ctp_Code
        orderReq.symbol = self.getCtpCode(orderDoc['code'])
        # 价格类型
        orderReq.priceType = PRICETYPE_MARKETPRICE
        # 买卖方向
        orderReq.direction = DIRECTION_LONG if orderDoc["mode"] == 1 else DIRECTION_SHORT
        # 平仓类型
        orderReq.offset = OFFSET_OPEN if orderDoc["isopen"] == 1 else OFFSET_CLOSE

        vid = self.API.tdApi.sendOrder(orderReq)
        print(orderReq.symbol, vid)
        self.printClass(orderReq)

        if callback is not None:
            self.ee.register(EVENT_TRADE + vid, callback)
            self.ee.register(EVENT_ORDER + vid, callback)
        else:
            self.ee.register(EVENT_TRADE + vid, self.onQryTrade)
            self.ee.register(EVENT_ORDER + vid, self.onQryOrder)


        # ----------------------------------------------------------------------

    def cancelOrder(self, cancelOrderReq):
        """撤单"""
        pass

        # ----------------------------------------------------------------------

    def qryAccount(self):
        """查询账户资金"""
        #self.ee.register(EVENT_ACCOUNT, self.onQryAccount)
        print('account')
        self.API.tdApi.qryAccount()
        # ------------------------      ----------------------------------------------

    def qryPosition(self):
        """查询持仓"""
        print('qryPosition')
        #self.ee.register(EVENT_POSITION, self.onQryPosition)

        self.API.tdApi.qryPosition()

        # -----------------------------------

    def qryOrder(self):
        """查询特定订单"""
        pass

    def qtyInstrument(self, callback=None):
        if callback is not None:
            self.ee.register(EVENT_CONTRACT, callback)
        else:
            self.ee.register(EVENT_CONTRACT, self.onQryContract)

        self.API.tdApi.reqID += 1
        self.API.tdApi.reqQryInstrument({}, self.API.tdApi.reqID)

    def onQryAccount(self, event):
        account = event.dict_['data']
        self.printClass(account)
        print('账户信息：', account.accountID, account.balance, account.available)

    def onQryTrade(self, event):

        data = event.dict_['data']
        print("trade response:", data)

    def onQryOrder(self, event):
        data = event.dict_['data']
        # print('---------------- order response:')
       # doc = self.printClass(data)
       #print(doc['InstrumentID'], doc['VolumeTotalOriginal'],doc[""])
        print('order response:', data.symbol, data.status)
        pass

    def onQryPosition(self, event):
        print('position')
        pos = event.dict_['data']
        doc = self.printClass(pos)
       # print(doc)
        pass

    list_codes = []

    def onQryContract(self, event):
        con = event.dict_['data']
        self.list_codes.append(con)
        self.printClass(con)
        last = event.dict_["last"]

        if last:
            print(len(self.list_codes))

    def onError(self, event):
        data = event.dict_['data']
        print(data.errorMsg)

    def printClass(self, obj):
        doc = {}
        for name in dir(obj):
            if name.find("__") > -1: continue
            value = getattr(obj, name)
            doc[name] = value
        return doc

    # 更新ctp Code
    def saveSymbol(self, event):
        con = event.dict_['data']
        sym = con.symbol
        t = ''.join(re.findall(r'[0-9]', sym))
        c = sym.replace(t, '')
        if c.upper() in self.symbolList and self.symbolList[c.upper()][-len(t):] == t:
            print(c, sym)
            self.Base.setCtpCode(c.upper(), c + '_' + str(len(t)), main=sym)

        last = event.dict_['last']
        if last:
            self.qtySymbolStatus = False

    symbolList = {}
    Base = None

    #
    def updateSymbol(self):
        if self.symbolList == {}:
            self.Base = future_baseInfo()
            Rice = interface_Rice()
            codes = [c[0] for c in self.Base.all(vol=0)]
            mCodes = Rice.getMain(codes)
            i = 0
            for code in codes:
                self.symbolList[code] = mCodes[i].replace(code, '')
                i += 1
            self.qtySymbolStatus = True

        self.qtyInstrument(callback=self.saveSymbol)
        while 1:
            if not self.qtySymbolStatus:
                self.API.close()
                self.ee.stop()
                break
            time.sleep(1)

    # 根据mCode，查询ctpCode
    def getCtpCode(self, mCode):
        t = ''.join(re.findall(r'[0-9]', mCode))
        c = mCode.replace(t, '')
        Base = future_baseInfo()
        doc = Base.getInfo([c])[0]
        ctp = doc['ctp_symbol'].split("_")
        sym = ctp[0] + t[-int(ctp[1]):]
        return sym

def order(code):
    Rice = interface_Rice()
    codes = [code]
    Base = future_baseInfo()
    config = [doc for doc in Base.getInfo(codes)][0]

    mcodes = Rice.getMain(codes)
    k = Rice.kline(mcodes)

    # print(k)
    doc = {
        "code": mcodes[0],
        "price": k[mcodes[0]]["close"].values[-1],
        "vol": 1,
        "mode": 1,
        "isopen": 1,
        "exchange": config["exchange"]
    }
    return doc


def updateSymbol():
    obj = interface_vnpy()
    obj.updateSymbol()


def main():
    action = {
        "symbol": 0,
        "order": 0,
        "account": 0,
        "pos": 1
    }

    if action["symbol"] == 1:
        updateSymbol()

    if action["order"] == 1:
        doc = order('MA')
        print(doc)
        obj = interface_vnpy()
        obj.sendOrder(doc)

    if action["pos"] == 1:
        obj = interface_vnpy()
        #time.sleep(4)
        obj.qryPosition()

    if action["account"] == 1:
        obj = interface_vnpy()
        obj.qryAccount()


if __name__ == '__main__':
    main()
