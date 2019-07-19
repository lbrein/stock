# -*- coding: utf-8 -*-
"""



"""
from com.base.public import config_ini, public, logger, public_basePath
import PyCTP
from com.ctp.order_trader import Order, OrderTrader
from com.object.obj_entity import future_baseInfo
from com.data.interface_Rice import interface_Rice, rq
import re
import time, datetime
import copy
import uuid
import numpy as np

"""
    --- 过程操作类，与uid 绑定 
"""
import PyCTP

class ProcessMap(object):
    modelParamMap = {
        "paird2": [['widthTimesPeriod', 'widthline', 'scaleDiff2', 'scaleDiff'], [3, 0.03, 0.5, 0.1]],
        "single_15": [['widthTimesPeriod', 'widthline', 'scaleDiff2', 'scaleDiff'], [3, 0.03, 0.5, 0.1]],
        "dema5": [['widthTimesPeriod', 'superline', 'turnline'], [3, 3.25, 2.0]],
        "dema6": [['widthTimesPeriod', 'powline', 'turnline'], [3, 0.25, 2.0]],
        "zhao": [['widthTimesPeriod'], [3]],
        "zhao55": [['widthTimesPeriod'], [3]],
        "fellow": [['bullLine', 'atrLine'], [3.5, 2]]
    }

    def __init__(self):
        self.map = {}
        self.currentUid = None
        self.codes = []  # 代码
        self.period = 0  # 布林带窗口大小
        self.scale = 0  # 标准差倍数
        self.kline = 0  # K线类型
        self.widthline = 0.05
        self.scaleDiff2 = 0.5

        # 过程变量
        self.atr = 0
        self.powm = 0

    def new(self, uid):
        self.currentUid = uid
        self.map[uid] = {
            "isOpen": 0,  # 策略状态 1-买多 -1 -买空 0-平
            "batchid": '',
            "status": 0,  # 进程状态
            "preNode": None,  # 之前的节点
            "mCodes": None,
            "ctpCodes": None,
            "interval": 0
        }

    @property
    def isOpen(self):
        return self.map[self.currentUid]['isOpen']

    @isOpen.setter
    def isOpen(self, value):
        self.map[self.currentUid]['isOpen'] = value

    @property
    def status(self):
        return self.map[self.currentUid]['status']

    @status.setter
    def status(self, value):
        self.map[self.currentUid]['status'] = value

    @property
    def preNode(self):
        return self.map[self.currentUid]['preNode']

    @preNode.setter
    def preNode(self, value):
        self.map[self.currentUid]['preNode'] = value

    @property
    def batchid(self):
        return self.map[self.currentUid]['batchid']

    @batchid.setter
    def batchid(self, value):
        self.map[self.currentUid]['batchid'] = value

    def get(self, name, uid=None):
        if uid is None: uid = self.currentUid
        if not uid in self.map.keys(): self.new(uid)
        return self.map[uid][name] if name in self.map[uid] else None

    # 设置record状态
    def setStatus(self, docs, status):
        for d in docs:
            d['status'] = status

    def set(self, name, value, uid=None):
        if uid is None: uid = self.currentUid
        if uid not in self.map.keys(): self.new(uid)
        self.map[uid][name] = value

    def setModelParam(self, map, modelname, num):
        if modelname in self.modelParamMap:
            params, defaults = self.modelParamMap[modelname][0], self.modelParamMap[modelname][1]
            for i in range(0, len(params)):
                j = num + 3 + i
                try:
                    self.__setattr__(params[i], float(map[j]) if map[j].find('.') > -1 else int(map[j]))
                except:
                    self.__setattr__(params[i], defaults[i])

    def setUid(self, map, num=2, params=None):
        """ 设置Uid """
        uid = '_'.join([str(n) for n in map])  # uid
        self.currentUid = uid
        self.codes = map[0:num]  # 代码

        if params is None:
            params = ['period', 'scale', 'kline']  # 布林带窗口大小, 标准差倍数, K线类型

        for i in range(len(params)):
            self.__setattr__(params[i], float(map[num + i]) if map[num + i].find('.') > -1 else int(map[num + i]))

        # 不同模型设置参数
        modelName = uid[uid.find('quick_') + 6:] if uid.find('quick_') > -1 else uid.split("_")[-1]
        self.setModelParam(map, modelName, num)
        return uid

    def setIni(self, uid, docs, status=0):
        """ 初始化节点 """
        if not uid in self.map.keys(): self.new(uid)

        self.map[uid]['isOpen'] = docs[0]['mode']
        self.map[uid]['batchid'] = docs[0]['batchid']
        self.map[uid]['preNode'] = docs
        self.map[uid]['status'] = status
        return [d['code'] for d in docs]

class BaseInfo(object):
    # code类，实现code,mainCode和ctp-Code互换，同时读取baseInfo数据
    def __init__(self, codes=[], Rice=None):
        self.Instruments = [self.parseCode(c) for c in codes]
        self.InstrumentMap = {}
        self.Rice = Rice if Rice else interface_Rice()

        self.config()
        cs = self.Instruments
        self.nightEnd = self.att(cs[-1], 'nightEnd')  # 收盘时间
        self.mainCodes = [self.att(c, 'mCode') for c in cs]  # 所有主力合约
        self.ctpCodes = [self.att(c, 'ctp') for c in cs]  # 所有ctpCode

    def config(self):
        # baseConfig信息
        Base = future_baseInfo()
        tmp = []
        for doc in Base.getInfo(self.Instruments):
            self.InstrumentMap[doc['code']] = doc
            tmp.append(doc['code'])

        # 空所有
        if len(self.Instruments) == 0:
            self.Instruments = tmp

        # 根据code返回主力code，同时生成ctp-code map
        for m in self.Rice.getMain(self.Instruments):
            c = self.parseCode(m)
            self.InstrumentMap[c]['mCode'] = m
            self.InstrumentMap[c]['ctp'] = self.parseCtpCode(m)

    @property
    def map(self):
        return self.InstrumentMap

    def att(self, code, name):
        c = self.parseCode(code)
        return self.InstrumentMap[c][name]

    def all(self):
        for c in self.Instruments:
            yield self.InstrumentMap[c]

    def doc(self, code):
        c = self.parseCode(code)
        return self.InstrumentMap[c]

    def mCode(self, code):
        return self.att(code, 'mCode')

    def ctpCode(self, code):
        return self.att(code, 'ctp')

    def parseCtpCode(self, mCode):
        # mCode 转换 ctpCode
        t = ''.join(re.findall(r'[0-9]', mCode))
        c = mCode.replace(t, '')
        ctp = self.InstrumentMap[c]['ctp_symbol'].split("_")
        sym = ctp[0] + t[-int(ctp[1]):]
        return sym

    def parseMCode(self, ctpCode):
        # ctpCode 转换 mCode
        return rq.id_convert(ctpCode)
        #t = ''.join(re.findall(r'[0-9]', ctpCode))
        #c = ctpCode.replace(t, '').upper()
        #y = '' if len(t) == 4 else str((datetime.datetime.now()).year)[2:3]
        #return '%s%s%s' % (c, y, t)

    def parseCode(self, mCode=None):
        t = ''.join(re.findall(r'[0-9]', mCode))
        return mCode.replace(t, '').upper()


class interface_pyctp(object):
    indexList = ['IH', 'IC', 'IF']

    def __init__(self, use=True, baseInfo=None, userkey='simnow1'):
        self.isCtpUse = use
        self.ordersSend, self.ordersResponse = [], []
        self.baseInfo = baseInfo
        self.ctpUser = userkey
        self.ctpPath = public_basePath + 'com/ctp/ctp.json'
        #print(self.ctpPath)

        self.prePosDetailMap = {}
        self.Trade = None
        self.iniAmount = 250000
        self.ordersWaits = []
        self.banAlterList = ['IC', 'IH', 'IF']  # 手动替换清单
        if use:
            try:
                self.Trade = OrderTrader(self.ctpPath, self.ctpUser)
            except:
                logger.error(("CTP 连接错误！"))

    @property
    def front(self):
        return self.Trade.trader.front

    @property
    def session(self):
        return self.Trade.trader.session

    @property
    def Available(self):
        # 查询资金可用余额和总额
        res = self.qryAccount()
        return (res['Available'], res['Balance'])

    @property
    def posMap(self):
        """ 返回持仓明细"""
        m = {}
        for pos in self.qryPosition():
            if pos['Position'] > 0:
                c, c1 = pos['InstrumentID'].decode('gbk'), self.baseInfo.parseCode(pos['InstrumentID'].decode('gbk'))
                if c1 not in self.baseInfo.map: continue
                mode = '0' if pos['PosiDirection'].decode('utf-8') == '2' else '1'
                key, key1 = '%s_%s' % (c, mode), '%s_%s' % (c1, mode)

                if key in m:
                    p = m[key][0] + pos['Position']
                    y = m[key][1] + pos['YdPosition']
                    m[key] = m[key1] = (p, y, m[key][2])
                else:
                    m[key] = m[key1] = (
                        pos['Position'], pos['YdPosition'], self.baseInfo.att(c1, 'exchange'))  # vol,昨日，exchangeID
        return m

    def getPrePosDetailMap(self, combin=True):
        """ 以数据记录方式返回持仓明细 """
        res = self.qryPositionDetail()

        m, sub = {}, {}
        if res is None: return m
        for pos in res:
            if pos['Volume'] > 0:
                # 过滤非本进程节点
                c = self.baseInfo.parseCode(pos['InstrumentID'].decode('gbk'))
                # if c not in self.baseInfo.Instruments: continue

                # 转换为主力合约代码
                code, posi = c, pos['Direction'].decode('gbk')
                key, tradeId = c + '_' + posi, pos['TradeID'].decode('gbk').strip()
                sub = {
                    "key": key,
                    "symbol": pos['InstrumentID'].decode('gbk'),
                    "code": self.baseInfo.parseMCode(pos['InstrumentID'].decode('gbk')),
                    "mode": 1 if posi == '0' else -1,
                    "isopen": 1,
                    "hands": pos['Volume'],
                    "orderID": tradeId,
                    "price": pos['OpenPrice'],
                    "profit": pos["PositionProfitByTrade"],
                    "fee": 0,
                    "createdate": public.parseTime(pos['OpenDate'].decode('utf-8'), format='%Y%m%d',
                                                   style='%Y-%m-%d %H:%M:%S')
                }

                if not key in m:
                    m[key] = sub
                # 合并,并计算价格
                else:
                    h0, h1 = m[key]["hands"], sub["hands"]
                    p0, p1 = m[key]["price"], sub["price"]
                    m[key].update({
                        "hands": h0 + h1,
                        "price": (h0 * p0 + h1 * p1) / (h0 + h1),
                        "profit": m[key]["profit"] + sub["profit"]
                    })
        return m

    def close(self):
        self.Trade.logout()

    """发单"""

    def sendOrder(self, orderDocs, callback=None):
        if not self.Trade:
            self.Trade = OrderTrader(self.ctpPath, self.ctpUser)

        orders, self.ordersSend, self.ordersResponse, self.ordersWaits = [], [], [], []
        for doc in orderDocs:
            # 参数赋值
            sym = doc["symbol"]  # ctp 提交Code

            # 多空 买卖
            mode = PyCTP.THOST_FTDC_D_Buy if doc["mode"] > 0 else PyCTP.THOST_FTDC_D_Sell

            # 平开
            isopen = PyCTP.THOST_FTDC_OF_Open
            if doc["isopen"] == 0:
                isopen = PyCTP.THOST_FTDC_OF_CloseToday if 'istoday' in doc and doc[
                    "istoday"] == 1 else PyCTP.THOST_FTDC_OF_Close  #

            stype = PyCTP.THOST_FTDC_HF_Speculation
            price = 'AskPrice1' if doc["mode"] > 0 else 'BidPrice1'
            vol = int(doc["hands"])

            doc.update({
                "status": 0,
                "session": self.session,
                "front": self.front,
                "direction": mode
            })
            # 发送前记录集
            self.ordersSend.append(doc)

            order = Order(bytes(sym, encoding='gbk'), mode, isopen, stype, vol, price, timeout=3, repeat=3)

            orders.append(order)

        if callback:
            return self.Trade.insert_orders(orders, callback=callback)
        else:
            return self.Trade.insert_orders(orders, callback=self.orderResult)

    def orderResult(self, ps):

        # 交易单成功记录
        for p in ps:
            # 处理等待中的元素
            k = p.field['InstrumentID'].decode('gbk')
            if p.status in [5, 6]:
                p.field['status'] = p.status
                p.field['volume'] = p.traded_volume

                if k in self.ordersWaits: self.ordersWaits.remove(k)
                self.ordersResponse.append(p.field)

            elif p.status not in [4]:
                # 等待中的状态
                if k not in self.ordersWaits: self.ordersWaits.append(k)

            elif p.status in [4]:
                if k in self.ordersWaits: self.ordersWaits.remove(k)

    # 返回集检查
    def checkResult(self):
        c0 = ['direction', 'hands']
        c1 = ['Direction', 'VolumeTotalOriginal']
        t, w = 0, 0

        times = 5
        # 等待处理中的记录
        if self.ordersWaits is not None:
            while len(self.ordersWaits) > 0:
                if w > times: break
                logger.info(('-------- order waiting.....', self.ordersWaits, w))
                w += 1
                time.sleep(3)

        for d0 in self.ordersSend:
            if not (d0['session'] == self.session and d0['front'] == self.front): continue
            for d1 in self.ordersResponse:
                #print('------------return--------------', d1)

                if d0['symbol'] == d1['InstrumentID'].decode('gbk') and [d0[k] for k in c0] == [d1[k] for k in c1]:
                    d0.update({
                        "status": d1['status'],
                        "price": d1['LimitPrice'],
                        "hands": d1['volume'],
                        "orderID": d1["OrderRef"].decode('gbk').strip(),
                        "vol": d1['volume'] * self.baseInfo.att(d0['symbol'], 'contract_multiplier')
                    })
                    t += 1
                    break

        logger.info(
            ('--------orderresult compare input-out', len(self.ordersSend), len(self.ordersResponse), 'match:', t))

        # 输出： 未匹配订单数，订单返回信息
        return (len(self.ordersSend) - t) if len(self.ordersWaits) == 0 else 4, self.ordersSend

    def qryAccount(self, ):
        """查询账户资金"""
        res = self.Trade.trader.ReqQryTradingAccount()
        return res.result

    def qryPosition(self):
        """查询持仓"""
        time.sleep(1)
        res = self.Trade.trader.ReqQryInvestorPosition()
        return res.result

    def qryPositionDetail(self, code=''):
        """查询持仓明细"""
        time.sleep(1)
        res = self.Trade.trader.ReqQryInvestorPositionDetail(InstrumentID=code.encode('gbk'))
        return res.result

    def qtyInstrument(self, callback=None):
        """查询所有合约"""
        res = self.Trade.trader.ReqQryInstrument()
        return res.result

    def cancelOrder(self, cancelOrderReq):
        """撤单"""
        pass

    def printClass(self, obj):
        doc = {}
        for name in dir(obj):
            if name.find("__") > -1: continue
            value = getattr(obj, name)
            doc[name] = value
        return doc

    # 程序启动时自动检查初始状态
    def iniPosition(self, codes, docs=[]):
        """查询账户资金"""
        if self.prePosDetailMap == {}:
            self.prePosDetailMap = self.getPrePosDetailMap()

        pdm = self.prePosDetailMap
        if len(docs) > 0:
            k = [codes[i] + '_' + str(0 if docs[i]['mode'] == 1 else 1) for i in [0, 1]]
            if k[0] in pdm and k[1] in pdm:
                if docs[0]['hands'] <= pdm[k[0]]["hands"] and docs[1]['hands'] <= pdm[k[1]]["hands"]:
                    # 减去持仓量
                    for i in [0, 1]:
                        pdm[k[i]]["hands"] = pdm[k[i]]["hands"] - docs[i]['hands']
                    return docs
        else:
            # 无关联字段的匹配历史数据
            for j in [0, 1]:  # 买卖方向
                k = [codes[i] + '_' + str((j + i) if (j + i) < 2 else 0) for i in [0, 1]]
                if k[0] in pdm and k[1] in pdm:
                    hs = [pdm[k[i]]["hands"] for i in [0, 1]]
                    # 交易量是否为零
                    if hs[0] == 0 or hs[1] == 0:  continue

                    cm = [self.baseInfo.att(codes[i], 'contract_multiplier') * pdm[k[i]]['price'] for i in
                          [0, 1]]  # 1手金额
                    h = [0, 0]
                    # 根据1手金额， 计算可交易量
                    if cm[0] / cm[1] < hs[1] / hs[0] * 1.0:
                        h[0] = round(hs[0] / 2, 0) if hs[0] > 5 else hs[0]
                        h[1] = round(h[0] * cm[0] / cm[1], 0)
                    else:
                        h[1] = round(hs[1] / 2, 0) if hs[1] > 5 else hs[1]
                        h[0] = round(h[1] * cm[1] / cm[0], 0)

                    # 更新字段
                    ds = []
                    for i in [0, 1]:
                        d = copy.deepcopy(pdm[k[i]])
                        c = self.baseInfo.att(codes[i], 'contract_multiplier')  # 1手吨数
                        r = self.baseInfo.att(codes[i], 'ratio')  # 手续费率
                        d.update({
                            "hands": h[i],
                            "vol": h[i] * c,
                            "fee": (h[i] * r) if r > 0.5 else (d["price"] * h[i] * c * r)
                        })
                        pdm[k[i]]["hands"] = pdm[k[i]]["hands"] - h[i]
                        ds.append(d)
                    return ds
        return None

    preMap = None

    def checkPosition(self, docs, isOpen, reverse=1, refresh=True):
        """ 提交前检查持仓和资金状况 """
        if isOpen == 1:
            # 开仓检查资金余额
            minRate = float(config_ini.get("minMarginRate", "CTP"))  # 保证金准备金比例
            fund = self.Available  # 可用保证金和所有保证金
            needs = sum([d['vol'] * self.baseInfo.att(d['code'], 'margin_rate') for d in docs])
            return 0 if (fund[0] > needs and ((fund[0] - needs) / fund[1] > minRate)) else -1

        else:
            # 平仓检查仓位余额
            if not refresh and self.preMap is not None:
                posMap = self.preMap
            else:
                self.preMap = posMap = self.posMap

            for doc in docs:
                # if doc['name'] =='IH': print(posMap.keys())
                if 'name' in doc:
                    key = '%s_%s' % (doc['name'], ('0' if (doc['mode'] * reverse < 0) else '1'))  #
                else:
                    key = '%s_%s' % (doc['symbol'], ('0' if (doc['mode'] * reverse < 0) else '1'))  #

                if not key in posMap or posMap[key][0] < doc['hands']:
                    return -2  #

                elif len(posMap[key]) > 2 and posMap[key][1] < doc['hands'] and posMap[key][2] == 'SHFE':
                    """区分是否时上期所平今仓, 仓位小于Yd持仓 """
                    doc.update({'istoday': 1})
            return 0

    def alterPosi(self, openDocs):
        """自动处理调仓, 调仓结束后，子进程启动时自动检查交易记录，生成新的买单"""
        self.baseInfo = BaseInfo([])
        map = self.posMap

        if map == {}: return None
        orders = []

        # 检查主力合约是否跨期，并自动调仓
        monthWeek, weekDay = public.getMonthDay()

        for code in openDocs:
            # 手动调仓清单
            if code in self.banAlterList: continue

            doc = openDocs[code][0]
            mCode = self.baseInfo.mCode(code)
            key = '%s_%s' % (code, '0' if doc['mode'] > 0 else '1')
            print('alterPosi check:', key, mCode, doc['code'])

            opt0 = mCode != doc['code']
            opt1 = (doc['name'] in self.indexList and monthWeek == 3 and weekDay == 3) # 股指期货调仓

            if (opt0 or opt1) and key in map and map[key][0] >= doc['hands']:
                logger.info(('alterPosi start: c1,c0, v, v0', key, mCode, doc['code'], map[key][0], doc['hands']))
                # 卖出单
                del doc['id']
                d0 = copy.deepcopy(doc)
                d0['mode'] = -d0['mode']
                d0['isopen'] = 0
                d0['isstop'] = 4
                d0['createdate'] = public.getDatetime()
                orders.append(d0)

                # 按原状态买入
                d1 = copy.deepcopy(doc)
                d1['code'] = mCode
                d1['symbol'] = self.baseInfo.parseCtpCode(mCode)
                d1['batchid'] = uuid.uuid1()
                d1['status'] = d0['status'] = 4
                d1['createdate'] = doc['createdate']
                d1['isstop'] = 0
                orders.append(d1)

        # 下单数
        if len(orders) > 0:
            self.sendOrder(orders)
            # return 0, orders

        return self.checkResult()

    # 配对交易不成功的强制平仓
    def forceOrder(self, docs):
        # 先检查初始状态
        orders = []
        for doc in docs:
            if doc['status'] == 6:
                newDoc = copy.deepcopy(doc)
                newDoc['mode'] = - doc['mode']
                newDoc['isopen'] = 0
                newDoc['istoday'] = 1
                orders.append(newDoc)
            else:
                # 状态设为-9
                doc['status'] = -9

        if len(orders) > 0:
            self.sendOrder(orders)
            return self.checkResult()

    # 处理僵尸持仓
    def orderDead(self, full_codes):
        """ 检查position中的僵尸币对,"""
        map = self.prePosDetailMap
        orders, ks = [], []
        for k in map:
            if k.split("_")[0] in full_codes and map[k]['hands'] > 0:
                doc = copy.deepcopy(map[k])
                doc['mode'] = - map[k]['mode']
                doc['isopen'] = 0
                doc['istoday'] = 0
                doc['method'] = 'dead'
                orders.append(doc)
                ks.append((k, map[k]['hands']))
                # break

        if len(orders) > 0:
            r = self.checkPosition(orders, 0)
            if r == 0:
                print('dead position:', ks)
                # 执行反向操作，清理已有记录
                self.sendOrder(orders)
                return self.checkResult()

        return -1, None


columns_account = ['PreBalance', 'Balance', 'CloseProfit', 'PositionProfit', 'Commission', 'CurrMargin', 'Available']

columns_posiDetail = ['InstrumentID', 'Direction', 'OpenPrice', 'Volume', 'CloseProfitByTrade', 'PositionProfitByTrade',
                      'ExchangeID', 'OpenDate']
columns_posi = ['InstrumentID', 'PosiDirection', 'Position', 'YdPosition', 'CloseProfit', 'PositionProfit']


def main():
    b = BaseInfo([])
    obj = interface_pyctp(baseInfo=b, userkey='zhao')

    print('终端信息:', [c for c in PyCTP.CTP_GetSystemInfo()])

    res = obj.qryAccount()
    for k in columns_account:
       print((k, res[k]))

    print()
    res = obj.qryPosition()
    res.sort(key=lambda k: (k.get('InstrumentID', 0).decode('gbk')))
    print(len(res), columns_posi)
    for r in res:
        # if r['Position'] > 0:
        print([r[c] for c in columns_posi])


if __name__ == '__main__':
    main()
