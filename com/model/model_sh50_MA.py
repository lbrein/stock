# -*- coding: utf-8 -*-

""" 
Created on  2018-05-03 
@author: lbrein
      
        sh50 期权价格记录和监控

"""

from com.base.public import public, config_ini, logger
import numpy as np
import talib as ta
from com.data.interface_Rice import interface_Rice
from com.object.obj_entity import sh50_orderForm
from com.data.data_interface import sinaInterface, s_webSocket
import copy
import traceback
import time
from com.object.mon_entity import mon_tick
import uuid


class model_sh50_ma:
    def __init__(self):
        # 新浪接口
        self.Int = sinaInterface()
        self.Record = sh50_orderForm()
        self.Rice = interface_Rice()
        self.Rice.iniTimeArea = [('09:30:00', "11:30:00"), ('13:00:00', '15:00:00')]
        self.Rice.setTimeArea('15:00:00')
        self.Tick = mon_tick()
        self.isParamSave = False

        self.codes = ['510050.XSHG']
        self.ktypeMap = {'510050.XSHG': 30}

        self.period = '30m'
        self.timePeriodList = [5, 10, 20, 60]
        self.timePeriods = 60
        self.iniAmount = 40000
        self.multiplier = 10000
        self.currentVol = 0
        self.currentPositionType = 0
        self.ratio = 2
        self.status = 0

        self.ws = s_webSocket("ws://%s:8090/etf50optiontrend" % config_ini.get("ws.host", session="db"))
        self.titleList = ['createdate', 'code', 'name', 'isBuy', 'mode', 'vol', 'price', 'ownerPrice']
        self.table_head = ""
        self.table_end = ""
        self.method = 'ma60'
        self.viewItemList = ['createTime', 'close', 'ma5', 'ma10', 'ma20', 'ma60', 'vol', 'mode']

    def start(self):
        self.uid = "510050_XSHG_%s" % self.period
        self.tickParam = None
        self.iniNode()

        logger.info(('etf50 - ma stage start', self.uid))
        self.Rice.startTick(self.codes, callback=self.onTick, source='kline', kPeriod=self.period, kmap=self.ktypeMap)
        # 初始化节点

    def iniNode(self):
        doc = self.Record.getPosition(method=self.method)
        if doc is not None:
            self.currentVol, self.currentPositionType, self.preNode = doc['vol'], doc['pos'], doc
            if self.method == 'bull' and abs(doc['mode']) in [3, 4]:
                self.status = np.sign(doc['pos'])
        else:
            self.preNode = None
            self.status = 0
        logger.info(('current status: vol - pos - status:', self.currentVol, self.currentPositionType, self.status))

    def onTick(self, tick):

        # 按时长读取k线数据
        kline = int(self.period[:-1])
        dfs = self.Rice.getKline(self.codes, ktype=kline, key=kline)

        # 检查间隔时间
        df0 = dfs[self.codes[0]]
        t0 = tick[self.codes[0]]

        apart = self.timeApart = (int(time.time()) % (60 * kline)) * 1.0 / (60 * kline)
        param = self.paramCalc(df0, t0)

        if param is not None and not np.isnan(param["close"]):
            # 执行策略，并下单
            # self.orderCheck(t0, param)

            if self.isParamSave:  self.debugT('', n=1, param=param)

            try:
                self.orderCheck(t0, param, apart=apart)
                pass
                # 计算指标
            except Exception as e:
                print(traceback.format_exc())

    def point(self, row):
        if np.isnan(row['ma60']): return 0

        close, vol, atr = row['close'], row['vol'], row['atr']
        p, j = 0, 1
        lists = self.timePeriodList

        # 上行标志
        for t in lists:
            ma, mam, mam2, mam3 = row['ma' + str(t)], row['mam' + str(t)], row['mam%s_2' % str(t)], row[
                'mam%s_3' % str(t)]
            bias = 2 * np.sign(row['bias' + str(t)]) if t == 60 else row['bias' + str(t)]
            if (abs(mam2) == 2 and abs(mam3) == 1) or (mam2 == 0 and (vol > 2.0 or atr > 0.005)):
                if abs(bias) > 0.35:
                    p = int(j * np.sign(bias))

            j += 1
        return p

    def mam(self, row, p):
        if np.isnan(row['ma' + str(p)]): return 0
        return int(np.sign(row['close'] - row['ma' + str(p)]))

    def apart(self, PS, ktype):
        apart = (int(time.time()) % (60 * ktype)) * 1.0 / (60 * ktype)
        return PS * apart + PS.shift(1) * (1 - apart)

    def paramCalc(self, df0, t0):
        # 替换即时k线
        df = copy.deepcopy(df0[:-1])
        # 计算相关参数
        df.loc[public.getDatetime()] = t0
        df = self.add_stock_index(df)
        df['createTime'] = df.index
        df['mode'] = df.apply(lambda row: self.point(row), axis=1)
        param = df.iloc[-1].to_dict()
        return param

    def add_stock_index(self, df, index_list=None):
        close = df["close"]
        for p in self.timePeriodList:
            df["ma" + str(p)] = ma = ta.MA(close, timeperiod=p)
            df["mam" + str(p)] = mam = df.apply(lambda row: self.mam(row, p), axis=1)
            df["mam" + str(p) + "_2"] = ta.SUM(mam, timeperiod=2)
            df["mam" + str(p) + "_3"] = ta.SUM(mam, timeperiod=3)
            df["bias" + str(p)] = (close - ma) / ma * 100

        df['vol'] = df['volume'] / ta.MA(df['volume'], timeperiod=60)
        df['atr'] = ta.ATR(df['high'], df['low'], close, timeperiod=1) / close.shift(1)
        return df

    itemCount = 0
    cvsMap = {}

    def debugT(self, str, n=1000, param=None):
        self.itemCount += 1
        if str != '' and self.itemCount % n == 0:
            self.inform()
            print(self.itemCount, str)

        if param is not None:
            # 初始化
            param['code'] = code = self.codes[0]
            for key in ['kdjm', 'sarm']:
                if key in param: param[key] = int(param[key])

            if code not in self.cvsMap:
                self.cvsMap[code] = [param]
                self.cvsMap['c_' + code] = 1
            else:
                self.cvsMap['c_' + code] += 1
                c, t = self.cvsMap[code], self.cvsMap['c_' + code]
                if len(c) > 2:
                    # 保存到mongodb
                    self.Tick.col.insert_many(self.cvsMap[code])
                    self.cvsMap[code] = []

                elif self.cvsMap['c_' + code] % n == 0:
                    self.cvsMap[code].append(param)


    def orderCheck(self, t0, param, apart=0):
        self.tickParam = param

        period, ini, unit = self.timePeriods, self.iniAmount / self.multiplier, self.multiplier
        mode, close = (param[key] for key in "mode,close".split(","))

        vol, pos = self.currentVol, self.currentPositionType
        isBuy, isRun = 0, False

        if vol > 0 and pos * mode < 0 and abs(mode) == 4:
            # 先平仓，再开仓
            self.order(param, -1, mode, vol)
            # 再开仓
            isBuy, pos, isRun, vol = 1, mode, True, ini

        elif vol > 0 and pos * mode < 0 and abs(mode) < 4:
            # 部分减仓
            if vol >= (5 - abs(mode)):
                isBuy, isRun, mode, vol = -1, True, mode, abs(mode) + vol - ini

        # 部分加仓
        elif ini > vol > 0 and pos * mode > 0 and abs(mode) < 4:
            # 部分减仓
            if vol < ini:
                isBuy, isRun, mode, vol = 1, True, mode, abs(mode) if (ini - vol) > abs(mode) else ini - vol

        elif vol == 0 and abs(mode) == 4:
            isBuy, pos, isRun, vol = 1, mode, True, ini


        self.debugT((param, isBuy, mode, vol), n=1)

        if apart >0 and apart < 0.833: return
        if isRun:
            self.order(param, isBuy, mode, vol)

    def order(self, n0, isBuy, mode, vol):
        pN = self.preNode
        cv, uid, unit = self.currentVol, self.uid, self.multiplier

        ETF = None
        # 查询匹配的期权当前价格
        if isBuy < 0:
            # 卖出
            ETF = self.Int.getETF(code=pN['code'])
        else:
            # 新开仓
            type = 1
            if mode == 0: mode = 1
            ETF = self.Int.getETF(sign=np.sign(mode), price=n0['close'])

        if ETF is None or len(ETF) == 0:
            # print(pN['code'] if type == 0 else 'new', mode, now)
            return False

        price = ETF["ask"].values[0] if isBuy > 0 else ETF["bid"].values[0]
        fee = vol * self.ratio

        # 修改状态
        self.currentPositionType = np.sign(mode) if isBuy > 0 else pN['pos']
        self.currentVol += isBuy * vol

        doc = {
            "code": ETF['code'].values[0],
            "name": ETF['name'].values[0],
            "createdate": public.getDatetime(),
            "price": price,
            "vol": vol,
            "mode": mode,
            "isBuy": isBuy,
            "pos": self.currentPositionType,
            "ownerPrice": n0['close'],
            "fee": fee,
            "amount": -isBuy * price * vol * unit - fee,
            "uid": uid,
            "batchid": uuid.uuid1() if (isBuy > 0 or self.preNode is None) else self.preNode['batchid'],
            "method": self.method
        }

        self.Record.insert(doc)
        # 设置上一个记录
        self.preNode = doc if self.currentVol > 0 else None

        # 交易提示1分钟
        s = 0
        while 1:
            self.inform(order=doc)
            if s > 5: break
            s += 1
            time.sleep(15)

        return True

    # 交易提示
    def inform(self, order=None):
        tradeHTML, tickHTML, recordHTML = "", "", ""
        # tick html
        if self.tickParam is not None:
            viewList = self.viewItemList
            tickHTML = "<center><font color=red font-size=14> 当前510050行情</font></center><br/><table width=96% align=center border=1 height=26>"
            tickHTML += "<tr bgcolor='#e3e3e3'>" + "".join(
                ["<th align=center><b>%s</b></th>" % k for k in viewList]) + "</tr>"
            tickHTML += "<tr>" + "".join(["<td align=center>%s</td>" % self.tickParam[k] for k in viewList]) + "</tr>"
            tickHTML += "</table><br/><br/>"

        if order is not None:
            viewList = ['code', 'name', 'isBuy', 'vol', 'price', 'ownerPrice']
            tradeHTML = "<center><font color=red font-size=14> 交易提示</font></center><br/><table width=96% align=center border=0 height=26>"
            tradeHTML += "".join(["<tr><td width=120><b>%s</b></td><td>%s</td></tr>" % (k, order[k]) for k in viewList])
            tradeHTML += "</table><br/><br/><script>alert('交易提示请尽快处理!!!')</script>"

        if self.table_head == "":
            self.table_head = "<center><font color=red font-size=14> 近期交易清单</font></center><br/>" \
                              "<table width=96% align=center border=1 height=26><tr bgcolor='#e3e3e3'>" + "".join(
                ["<th>%s</th>" % c for c in self.titleList]) + "</tr>"
            self.table_end = "</table>"

        tmp = ""
        for doc in self.Record.getRecords(method=self.method):
            tmp += self.parseHtml(doc, '')

        recordHTML = self.table_head + tmp + self.table_end

        html = "<html><body>" + tradeHTML + tickHTML + recordHTML + "</body></html>"
        self.ws.send(html)

    def parseHtml(self, doc, bg, doc2=None):

        html = "<tr %s>" % bg + "".join(
            ["<td align=center>%s</td>" % doc[c] for c in self.titleList])

        return html + "</tr>"


def main():
    actionMap = {
        "start": 1,  #
        "inform": 0,
    }

    obj = model_sh50_ma()
    if actionMap["start"] == 1:
        obj.start()

    if actionMap["inform"] == 1:
        obj.inform()


if __name__ == '__main__':
    main()
