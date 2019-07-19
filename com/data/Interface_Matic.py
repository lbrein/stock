# -*- coding: utf-8 -*-

""" 
Created on  2018-05-05 
@author: lbrein
       从文件导入历史数据到sqlserver,包括：
       1、股票code清单 （futu)
       2、交易所软件导出的历史数据
"""

from com.object.obj_entity import maticIndex
from com.base.public import csvFile, config_ini, public, logger
import pandas as pd
import datetime
import time

class Matic_interface:
    # 报单配置
    matic_csv_map = {
        # 股票报单
        'stock_order': {
            'filename': 'order_%s.%s.csv',
            'columns':
                [
                    ('local_entrust_no', '', ''),  #
                    ('fund_account', 'account', ''),
                    ('exchange_type', 'market', ''),
                    ('stock_code', 'code', ''),
                    ('entrust_bs', 'type', ''),
                    ('entrust_prop', '', '0'),
                    ('entrust_price', 'price', ''),
                    ('entrust_amount', 'volume', ''),
                    ('batch_no', '', ''),
                    ('client_filed1', '', ''),
                    ('clientfield2', '', '')
                ]
        },
        # 撤单
        'stock_cancel': {
            'filename': 'cancel_%s.%s.csv',
            'columns': ['local_withdraw_no', 'fund_account', 'batch_flag', 'entrust_no', 'client_filed1',
                        'clientfield2']
        }
    }

    # 市场类别 
    market_type = {'SH': '1',
                   'SZ': '2',
                   'SHK': 'G',  # G-沪港通
                   'SZK': 'S'  # S-深港通
                   }

    actionList = ["orders", "results", "records"]
    recordList = ["DEAL", "ENTRUST", "FUND", "POSITION"]

    param_map = {
        "stage": 'ETF',
        "market": 'SH',
    }

    def __init__(self):
        self.maticAccount = self.iniKey("account")
        self.maticIndex = None

    def iniKey(self, key, db='Matic'):
        return config_ini.get(key, session=db)

    def _link_maticIndex(self):
        if self.maticIndex is None:
            self.maticIndex = maticIndex()

    # 下单
    def order(self, docs, type='stock_order', param=None):
        if docs is None: exit()
        # 策略名称
        stage = self.param_map["stage"]
        if param is not None:
            stage = param["stage"]

        maps = self.matic_csv_map[type]

        # 文件夹对应文件
        filename = maps["filename"] % (stage, self.getTime())
        CSV = csvFile(folder=self.iniKey("orders"),
                      filename=filename,
                      headers=[n[0] for n in maps["columns"]])
        
        # 生成csv文件
        No = self.getIndex() # 日期内唯一序号
        for doc in docs:
            rec = self.parseDoc(doc, "stock_order", No)
            CSV.writeLine(rec)
            No += 1 
        
        # 执行完回写index    
        self.setIndex(No)    
        CSV.close()

    def parseDoc(self, doc, type, No):
        maps = self.matic_csv_map[type]
        rec = {}
        for item in maps["columns"]:
            if item[1] in ['price','volume']:
                rec[item[0]] = doc[item[1]]
            else:
                # 默认值
                rec[item[0]] = item[2]
                
        # 特殊处理
        code = doc["code"].split(".")
        
        rec.update({
            "local_entrust_no": No,
            "fund_account": self.maticAccount, 
            "exchange_type": self.market_type[code[0]], 
            "stock_code": code[1],
            "entrust_bs": 1 if doc["type"]==1 else 2  
        })
        return rec
   
    # 撤单
    def cancel(self, docs, param=None):
        self.order(docs, type='stock_cancel', param=param)

    # 检查结果
    def result(self):
        pass

    # 查询matic文件, 获取账户余额和股票明细
    def getFund(self, account=None):
        # 查询资金余额
        df_0 = self.getCsv(account=account)

        # 查询账户明细
        df_1 = self.getCsv(account=account, record_type="POSITION")

        if df_0 is None: return None

        df = df_1.loc[:, ["stock_code", "current_amount", "enable_amount"]]
        # df.loc[df["stock_code"].count()] = ["free", df_0["enable_balance"][0],""]
        df.loc[df["stock_code"].count()] = ["free", 1000000, 1000000]
        return df

    def getTime(self):
        return str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))[:-3]

    def getDate(self):
        return datetime.datetime.now().strftime("%Y%m%d")

    def getTime1(self, m=0):
        return (datetime.datetime.now() + datetime.timedelta(days=0, hours=0, minutes=m)).strftime("%H%M00")

    # 读取csv结果文件
    def getCsv(self, action="records", account=None, record_type="FUND"):
        # 限定操作类型
        if not action in self.actionList:
            return None

            # 检查并默认账户
        if account is None:
            account = self.maticAccount

        if action == "records":
            if not record_type in self.recordList:
                return None

            folder = self.iniKey(action)
            filename = "%s_%s.%s.csv" % (account, record_type, str(self.getDate()))

            return pd.read_csv(folder + "\\" + filename, encoding='gb2312')

        elif action == "results":
            pass

    # 将交易记录detail保存到数据库
    def to_sql(self, full=False):

        pass

    def parseTwap(self, docs, type=-1):
        lst, vlst, values = "", "", ""
        for doc in docs:
            lst += doc["code"].replace(".", "") + "/"
            vlst += str(doc["volume"]) + "/"
            values += "%s:  bais-%s  price-%s volume-%s <br>" % \
                      (doc["code"], str(doc["value"]), str(doc["price"]), str(doc["volume"]))

        data = [
            ("买卖方向", 0 if type == 1 else 1),
            ("证券列表", lst),
            ("证券数量列表", vlst),
            ("最小委托单位", 1000),
            ("撤单时间", 12),
            ("开始时间", self.getTime1()),
            ("结束时间", self.getTime1(m=10)),
            ("委托档位", 2),
            ("委托属性", 0),
            ("备注", ""),
            ("暂停时允许撤单", 1)
        ]

        s = ""
        for item in data:
            s += "%s=%s<br>" % (item[0], item[1])

        return s + "<br>" + values

    def Twap(self, docs):
        types = [1, -1]  # 分为买卖2类进行下单操作
        results = ""
        for t in types:
            results += self.parseTwap([doc for doc in docs if doc["type"] == t], t) + "<br><br><br>"

        return results + " ---------------- " + public.getDatetime() + "----------------"

    def getIndex(self):
        self._link_maticIndex()
        return self.maticIndex.getIndex()

    def setIndex(self, index):
        self._link_maticIndex()
        self.maticIndex.setIndex(index)


# 华泰下单拆单
class seprate_order:

    # 检查资金账户，按value值
    Funds = None

    def order(self, docs, price=None):
        if self.Funds is None:
            self.Funds = self.Matic.getFund()

        # 查询记录，获取总量
        df = self.Funds

        # 可用余额和已存在股票清单
        free = df[df["stock_code"] == "free"]["enable_amount"].values[0]
        exists = df["stock_code"].tolist()

        # 每份额度 ，当前可买份数和实际份额
        eachAmount = int(self.iniKey("eachAmount"))
        pages = int(round(free / eachAmount, 0))
        realAmount = free / pages

        recs = []
        # 先卖后买
        for doc in docs:
            # 卖出
            if doc["type"] == -1:
                if not doc["code"] in exists: continue
                # 检查是否允许卖出
                a = df[df["stock_code"] == doc["code"]]["enable_amount"].values[0]
                if a > 0:
                    doc["volume"] = a
                    pages += 1  # 可买份数加1
                    recs.append(doc)


        # 买de1, 按values倒序
        docs.reverse()
        for doc in docs:
            if doc["type"] == 1:
                # 买入基金，份额低于0则退出
                if doc["code"] not in exists:
                    pages = pages - 1
                    doc["amount"] = realAmount
                    recs.append(doc)
                    if pages == 0:
                        break
        return recs


    # 按orderbook 进行拆单交易
    def Vwap(self, docs):
        v_inter, v_rate = int(self.iniKey('vwap_interval')), float(self.iniKey('vwap_volume_rate'))
        No = 0
        # 根据深度数据，自动批量下单
        while True:
            # 总共需要卖出的手数 用于卖出
            total_v = sum([doc["volume"] for doc in docs if doc["type"] == -1])

            # 总共需要买入金额 用于买入
            total_a = sum([doc["amount"] for doc in docs if doc["type"] == 1])

            if total_v <= 0 and total_a <= 0:
                break
            cur_docs, docs = self.vwap_volume(docs, v_rate)
            # 下单
            self.Matic.order(cur_docs)
            No += 1
            logger.info("批量下单批次号: %s" % No)
            # 下单间隔时间
            time.sleep(5)


    # 查询当前时间点的level,并选择1/2档的一半进行下单操作

    def vwap_volume(self, docs, v_rate):
        rec, recs = {}, []
        # 深度层次
        dep_length = int(self.iniKey("vwap_depth_length"))

        # 查询深度数据，并计算每次交易的量
        for doc in docs:
            rec = {}
            rec.update(doc)

            # 获得深度数据
            book = self.FT.get_order_book(doc["code"])
            if doc["type"] == -1:
                # 卖出的量
                rec = self.v_depth(rec, book['Ask'][:dep_length], v_rate, doc["volume"])
                # 递减初始交易量
                doc["volume"] = doc["volume"] - rec["volume"]
            else:
                # 买入的量价
                rec = self.v_depth(rec, book['Bid'][:dep_length], v_rate, doc["amount"])
                # 递减初始交易量
                doc["amount"] = doc["amount"] - rec["amount"]

            if rec["amount"] > 0:
                recs.append(rec)

        return recs, docs

    # 深度数据计算
    def v_depth(self, rec, dep, v_rate, cur_amount):
        if len(dep) == 0:
            return rec

        p = sum([n[0] for n in dep]) / len(dep)
        v = sum([n[1] * v_rate for n in dep])
        a = sum([n[0] * n[1] * v_rate * 1.0 for n in dep])

        # 卖出通过量计算a
        if rec["type"] == -1:
            v = v if cur_amount > v else cur_amount
            a = v * p

            # 买入通过金额进行手数
        else:
            a = a if cur_amount > a else cur_amount
            v = int(round(a / p / 100, 0) * 100)

        rec.update({
            "price": p,
            "volume": v,
            "amount": a,
        })
        return rec

        # 检查最新报价， inform提示价格和交易量

    def priceCheck(self, docs):
        # 更新price 和 volume
        df_prices = self.FT.get_cur_quote([doc["code"] for doc in docs])
        for doc in docs:
            p, v = doc["price"], doc["volume"]
            p1 = df_prices.loc[(df_prices["code"] == doc["code"]), "last_price"].values[0]
            doc["price"] = p1
            doc["volume"] = int(round(p * v / p1 / 100, 0) * 100)

        return docs

    def resultCheck(self):
        pass


def main():
    actionMap = {
        "order": 0,
        "cancel": 0,
        "result": 0,
        "records": 1
    }

    obj = Matic_interface()
    if actionMap["order"] == 1:
        obj.order([])

    if actionMap["cancel"] == 1:
        obj.cancel([])

    if actionMap["result"] == 1:
        obj.result()

    if actionMap["records"] == 1:
        res = obj.getFund()


if __name__ == '__main__':
    main()
