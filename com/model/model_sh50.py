# -*- coding: utf-8 -*-

""" 
Created on  2018-05-03 
@author: lbrein
      
        sh50 期权价格记录和监控

"""

from com.base.public import public, config_ini, logger
from com.object.obj_entity import sh50_price, sh510050, sh50_spy
from com.data.data_interface import sinaInterface, s_webSocket
import math
import pandas as pd


class model_sh50:
    def __init__(self):
        # 新浪接口
        self.Int = sinaInterface()
        # sqlserver 期权每3秒行情记录
        self.sPrice = sh50_price()
        # sqlserver sh510050 每3秒行情记录
        self.Owner = sh510050()
        # sqlserver 监测结果表
        self.Spy = sh50_spy()

        self.price_owner = 0
        # 资金成本
        self.interest = float(config_ini.get("interest", session="SH50")) / 365
        # 带融券的资金成本
        self.interest_rong = float(config_ini.get("interest_rong", session="SH50")) / 365
        # 买入阈值
        self.buy_line_0 = float(config_ini.get("buy_line_0", session="SH50"))
        self.buy_line_1 = float(config_ini.get("buy_line_1", session="SH50"))

        self.table_head = ""
        self.table_end = ""
        self.method = 'ma60'
        self.ws = s_webSocket("ws://%s:8090/etf50option" % config_ini.get("ws.host", session="db"))
        self.keylist2 = ["result", "price_ask", "price_bid"]
        self.titlesList = self.Spy.keylist[:-2] + ["result_1", "price_ask", "price_bid"]

    def record(self):
        # 510050 行情
        stock = self.Int.get510050()
        stock["datetime"] = public.getDatetime()
        self.Owner.insert(stock)

        # 期权行情
        etfs = self.Int.get_price()
        spys = []

        self.price_owner = float(stock["price"])

        for i in range(0, len(etfs), 2):
            # 计算剩余天数
            r = public.dayDiff(self.Int.expireMap[etfs[i]["code"]], public.getDate())

            rec0 = self.calc(etfs[i], etfs[i + 1], r, mode=0)
            spys.append(rec0)

            rec1 = self.calc(etfs[i + 1], etfs[i], r, mode=1)
            spys.append(rec1)


        # inform 通知
        try:
            self.inform(spys)
        except:
            logger.error("ws disconnect")

        # 写入数据库
        self.sPrice.insertAll(etfs)

        self.Spy.insertAll(spys)

    # 监听仅通知 - 手动操作
    def inform(self, spys):
        if self.table_head == "":
            self.table_head = "<table width=96% align=center border=1 height=26><tr bgcolor='#e3e3e3'>" + "".join(
                ["<th>%s</th>" % c for c in self.titlesList]) + "</tr>"
            self.table_end = "</table>"
        tmp = ""

        # ups = [doc for doc in spys if (doc["mode"] == 0 and doc["result"] < 0.999 and doc['price_ask'] != 0)]

        ups = [doc for doc in spys if (doc["mode"] == 0 and doc['price_ask'] != 0 and doc['price_bid'] != 0)]
        downs = [doc for doc in spys if (doc["mode"] == 1)]
        downs_map = {}

        for doc2 in downs:
            downs_map[doc2["code"]] = doc2

        # paixu
        ups.sort(key=lambda x: x["result"])

        for doc in ups:
            bg = "" if doc["result"] > self.buy_line_0 else " bgcolor='green' "
            doc2 = None
            if doc["code_1"] in downs_map.keys():
                doc2 = downs_map[doc["code_1"]]

            tmp += self.parseHtml(doc, bg, doc2=doc2)

        html = self.table_head + tmp + self.table_end
        # print(html)
        self.ws.send(html)

    def parseHtml(self, doc, bg, doc2=None):
        # -2
        html = "<tr %s>" % bg + "".join(
            ["<td align=center>%s</td>" % doc[c] for c in self.Spy.keylist[:-2]])

        if doc2:
            html += "".join(["<td align=center>%s</td>" % (doc2[c]) for c in self.keylist2])
        else:
            html += "".join(["<td align=center>%s</td>" % i for i in range(len(self.keylist2))])

        return html + "</tr>"

    def calc(self, doc1, doc2, remainDays, mode=0):
        rec = {
            'code': doc1["code"],
            'code_1': doc2["code"],
            'price_bid': doc1["bid"],
            'price_ask': doc2["ask"],
            'price_owner': self.price_owner,
            'remainDay': int(remainDays),
            'power': doc1["power"],
            'mode': mode,
            'currentTime': public.getDatetime()
        }

        if mode == 0:
            if doc2["ask"] == 0 or doc1["bid"] == 0:
                res = 1.0
            else:
                res = (float(self.price_owner) + float(doc2["ask"]) - float(doc1["bid"])) / float(
                    doc1["power"]) / math.exp(
                    -1 * self.interest * remainDays)

                res1 = 0
        else:
            if doc2["ask"] == 0 or doc1["bid"] == 0:
                res = 1.0
            else:
                res = (float(self.price_owner) - float(doc2["ask"]) + float(doc1["bid"])) / float(
                    doc1["power"]) / math.exp(
                    -1 * self.interest * remainDays)

                res1 = (float(self.price_owner) - float(doc2["ask"]) + float(doc1["bid"])) / float(
                    doc1["power"]) / math.exp(
                    self.interest_rong * remainDays)

        rec["result"], rec["result1"] = res, res1
        return rec

    def order(self):
        # 下单
        pass

from com.object.obj_entity import sh50_daily_1, sh50_extend_1
from scipy.stats import pearsonr


class sh50_data_analysis(object):

    def __init__(self):
        self.Sh50 = sh50_daily_1()
        self.Extend = sh50_extend_1()
        self.columns = ['code', 'rel_expireDay_disstance', 'rel_ownerPrice_disstance', 'rel_bsPrice_disstance']

    def start(self):
        docs = []
        for code in self.Extend.getCodes():
            df = self.Sh50.getData(code=code)
            doc = {
                "code": code,
                # 剩余时间与diss 关联性
                'rel_expireDay_disstance': self.per(df["expireDay"], df["calc_diss"]),
                #
                'rel_ownerPrice_disstance': self.per(df["owner_price"], df["calc_diss"]),
                #
                'rel_bsPrice_disstance': self.per(df["bs_price"], df["calc_diss"])
            }
            print(code)
            docs.append(doc)

        df1 = pd.DataFrame(docs, columns=self.columns)
        df1.to_csv("e:/stock/sh50_pearsonr.csv")

    # 皮尔森相似度计算
    def per(self, df0, df1):
        d1 = df1.tolist()
        if df0.count() < df1.count():
            d1 = d1[-df0.count():]
        return pearsonr(df0.tolist(), d1)[0]

    # 线性回归
    def line(self):
        pass

    def stat(self):
        pass


# 分类报表及分析
class sh50_report(sh50_daily_1):
    sql_report = """
        select  %s, 
                avg(a.calc_disstance) as ave_diss ,
        STDEV(a.calc_disstance) as std_diss, 
        count(*) as count 
        
        from sh50_daily_1 a 
        LEFT join sh50_extend_1 b on a.code = b.code 
        where b.exe_mode = 1 
        group by  %s     
        ORDER BY  %s;
    """
    report_Map = {
        # 按到期天数统计
        "expireDay": [("expireDay", "a.expireDay")],

        # 按年-月，到期天数统计
        "exp_month": [
            ("year", "datepart(year, a.date)"),
            ("month", "datepart(month, a.date)"),
            ("expireDay", "a.expireDay")
        ],
        "exp_execPrice" : [
            ("diss", ""),
            ("expireDay", "a.expireDay")
        ]
    }

    subColumns = ["ave_diss", "std_diss", "count"]

    def start(self, report='exp_month'):
        columns, exp, gd = self.parseItem(report)
        self.keylist = columns + self.subColumns
        sql = self.sql_report % (exp, gd, gd)
        print(sql)
        df = self.df(sql)
        print(df)

    def parseItem(self, report):
        columns, exp, gd = [], "", ""
        for item in self.report_Map[report]:
            exp += " %s as %s, " % (item[1], item[0])
            gd += item[1] + ","
            columns.append(item[0])
        return columns, exp[:-2], gd[:-1]

def main():
    actionMap = {
        "price": 1,  #
        "inform": 0,
        "analy": 0,
        "report": 0
    }

    obj = model_sh50()
    if actionMap["price"] == 1:
        obj.record()

    if actionMap["inform"] == 1:
        obj.inform()

    if actionMap["analy"] == 1:
        obj1 = sh50_data_analysis()
        obj1.start()

    if actionMap["report"] == 1:
        obj1 = sh50_report()
        obj1.start()


if __name__ == '__main__':
    main()
