# -*- coding: utf-8 -*-

""" 
Created on  2018-01-04 
@author: lbrein
       ---  通用接口 ---      
- 
"""
import requests
from websocket import create_connection
import time
import sys
from com.base.public import logger, public
from com.object.obj_entity import sh50_baseinfo, sh510050, sh50_price, sh50_baseinfo_tmp
import json
import pandas as pd


class c_interface(object):
    # 读取url
    keylist = []

    def getI(self, url=None, type='xml'):
        if url is None:
            url = self.url

        result = requests.get(url)
        result.encoding = 'gb2312'
        return result.text

    def getText(self, nodelist):
        r = ""
        for nxd in nodelist.childNodes:
            r = r + nxd.nodeValue
        return r.strip()

    def post(self, url, headers, form):
        session = requests.session()
        session.headers.update(headers)
        response = session.post(url, data=form)
        return response.text

    def set(self, item):
        k = 0
        doc = {}
        for key in self.keylist:
            try:
                doc[key] = item[k]
                k = k + 1
            except:
                logger.error(sys.exc_info())
                continue
        return doc

# 新浪交易所期权信息
class sinaInterface(c_interface):
    # 字段说明
    url_510050 = "https://hq.sinajs.cn/list=s_sh510050"
    url_month = "http://stock.finance.sina.com.cn/futures/api/openapi.php/StockOptionService.getStockName"
    url_up = "http://hq.sinajs.cn/list=OP_UP_%s%s"
    url_down = "http://hq.sinajs.cn/list=OP_DOWN_%s%s"
    url_day = "http://stock.finance.sina.com.cn/futures/api/openapi.php/StockOptionService.getRemainderDay?date=%s"
    url_future = "http://hq.sinajs.cn/list=%s"

    def __init__(self):
        self.url = "https://hq.sinajs.cn/list="
        self.paramStyle = "CON_OP_%s"
        self.Base, self.Base_tmp = sh50_baseinfo(), sh50_baseinfo_tmp()
        self.codelist = []
        self.expireMap = {} #

    def baseInfo(self):
        # 清空历史数据
        #self.Base_tmp.empty()

        docs = []
        usedMon = []

        res = json.loads(self.getI(self.url_month))
        data = res["result"]["data"]
        print(data)

        for mon in data["contractMonth"]:
            # 过滤重复的月份
            if not mon in usedMon:
                usedMon.append(mon)
            else:
                continue
            doc = {
                "cateId": data["cateId"],
                "cateList": ",".join(data["cateList"]),
                "stockId": data["stockId"],
                "contractMonth": mon
            }

            # 查询到期日
            eTmp = json.loads(self.getI(self.url_day % mon))
            eDay = eTmp["result"]["data"]["expireDay"]

            # 获取上行和下行列表
            param = mon[2:].replace("-", "")

            res_up = self.extract(self.getI(self.url_up % (data["stockId"], param)))
            res_down = self.extract(self.getI(self.url_down % (data["stockId"], param)))
            k = 0

            for k in range(len(res_up)):
                if res_up[k].strip() == "": continue
                doc_new = {}
                doc_new.update(doc)
                doc_new.update({
                    "code": res_up[k][-8:],
                    "code_down": res_down[k][-8:],
                    "expireDay": eDay
                })
                docs.append(doc_new)
                k += 1

        # 添加到临时数据库
        #self.Base_tmp.insertAll(docs)
        # 更新正式库
        #self.Base_tmp.updateBase()
        logger.info(" update sh50 baseInfo")

    def get_price(self, type=None):
        self.keylist = sh50_price().keylist
        ary_code = []

        # 缓存
        if len(self.codelist) == 0:
            res = self.Base.getCodes()
            for doc in res:
                ary_code.append(doc["code"])
                ary_code.append(doc["code_down"])
                # expireDay
                self.expireMap[doc["code"]] = self.expireMap[doc["code_down"]] = doc["expireDay"]

            self.codelist = ary_code
        else:
            ary_code = self.codelist

        param = ",".join([self.paramStyle % c.strip() for c in ary_code])
        href = self.url + param
        # 查询代码
        res = self.getI(href)
        dfs = []

        i = 0
        # 写入到pandas
        for item in res.split(";")[:-1]:
            tmp = [ary_code[i], public.getDatetime()] + self.extract(item)
            dfs.append(self.set(tmp))
            # dfs.append([0 if n=="" else n for n in tmp])
            i += 1
        # df = pd.DataFrame(dfs, columns =self.keylist)
        return dfs

    def getETF(self, sign=1, price=2.30, code=None):
        df = pd.DataFrame(self.get_price(), columns =self.keylist, dtype=float)
        df['code'] = df['code'].astype(int).astype(str)

        if code is not None:
            return df[df['code']==code]
        else:
            Base = sh50_baseinfo()
            codes = Base.findCodes(sign=sign)
            df = df[(df['code'].isin(codes)) & ((df['power'] - price) * sign >=0) & ((df['power'] - price) * sign <=0.05)]
            return df
        pass

    # 解码文件为list
    def extract(self, item, shift=2):
        str = item.strip()[item.find("=") + shift:]
        str = str[:str.find('"')].strip()
        return str.split(",")

    def get510050(self):
        Owner = sh510050()
        self.keylist = Owner.keylist[:-1]
        res = self.extract(self.getI(self.url_510050))
        return self.set(res)


class TensentInfo:
    Interface = None

    def __init__(self):
        self.Int = c_interface()

    # 查询腾讯API，获取其他数据
    def getWar(self, code_list):
        ss = [n.replace("HK.", "hk") for n in code_list]
        href = "http://qt.gtimg.cn/q=" + ",".join(ss)
        res = self.Int.getI(href)

        if res.find('";') > 0:
            codes = res.split(";")

        docMap = {}
        for txt in codes:
            doc = {'exercise_price': 0, 'exchange_ratio': 0, 'due_date': ''}
            kl = txt[txt.find('="') + 2:-2].split("~")
            if len(kl) > 47:
                doc.update({'exercise_price': kl[45], 'exchange_ratio': kl[44], 'due_date': kl[47]})
                docMap['HK.' + kl[2]] = doc

        return docMap

# 简单webscoket链接
class s_webSocket(object):
    def __init__(self, href = "ws://116.62.112.161:8090/forward"):
         self.href = href

    def send(self,  str):
        times = 0
        ws = None
        while True:
            try:
                ws = create_connection(self.href)
                break
            except:
                times += 1
                if times > 3: break
                time.sleep(1)

        if ws:
            ws.send(str)
            print("ws send success")
        else:
            print("ws not connect")


def main():
    ws = sinaInterface()
    #docs = ws.getETF(code='10001329')
    #docs = ws.getETF(sign=1, price=2.64)
    #print(docs)
    ws.baseInfo()

    #res = ws.get510050()
    #print(res)

    #  & &

if __name__ == '__main__':
    main()
