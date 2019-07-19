# -*- coding: utf-8 -*-

""" 
Created on  2018-06-22
@author: lbrein
    广发文件报单接口


"""

from com.base.public import csvFile, config_ini, public, ftpFile
from com.object.obj_entity import maticIndex
import pandas as pd
import uuid
import time
import copy
"""
local_group_no
local_group_name
local_report_no
projectid
market
stkcode
hedgeflag
bsflag
price
qty
diy1
diy2
instr_xxx.yyyyMMddHHmmssSSS
"""

class ProcessMap(object):
    modelParamMap = {
        "pop_01": [['timePeriods', 'dropLine', 'sarStart', 'sarEnd'], [60, 0.2, 0.035, 0.035]]
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
            "isOpen": 0, # 策略状态 1-买多 -1 -买空 0-平
            "isBuy": -1,
            "batchid": '',
            "preNode": None,  # 之前的节点
        }

    @property
    def isOpen(self):
        return self.map[self.currentUid]['isOpen']

    @isOpen.setter
    def isOpen(self, value):
        self.map[self.currentUid]['isOpen'] = value

    @property
    def isBuy(self):
        return self.map[self.currentUid]['isBuy']

    @isBuy.setter
    def isBuy(self, value):
        self.map[self.currentUid]['isBuy'] = value

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
        return self.map[uid][name]

    def set(self, name, value, uid=None):
        if uid is None: uid = self.currentUid
        if uid not in self.map.keys(): self.new(uid)
        self.map[uid][name] = value

    def setModelParam(self, map, modelname, num=1):
        if modelname in self.modelParamMap:
            params, defaults = self.modelParamMap[modelname][0], self.modelParamMap[modelname][1]
            for i in range(0, len(params)):
                j = num + i
                try:
                    self.__setattr__(params[i], float(map[j]) if map[j].find('.') > -1 else int(map[j]))
                except:
                    self.__setattr__(params[i], defaults[i])

    def setUid(self, uid, method='pop'):
        """ 设置Uid """
        map = uid.split("_")
        self.codes = map[0:1]  # 代码
        self.currentUid = map[0]

        # 不同模型设置参数
        self.setModelParam(map, method)
        return uid

    def setIni(self, uid, docs, status=0):
        """ 初始化节点 """
        if not uid in self.map.keys(): self.new(uid)
        self.map[uid]['isOpen'] = 1
        self.map[uid]['batchid'] = docs['batchid']
        self.map[uid]['preNode'] = docs
        return docs


class GuangFa_interface:
    # 报单配置
    matic_csv_map = {
        # 股票报单
        'stock_order': {
            'filename': 'instr_%s.%s.csv',

            'columns':
                [
                    ('local_group_no', '', '1'),  #
                    ('local_group_name', 'code', 'sh50'),
                    ('local_report_no', 'uid', '10001'),
                    ('projectid', 'productid', ''),
                    ('market', 'market', ''),
                    ('stkcode', 'code', ''),
                    ('hedgeflag', '', '0'),
                    ('bsflag', 'mode', ''),
                    ('price', 'price', ''),
                    ('qty', 'volume', ''),
                    ('diy1', '', ''),
                    ('diy2', '', '')
                ]
        },
        # 撤单
        'stock_cancel': {
            'filename': 'cancel_%s.%s.csv',
            'columns': [
            ]
        }
    }

    """
    market:
        上海'1',深圳'0',沪港通 '8',深圳通'G',中金'F',上期'H',郑商'Z',大商'D'
    
    hedgeflag:
        '0'投机,'1'保值,'2'套利 非期货默认送0     
    
    bsflag:
        
        0B 证券买入,0S证券卖出,3B 债券买入,3S债券卖出
        1U 多头开仓,1V 多头平仓,1W 空头开仓,1X 空头平仓
    """

    bsflag_map = {
        "stock_buy": "0B",
        "stock_sell": "0S",
        "debt_buy": "3B",
        "debt_sell": "3S",
        "futures_long_open": "1U",
        "futures_long_close": "1V",
        "futures_short_open": "1W",
        "futures_short_close": "1X"
    }

    # 市场类别
    market_map = {'SH': '1',
                  'SZ': '0',
                  'SHK': '8',  # G-沪港通
                  'SZK': 'G'  # S-深港通
                  }

    action_map = {
        "orders": ["instr", "cancel"],
        "results": ["result_instr", "result_cancel"],
        "records": ["order", "match", "asset", "stkhold"]
    }
    # 文件日期格式
    dateStyle = "%Y%m%d"
    # 文件时间格式
    timeStyle = '%Y%m%d%H%M%S%f'

    account_info={
        "account": '1875',
    }

    def __init__(self):
        self.Folder = self.iniKey("orderPath")
        self.stage = self.iniKey("stage")
        self.productid = int(self.iniKey("productid"))
        self.company = "GuangFa"
        self.account = "1875"
        self.maticIndex = maticIndex()
        # 初始Index
        self.recordIndex = self.getIndex()
        self.ftpUsed = True
        self.iniHold, self.iniFund = None, None

    def iniKey(self, key, db='GuangFa'):
        return config_ini.get(key, session=db)

    def check(self, doc):
        df = self.iniHold
        if self.iniHold is None:
           self.iniHold = df = self.getHold()

        if self.iniFund is None:
            self.iniFund, asset = self.getFree()

        if doc['mode'] > 0:
            if self.iniFund > doc['amount']:
                self.iniFund -= doc['amount']
                return True

        elif doc['mode'] < 0 and df is not None and len(df) > 0:
            code = int(doc['code'])
            try:
                m = df[df['stkcode']==code & df['projectid']==self.productid & df['stkholdqty']>= doc['vol']]
                if len(m) > 0:
                    return True
            except:
                # 后续改为False
                return True
        return False

         # 下单
    def order(self, docs, type='stock_order', callback=None):
        if not docs or len(docs) == 0:   return None

        maps = self.matic_csv_map[type]

        # 文件夹对应文件
        filename = maps["filename"] % (self.stage, str(public.getDatetime(style=self.timeStyle))[:-3])
        CSV = csvFile(folder=self.Folder,
                      filename=filename,
                      headers=[n[0] for n in maps["columns"]])

        orders = []
        for doc in docs:
            order = copy.deepcopy(doc)
            c = doc['code']
            order.update({
               "code": c[:c.find('.')],
               "market": 'SZ' if c.find('XSHE')>-1 else 'SH' if c.find('XSHG')>-1 else '',
               'mode': doc['isBuy'],
               'type': "stock",
               'volume': doc['vol'],
               'createtime': public.getDatetime()
            })

            # 检查资金和持仓
            if not self.check(order): continue

            rec = self.parseDoc(order, "stock_order")
            CSV.writeLine(rec)

            doc.update(
                {
                 "orderID": rec["local_report_no"],
                 "returnid": rec["local_report_no"],
                })
            orders.append(doc)

        CSV.close()
        # ftp 上传
        if self.ftpUsed:
            self.ftpUpload(filename)

        # 保存序号, 待bug修复后 改为uuid
        self.setIndex(self.recordIndex)

        # 检查操作结果，回写数据库

        return self.result(orders)

        #orderRecord().insertAll()

    def parseDoc(self, doc, type):
        maps = self.matic_csv_map[type]
        rec = {}
        for item in maps["columns"]:
            if item[1] in ['price', 'volume', 'code']:
                rec[item[0]] = doc[item[1]]

            elif item[1] == 'mode':
                key = '%s_%s' % (doc["type"], ("buy" if doc["mode"] > 0 else "sell"))
                rec[item[0]] = self.bsflag_map[key]

            elif item[1] == 'market':
                rec[item[0]] = self.market_map[doc[item[1]]]

            elif item[1] == 'productid':
                rec[item[0]] = self.productid

            elif item[1] == 'uid':
                rec[item[0]] = self.recordIndex
                self.recordIndex += 1

            elif item[2] == '$uuid':
                rec[item[0]] = self.get_uuid()

            else:
                rec[item[0]] = item[2]
        return rec

    # 撤单
    def cancel(self, docs, param=None):
        pass

    # 检查结果,
    def result(self, docs=None):
        # 读取执行结果，将记录写入status
        time.sleep(2)
        resDf = self.getCsv("results")
        for doc in docs:
            df = resDf[resDf["local_report_no"] == doc["returnid"]]
            if len(df)>0:
                 doc["status"] = df["result"].values[0]
        return docs

    def getMatch(self):
        return self.getCsv(record_type="match")

    # 查询持仓情况
    def getHold(self):
        return self.getCsv(record_type="stkhold")

    # 查询资金余额和总资产
    def getFund(self):
        return self.getCsv(record_type="asset")

    def getFree(self):
        df = self.getFund()
        if df is None: return 0, 0

        sf = df[df["projectid"] == self.productid]
        return sf["instravl"].values[0], sf["stkasset"].values[0]

    # 查询获得code的持股成本和数量
    def getHoldByCode(self, code, account=None):
        # 获得股票持股成本和量
        df = self.getHold()
        sf = df[df["stock_code"] == code]
        return sf["costprice"].values[0], sf["stkholdqty"].values[0]

    # ftp 上传
    def ftpUpload(self, filename):
        ftp = ftpFile()
        ftp.upload(self.Folder, filename)
        ftp.close()

    def ftpDownload(self, filename):
        ftp = ftpFile()
        ftp.download(self.Folder, filename)
        ftp.close()

    # 通过ftp，读取csv结果文件
    def getCsv(self, action="records", account=None, record_type="stkhold"):
        # 限定操作类型
        if not action in self.action_map.keys():
            return None

        if action == "records":
            if not record_type in self.action_map[action]:
                return None

        diff = 0
        filename = self._getFilename(action=action, record_type=record_type, diff=diff)
        if self.ftpUsed:
            k = 0
            # 查询当天或者最近一天的文件
            while 1:
                try:
                    self.ftpDownload(filename)
                    break
                except:
                    diff += -1
                    k += 1
                    filename = self._getFilename(action=action, record_type=record_type, diff=diff)
                    if k > 10: return None

        folder = self.Folder
        try:
            df =  pd.read_csv(folder + '/' + filename, encoding='gb2312')
            #print(df is None)
            return df
        except:
            return None

    def _getFilename(self, action="records", record_type="stkhold", diff=0):
        cur_date = str(public.getDatetime(diff=diff, style=self.dateStyle))

        filename = ''
        if action == "records":
            filename = "%s.%s.csv" % (record_type, cur_date)
        elif action == "results":
            filename = "result_instr.%s.csv" % cur_date

        return filename

    def get_uuid(self):
        return str(uuid.uuid1())

    # 每日序列号
    def _link_maticIndex(self):
        if self.maticIndex is None:

            self.maticIndex = maticIndex()

    def getIndex(self):
        self._link_maticIndex()
        return self.maticIndex.getIndex()

    def setIndex(self, index):
        self._link_maticIndex()
        self.maticIndex.setIndex(index)

def test():
    docs = []
    doc = {
        "code": 159901,
        "price": 4.27,
        "volume": 10000,
        "amount": 100000,
        "market": 'SH',
        "type": "stock",
        "mode": 1
    }
    docs.append(doc)
    obj = GuangFa_interface()
    obj.order(docs)


def main():
    actionMap = {
        "order": 0,
        "hold": 0,
        "result": 0,
        "match": 1
    }

    obj = GuangFa_interface()
    if actionMap["order"] == 1:
        obj.order([])

    if actionMap["hold"] == 1:
        res = obj.getHold()
        print(res)

    if actionMap["result"] == 1:
        obj.result()

    if actionMap["match"] == 1:
        res = obj.getMatch()
        print(res)


if __name__ == '__main__':
    main()
    # test()
