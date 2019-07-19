# -*- coding: utf-8 -*-
"""
Created on 2019 - 6 - 21
@author: lbrein

 ----- 大师期权策略：买最低价值策略

"""

from com.base.public import public
from com.object.obj_entity import  train_future, option_tmp, sh50_daily
import uuid
from com.data.interface_Rice import interface_Rice
from com.option.data_Option_Rice import data_Option_Rice
import copy


# 回归方法
class train_option_minBuy(object):
    """

    """
    def __init__(self):
        # 费率和滑点
        self.iniAmount = 100000

        self.isNewTmp = False

        self.ratio = 2
        self.contract_multiplier = 10000
        self.klinePeriodList = ['1d']
        self.ownerCode = '510050.XSHG'

        self.startDate = '2015-02-01'
        self.endDate = public.getDate()
        self.method = 'minBuy'
        self.total_tablename = 'train_total_2'
        self.detail_tablename = 'train_future_3'
        self.uidKey = '%s_option_' + self.method

        # 查询参数写入到param表
    def pool(self):

        self.Rice = interface_Rice()
        # 临时sql
        self.Tmp = option_tmp()

        if self.isNewTmp:
            self.Tmp.empty()
            self.saveETF()

        #
        self.Train = train_future()
        self.Train.tablename = self.detail_tablename
        self.Train.empty()

        Option = data_Option_Rice()

        for m in ['C', 'P']:
            # 查询获得期权列表，以{code: expareDate} 字典表返回
            codeList = Option.allCodes(mode=m)
            self.start(codeList, m)

    # 保存正股
    def saveETF(self):
        Daily = sh50_daily()
        Daily.empty()
        ow = self.ownerCode
        df_ow = self.Rice.kline([ow], period=self.klinePeriodList[0], start=self.startDate, end=self.endDate, pre=1, type='ETF')[ow]
        print(df_ow)
        df_ow.loc[:, 'date'] = df_ow.index
        Daily.insertAll(df_ow.to_dict(orient='record'))


    def saveTmp(self, dfs, codeList):
        for c in self.codes:
            df = dfs[c]
            df.loc[:, 'code'] = c
            df.loc[:, 'type'] = self.Options
            df.loc[:, 'datetime'] = df.index
            df.loc[:, 'd_date'] = codeList[c]
            df.loc[:, 'monthdiff'] = df['datetime'].apply(lambda x: public.monthDiff(str(x), codeList[c]))

            docs = copy.deepcopy(df[(df['monthdiff'] == 2) & (df['close'] != 0)]).to_dict(orient='record')
            if len(docs) > 0:
                print(c, len(docs))
                self.Tmp.insertAll(docs)

    def start(self, codeList, m):
        #
        self.codes = [c for c in codeList.keys()]
        self.Options = m

        period = self.klinePeriodList[0]

        # 读取日线历史数据
        dfs = self.Rice.kline(self.codes, period=period, start=self.startDate, end=self.endDate, pre=1,
                              type='Option')

        #

        # 将数据整理保存到sqlserver
        if self.isNewTmp:
            self.saveTmp(dfs, codeList)

        # 从sqlserver 分组取每日最低价格的期权清单
        orders = self.Tmp.getOrders(type=m)

        # 模拟交易
        self.stageApply(orders, dfs)

    def stageApply(self, orders, dfs):
        isOpen, preNode, preCode = 0, None, None
        doc, docs =  {}, []

        # 按顺序读取每日最低值期权，卖掉非最低值，买入最低值
        self.index = uuid.uuid1()
        for i in range(len(orders)):

            doc = orders[i]

            if preCode!=doc['code']:
                if preCode is not None:
                    doc0 = dfs[preCode].loc[doc['datetime']]
                    doc0["datetime"]= str(doc['datetime'])
                    doc0["code"] = preCode

                    docs.append(self.order(doc0, 0))

                preCode = doc['code']
                docs.append(self.order(doc, 1))

        if len(docs) > 0:
            print(len(docs))
            self.Train.insertAll(docs)

    preNode, index = None, None
    def order(self, n0,  mode):

        # 交易量
        v0 = round(self.iniAmount/ ( n0["close"] * self.contract_multiplier + self.ratio *2), 0)
        # 费率
        fee0 = self.ratio * v0

        doc = {
            "createdate": n0["datetime"],
            "code": n0["code"],
            "price": n0["close"],
            "vol": v0,
            "mode":  mode if not self.preNode else -self.preNode["mode"],
            "isopen": 0 if mode == 0 else 1,
            "fee": fee0,
            "income": 0,

            # 类型
            "options": 1 if self.Options=='C' else -1,
            "batchid": self.index
        }

        if mode == 0 and self.preNode:
            p0 = self.preNode
            doc["income"] = (n0["close"] - p0["price"]) * p0["vol"] * self.contract_multiplier - p0["fee"]
            self.preNode = None
            self.index = uuid.uuid1()

        else:
            doc["income"] = -doc["fee"]
            self.preNode = doc

        return doc

def main():
    obj = train_option_minBuy()

    obj.pool()

if __name__ == '__main__':
    main()

