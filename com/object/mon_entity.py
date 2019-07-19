# -*- coding: utf-8 -*-
"""
Created on 2018-1-5
@author: lbrein
        mongodb 数据库映射类
"""

from com.base.public import baseObject, public
from datetime import datetime
import pandas as pd


# 币对组合比价路径类
class m_history(baseObject):
    colName = "history"

    # 查询所有比对
    def getPrice(self, code, start=None):
        if start is not None:
            d = start
            if type(d) == str:
                d = datetime.datetime.strptime(start, '%Y-%m-%d')
            res = self.col.find({"code": code, 'date': {'$gt': d}})
        else:
            res = self.col.find({'code': code})
        return res


class mon_tick(baseObject):
    colName = 'tick'

    def getTick(self, code, count=5000, method='dema5'):
        docs = self.col.find({"code": code, "method":method}, sort=[("datetime", -1)], skip=0, limit=count)
        return [doc for doc in docs][0:count]

class stock_tick(baseObject):
    colName = 'stock_tick'
    def getTick(self, code, count=5000):
        docs = self.col.find({"code": code}, sort=[("datetime", -1)], skip=0, limit=count)
        return [doc for doc in docs][0:count]

class mon_trainOrder(baseObject):
    colName = 'trainOrder'

    def getTick(self, stage, code=None, columns=None):
        filter = {"method": stage}
        if code is not None:
            filter.update({"code":code})

        print(filter)
        docs = self.col.find(filter)

        if columns is None:
            return pd.DataFrame([doc for doc in docs])
        else:
            return pd.DataFrame([doc for doc in docs], columns=columns)

    def drop(self, stage):
        self.col.remove({"method": stage})

def main():
    map = {
        "tick": 0,
    }

    if map["tick"] == 1:
        pass


if __name__ == '__main__':
    main()
