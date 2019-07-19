# -*- coding: utf-8 -*-

""" 
Created on  2018-01-04 
@author: lbrein
       ---  FUTu 信息接口 ---      
 
"""

from com.base.public import public
from com.object.obj_entity import future_baseInfo
from com.object.obj_model import future_code

from com.data.interface_Rice import interface_Rice
import time

class data_future(object):

     def __init__(self):
         self.usedLine = 0.7
         pass

     def his(self):
         self.Rice = Rice = interface_Rice()
         Base = future_baseInfo()
         bCode = future_code()
         bCode.empty()

         years = [ str(y)[-2:]  for y in range(2010, 2021)]
         months = [ ('0' if m < 10 else '') + str(m) for m in range(1, 13)]
         lists = Base.getUsedMap(hasIndex=True)

         for c in lists:
             docs = []
             mains = Rice.getAllMain(c, start='2010-01-01')
             mainMap = {}

             # 主力合约日期清单
             for ma in mains:
                 mainMap[ma[1]] = str(ma[0])
             # 第一个主力合约的初始交易量
             mVol = self.getVol(mains[0][1], str(mains[0][0]))

             # 循环查询合约交易
             for y in years:
                 for m in months:
                     mcode = '%s%s%s' % (c, y, m)
                     obj = Rice.detail(mcode)
                     if obj is not None:
                        doc = obj.__dict__
                        mC = doc['order_book_id']
                        doc.update({
                            'code': mC,
                            'name': doc['underlying_symbol'],

                        })

                        d, de, v = self.getUsedVol(mC, start = doc['listed_date'], end =doc['de_listed_date'] , mVol =mVol)
                        if d is not None:
                            doc['used_date'], doc['de_used_date'], doc['used_volume'] = d, de, v

                        if mC in mainMap:
                            doc['main_date'] = mainMap[mC]
                            doc['used_volume'] = mVol = self.getVol(mC, mainMap[mC])

                        docs.append(doc)

             print(doc['name'], len(docs))
             time.sleep(0.2)
             bCode.insertAll(docs)
             #break

     def getVol(self, code, start=None):
         df = self.Rice.kline([code], period='1d', start=start,  end=start, pre=1)[code]
         df = df[df.index >= start]
         return  df.iloc[0]['volume'] if len(df)> 0 else None


     def getUsedVol(self, code, start=None, end=None, mVol=10000):
         if start < '2010-01-01': start = '2010-01-01'
         df = self.Rice.kline([code], period='1d', start=start, end=end, pre=1)[code][start:]
         if mVol is None or mVol==0:    mVol=10000
         df = df[(df['volume'] / mVol > self.usedLine) | (df['volume'] > 50000)]

         if len(df) > 0:
            return df.index[0], df.index[-1], df.iloc[0]['volume']
         else:
            return None, None, None

     def cur(self):
         Rice = interface_Rice()
         res = Rice.getActive()

         print(len(res))
         for r in sorted(res):
             print(r, res[r])


def main():
    action = {
        "his": 1,
        "cur": 0
    }

    obj = data_future()
    if action["his"] == 1:
        obj.his()

    if action["cur"] == 1:
        obj.cur()

if __name__ == '__main__':
    main()
