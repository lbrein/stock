# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

      期货单品种交易

"""


from multiprocessing import Pool, Manager
from com.model.model_future_zhao_ctp_v1 import model_future_zhao_v1
from com.ctp.interface_pyctp import BaseInfo
from com.model.model_future_ctp import  Manager2
import numpy as np
import time

# 期货混合交易模型
class model_future_zhao(model_future_zhao_v1):

    def __init__(self):
        # 统一参数
        super().__init__()

        self.isWorking = False  # 是否正式运行
        self.isAutoAlterPosition = True
        self.isTest = False

        self.isTickSave = False
        self.topUse = False  # 启用定时统计列表
        self.volumeCalcType = 0  # 交易量计算类别
        self.ctpuser = 'fz'

        self.methodName = 'zhao'  # 策略名称
        self.relativeMethods = ['zhao', 'zhao55']

        self.iniAmount = 4000000  # 初始总金额
        self.stopLine = 0.0025

    def iniWorking(self):
        if self.isWorking:
            self.orderFormTest = 'future_orderForm'
            self.isCTPUse = True

        else:
            self.orderFormTest = 'future_orderForm'
            self.isCTPUse = False
            self.ctpuser = 'simnow1'

    def pool(self):
        pool = Pool(processes=5)
        self.iniWorking()
        # 初始化codes列表
        self.filterCodes()
        share = Manager2()
        #CTP = share.interface_pyctp(use=self.isCTPUse, baseInfo=BaseInfo([]), userkey=self.ctpuser)
        CTP = None
        Rice = None

        pid = 0
        for codes in self.seperate():
            #if 'RB' not in codes: continue
            print('pool send:', pid, len(codes))
            #self.start(codes, Rice, CTP)
            try:
                pool.apply_async(self.start, (codes, Rice, CTP))
                time.sleep(3)
                pid += 1
                pass
            except Exception as e:
                print(e)
                continue

        pool.close()
        pool.join()

from com.ctp.interface_pyctp import BaseInfo, ProcessMap,interface_pyctp
from com.data.interface_Rice import interface_Rice
from com.object.obj_entity import future_orderForm

def test():
    obj = model_future_zhao()
    obj.isCTPUse = True

    obj.procMap  = ProcessMap()
    obj.Rice = interface_Rice()
    Rec = future_orderForm()
    openMap = Rec.getOpenMap('zhao', codes=['RB'], batchNum=1)
    obj.CTP = interface_pyctp(use=obj.isCTPUse, userkey=obj.ctpuser)
    obj.CTP.baseInfo = BaseInfo(['RB'], obj.Rice)

    print('------------', openMap)
    key, obj.uid = 'RB', 'RB_40_2.0_1_0_zhao'
    obj.procMap.setIni(obj.uid, openMap[key], status=0)

    doc = {'createdate': '2019-06-25 09:40:27', 'code': 'RB1910', 'name': 'RB', 'symbol':
        'rb1910', 'price': 3962.0, 'vol': 30.0, 'hands': 1.0, 'ini_hands': 1.0, 'ini_price': 3962.0,
           'mode': -2, 'isopen': 0, 'isstop': 1, 'fee': 1.1885999699734384, 'income': 0,
           'rel_price': 3962.0, 'stop_price': 3650.0, 'batchid': '359b8070-96ea-11e9-ab77-382c4a6d1b55',
           'status': 0, 'method': 'zhao', 'uid': 'RB_40_2.0_1_0_zhao', 'istoday':1}

    doc1 = {'createdate': '2019-06-25 10:02:27', 'code': 'EG1909', 'name': 'EG', 'symbol': 'eg1909', 'price': 4552.0,
     'vol': 10.0, 'hands': 1, 'ini_hands': 1, 'ini_price': 4552.0, 'mode': 0, 'isopen': 0, 'isstop': 0, 'fee': 9.0,
     'income': 0, 'rel_price': 4552.0, 'stop_price': 0, 'batchid': '', 'status': 0, 'method': 'zhao',
     'uid': 'EG_40_2.0_1_0_zhao'}

    obj.setIncome([doc], 0)
    obj.record([doc], -2)

def main():
    obj = model_future_zhao()
    obj.pool()


if __name__ == '__main__':
    #test()
    main()
