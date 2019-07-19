# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

      期货多币对 多进程对冲交易程序
      针对 1分钟k线的测试
      多个币对按收盘时间统一为多个进程

"""

from multiprocessing import Pool
from com.ctp.interface_pyctp import  BaseInfo
import  time

from com.model.model_future_detect_single_ctp_v10 import model_future_detect_single, Manager2
import talib as ta
import pandas as pd

# 期货混合交易模型
class model_future_single_0(model_future_detect_single):

    def __init__(self):
        # 统一参数
        super().__init__()

        self.scaleDiff2 = 0.8
        self.powline = 0.25
        self.widthTimesPeriod = 3
        #

        self.topUse = True  # 启用定时统计列表
        self.isCTPUse = True  # 是否启用CTP接口（并区分
        self.isAutoAlterPosition = True #是否启用自动调仓
        self.isTickSave = False # 是否记录tick

        self.banCodeList = ['SC', 'CU']  # 暂时无权限操作的code
        self.isWorking = True
        self.isTest = False

        self.topTableName = 'train_total_dema4'
        self.bestTopNumbers = 3
        self.topNumbers = 7  # 最大新币对 (监控币对还包括历史10和未平仓对)
        self.sourceType = 'combin'

        self.topFilter = """ (count>10) """
        self.ctpuser = 'fz'
        self.minRate = 0.02  # 最低收益率

    def pool(self):
        pool = Pool(processes=5)

        # 初始化codes列表
        if self.topUse:
            self.filterCodes()

        share = Manager2()
        Base = BaseInfo([])
        CTP = share.interface_pyctp(use=self.isCTPUse, baseInfo=Base, userkey=self.ctpuser)
        #CTP = None
        Rice = None

        pid = 0
        for codes in self.seperate():
            print('pool send:', pid, len(codes))
            #if 'TA' not in codes:continue
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

def main():

    obj = model_future_single_0()
    #print(obj.__class__.__name__)
    obj.pool()


if __name__ == '__main__':
    main()
