# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein

      期货多币对 多进程对冲交易程序
      针对 1分钟k线的测试
      多个币对按收盘时间统一为多个进程

"""

import time
from multiprocessing import Pool, Manager

from com.model.model_future_detect_multi_ctp import model_future_detect_multi
# 期货混合交易模型
class model_future_detect_1m(model_future_detect_multi):

    def __init__(self):
        # 统一参数
        super().__init__()

        self.isCTPUse = True  # 是否启用CTP接口（并区分
        self.isAutoAlterPosition = True
        self.stopTimeLine = 10
        self.isWorking = True

        self.topFilter = "uid like '%_1_kline%'"
        self.sameLimit = 2  # 同一code允许出现次数
        self.topNumbers = 20  # 最大新币对 (监控币对还包括历史10和未平仓对)
        self.minRate = 0.01  # 最低收益率
        self.topTableName = 'train_total_1'
        self.methodName = '1m'  # 策略名称

    def pool(self):
        pool = Pool(processes=4)
        shareDict = Manager().list([])
        CTP = None
        # 初始化codes列表
        if self.topUse:
            self.filterCodes()

        for codes in self.seperate():
            #if "CU" in codes or "SR" in codes: continue
            #self.start(codes, shareDict, CTP)
            try:
                pool.apply_async(self.start, (codes, shareDict, CTP))
                time.sleep(3)
                pass
            except Exception as e:
                print('error', e)
                continue

        pool.close()
        pool.join()

    def orderCheck(self, cur, param):
        isOpen, isRun, isstop = self.procMap.isOpen, False, 0
        # 开仓
        if isOpen == 0:
            if param["delta"] > self.deltaLimit: return None
            # 已关闭的交易对只平仓， 不再开仓
            if self.procMap.codes in self.noneUsed: return None
            # 大于上线轨迹
            if param["p_h"] > param["top"]:
                isOpen, isRun = -1, True

            # 低于下轨线
            elif param["p_l"] < param["lower"]:
                isOpen, isRun = 1, True

        # 平仓
        else:
            stopMinutes = self.stopTimeLine * self.procMap.period * self.procMap.kline
            preNode = self.procMap.get('preNode', uid=self.uid)

            # 回归ma则平仓  或  超过24分钟 或到收盘时间 强制平仓
            if isOpen * (param["close"] - param["ma"]) >= 0:
                if param["delta"] > self.deltaLimit: return None
                isOpen, isRun = 0, True

            # 止损
            elif stopMinutes > 0 and preNode is not None:
                tdiff = self.Rice.timeDiff(str(preNode[0]['createdate']),quick=stopMinutes)
                if tdiff > stopMinutes:
                    isOpen, isstop = 0, 1
                    isRun = True

        if isRun:
            self.order(cur, isOpen, param, isstop=isstop)

def main():
    obj = model_future_detect_1m()
    obj.pool()


if __name__ == '__main__':
    main()
