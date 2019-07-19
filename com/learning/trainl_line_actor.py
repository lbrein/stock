# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ----- 期货训练模型 - 单
"""

from com.base.public import public, logger
import pandas as pd
import talib as ta
import numpy as np
import copy
import uuid
from com.data.interface_Rice import interface_Rice
from com.object.mon_entity import mon_trainOrder
from com.object.obj_entity import future_baseInfo
from com.model.model_simple import modelSimple
import itchat

# 回归方法
class future_train_lineactor(object):
    """

    """
    stageMap ={
        'filter0': {
            'columns':[]
        }
    }

    def __init__(self):
        # 费率和滑点
        self.stage = 'dema5'
        self.columns0 = ['vol', 'ma', 'std',
                         'width', 'wd1', 'wdstd', 'slope', 'slope60', 'kdj_d2', 'sard', 'atr', 'atr1',
                         'atrb', 'wdd', 'pow', 'trend', 'rsi']

        self.columns1 = ['width', 'wd1', 'wdstd', 'slope', 'slope60', 'kdj_d2', 'sard', 'atr', 'atr1',
                         'wdd', 'trend', 'vol', 'rsi']

        self.Rice = interface_Rice()
        self.Model = modelSimple()
        self.modeMap = {'5': [4, 5, -4, -5], '6': [6, -6], '2': [1, 2, -1, -2], '3': [3, -3]}

        pass

    def start(self, stage=None):
        Source = mon_trainOrder()
        if stage is None: stage = self.stage

        df = Source.getTick(stage, 1)

        df['income_b'] = df['income'].apply(
            lambda x: 0 if abs(x) < 250 else np.sign(x) * 0.5 if abs(x) < 1500 else round(x / 2500, 0) if abs(
                round(x / 2500, 0)) < 3 else 2 * np.sign(x))

        c = copy.deepcopy(self.columns1)
        df0 = copy.deepcopy(df)

        for key in c:
            df[key + '_b'] = df[key].apply(lambda x: abs(x))

            # 归一化
            mi, mx = df[key].min(), df[key].max()
            df0[key] = df[key].apply(lambda x: (x - mi) / (mx - mi))
            # 绝对值
            mi, mx = df[key + '_b'].min(), df[key + '_b'].abs().max()
            df0[key + '_b'] = df[key + '_b'].apply(lambda x: (x - mi) / (mx - mi))

            self.columns0.append(key + '_b')
            self.columns1.append(key + '_b')

        #  分类别统计 sub 子串
        docs = []
        for mm in self.modeMap:
            # 线性回归
            res = self.Model.train(x=np.array(df0.loc[:, self.columns1]), y=np.array(df0.loc[:, ["income_b"]]),
                                   xcolumns=self.columns1)
            for d in res:
                if d is not None:
                    print(" -----stage----", mm)
                    d['key'] = mm
                    print(d)
                    docs.append(d)

            # cvs group 结果

            df1 = df[df['mode'].isin(self.modeMap[mm])].loc[:, self.columns0 + ['income_b']]
            gp1 = df1.groupby('income_b').mean()
            gp1['size'] = df1.groupby('income_b').size()
            gp1.to_csv(self.Rice.basePath + 'byIncome_%s_%s.csv' % (self.stage, mm))

            df1 = df0[df0['mode'].isin(self.modeMap[mm])].loc[:, self.columns0 + ['income_b']]
            gp1 = df1.groupby('income_b').mean()
            gp1['size'] = df1.groupby('income_b').size()
            gp1.to_csv(self.Rice.basePath + 'byIncome2_%s_%s.csv' % (self.stage, mm))

        pd.DataFrame(docs).to_csv(self.Rice.basePath + 'byIncome_linear.csv')

    def groupOutput(self, stage, unit=5000):
        Source = mon_trainOrder()
        if stage is None: stage = self.stage
        df = Source.getTick(stage, isBuy =1)

        df['income_b'] = df['income'].apply(lambda x: round(x / unit, 0) * unit)

        df1 = df
        gp1 = df1.groupby('income_b').mean()
        gp1['size'] = df1.groupby('income_b').size()
        gp1.to_csv(self.Rice.basePath + 'byIncome2_%s.csv' % (stage))

    pass


            # 简单策略测试
    # 跳空策略

    def stage_jump(self):
        Base = future_baseInfo()
        # 交易量大的，按价格排序, 类型:turple,第二位为夜盘收盘时间
        Rice = interface_Rice()
        codes = [l[0] for l in Base.all(vol=100)]
        mCodes = Rice.getMain(codes)

        docs, res, doc = [], [], {}
        period = '5'

        for i in range(len(codes)):
            klines = Rice.kline([mCodes[i]], period='%sm' % period, start=public.getDate(diff=-10))

            df = klines[mCodes[i]].dropna().reindex()
            print('width15: ', i, codes[i], mCodes[i], len(df))

            if len(df) > 0:
                # 计算跳空幅度和跳空前趋势
                df['date'] = df.index
                df['time'] = df['date'].apply(lambda x: str(x)[11:].strip())
                sum, c = 0, 0

                for j in range(1, len(df)):

                    if df.ix[j, 'time'] in ['09:05:00' , '21:05:00']:
                        s0, e0, s1, e1 = df.ix[j - 1, 'open'], df.ix[j - 1, 'close'], df.ix[j, 'open'], df.ix[
                            j, 'close']
                        income = np.sign(e0 - s0) * (s1 - e0)

                        if income!=0:
                            docs.append({'date': df.ix[i, 'date'], 'open':s0, 'close':e0, 'open1':s1, 'income':income})
                            sum += income
                            c+=1

                #break
                res.append({'code':codes[i], 'sum':sum , 'rate': sum/df['close'].mean(), 'count':c})
                print(codes[i], sum, sum/df['close'].mean(), c)

        df0 = pd.DataFrame(docs)
        df0.to_csv(self.Rice.basePath + 'stage_jump_detail_%s.csv' % period)
        pd.DataFrame(res).to_csv(self.Rice.basePath + 'stage_jump_total_%s.csv' % period)


def main():
    action = {
        "line": 0,
        "group":0,
        "jump": 0,
        "atr":1,
    }

    obj = future_train_lineactor()
    if action['line']==1:
          obj.start()

    if action['group']==1:
          obj.groupOutput('obv')

    if action['jump']==1:
          obj.stage_jump()

    if action['atr'] == 1:
        codes = "AP,CS,C".upper().split(",")
        obj.getAtr(codes)


if __name__ == '__main__':
    main()
