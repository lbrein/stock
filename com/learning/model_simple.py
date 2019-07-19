# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ----- 对处理好的purchaseRecord表进行回归类选择策略 
"""
import numpy as np

# import sys

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVC,SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


# 回归方法
class modelSimple(object):
    """
       简单回归模型，用于确定特性相关性    
    """
    # 模型清单
    modelMap = {
         "Linear": LinearRegression(),
         "svc":SVC(),
        # "svr":SVR(),
      #   "DecisionTree":tree.DecisionTreeRegressor(),
      #   "KNeighbors": KNeighborsRegressor(),
         #"RLR": RandomizedLogisticRegression(),
         "RandomForest":RandomForestRegressor(n_estimators=10),
         # "AdaBoost":AdaBoostRegressor(n_estimators=50),
         "Logistic": LogisticRegression(),
         "MLP": MLPClassifier(activation='tanh', solver='adam', alpha=0.0001, learning_rate='adaptive',
                                learning_rate_init=0.001, max_iter=200),
      #   "Gradien":GradientBoostingRegressor(n_estimators=100)
    }

    logMap = {
        "Logistic": LogisticRegression(),
    }

    np_ary, mat = None, None

    def train(self, methods=[], x=None, y=None, mode='all'):

        if x.size == 0 or y.size == 0 or y.shape[0] == 0:
            return None

        objMap = self.modelMap
        if mode == 'log': objMap = self.logMap

        if len(methods) == 0:
            methods = objMap.keys()

        for name in methods:
            model = objMap[name]

            if name == 'RandomForest':
                y1 = np.array([1 if c > 0 else -1 for c in y]).ravel()

            elif name in ['svc', 'Logistic', 'RLR', 'MLP']:
                y1 =np.array([int(c*100) for c in y])
                #print(y1.tolist())

            else:
                y1 = y

            x0, y0 = x, y1
            if name in ["Linear"]:  # 线性回归不支持(22,)模式
                x0 = self.reshp(x)
                y0 = self.reshp(y)

            # 分割测试集
            x_train, x_test, y_train, y_test = train_test_split(x0, y0, test_size=0.2, random_state=None)
            if y_train.shape[0] == 0: return None

            # 模型测算              
            model.fit(x_train, y_train)
            p0 = 0
            if name not in ['RLR']:
                p0 = model.score(x_test, y_test)

            print(p0)

            # 结果分析 
            if abs(p0) < 1 and \
                    ((name == "Linear" and p0 > 0.032)
                     or (name == "RandomForest") \
                     or (name == "KNeighbors" and p0 > 0.5) \
                     or name == "Logistic" and p0 > 0.48
                    ) or True:

                record = {
                    "type": name,
                    "score": p0,
                    "c0": 0,
                    "c1": 0,
                    "c2": 0,
                    "c3": 0,
                    "c4": 0,
                    "c5": 0,
                    "c6": 0,
                    "c7": 0,
                    "c8": 0,
                    "c9": 0,
                    "amount": y0.shape[0],
                }

                # logger.info(("model: ",name ,"  score: ", p0))
                if name in ["Linear"]:
                    c, c0 = model.coef_[0], model.intercept_[0]
                    # logger.info ((model.coef_,model.intercept_))
                    record["c0"] = c0
                    for i in range(len(c)):
                        record["c" + str(i + 1)] = c[i]

                elif name in ["RandomForest"]:
                    record["weightlist"] = model.feature_importances_

                elif name in ['RLR']:
                    record["weightlist"] = model.get_support()

                # 返回记录
                yield record
            else:
                yield None

    def reshp(self, na):
        if len(na.shape) == 1:
            return na.reshape((na.size, 1))
        else:
            return na

    def modelload(self):
        return joblib.load(self.modelfile)


def main():
    pass


if __name__ == '__main__':
    main()
