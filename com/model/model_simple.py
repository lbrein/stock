# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ----- 对处理好的purchaseRecord表进行回归类选择策略 
"""
# import numpy as np

# import sys
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVC,SVR
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn import tree
# from sklearn.ensemble import ExtraTreesClassifier

from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

# 回归方法
class modelSimple(object):
    """
       简单回归模型，用于确定特性相关性    
    """
    # 模型清单
    modelMap = {
        "Linear": LinearRegression(),
        # "svc":SVC(),
        # "svr":SVR(),
        # "DecisionTree":tree.DecisionTreeRegressor(),
        # "KNeighbors": KNeighborsRegressor(),
        #  "RandomForest":RandomForestRegressor(n_estimators=100),
        # "AdaBoost":AdaBoostRegressor(n_estimators=50),
        # "Gradien":GradientBoostingRegressor(n_estimators=100)
    }

    logMap = {
        "Logistic": LogisticRegression(),
    }

    np_ary, mat = None, None

    def train(self, methods=[], x=None, y=None, mode='all', xcolumns=None):

        if x.size == 0 or y.size == 0 or y.shape[0] == 0:
            return None

        objMap = self.modelMap
        if mode == 'log': objMap = self.logMap

        if len(methods) == 0:
            methods = objMap.keys()

        for name in methods:
            record = {}
            model = objMap[name]
            p0 = 0

            x0, y0 = x, y
            if name in ["Linear"]:  # 线性回归不支持(22,)模式
                x0 = self.reshp(x)
                y0 = self.reshp(y)

            # 分割测试集
            x_train, x_test, y_train, y_test = train_test_split(x0, y0, test_size=0.2, random_state=None)
            if y_train.shape[0] == 0: return None

            # 模型测算              
            model.fit(x_train, y_train)
            p0 = model.score(x_test, y_test)

            # 结果分析 
            if abs(p0) < 1 and \
                    ((name == "Linear" and p0 > 0.0)
                     or (name == "RandomForest" and p0 > 0.10) \
                     or (name == "KNeighbors" and p0 > 0.5) \
                     or name == "Logistic" and p0 > 0.48
                    ):
                record = {
                    "type": name,
                    "score": p0,
                    "amount": y0.shape[0],
                }

                # logger.info(("model: ",name ,"  score: ", p0))
                if name in ["Linear"]:
                    c, c0 = model.coef_[0], model.intercept_[0]
                    # logger.info ((model.coef_,model.intercept_))
                    record["c0"] = c0
                    for i in range(len(c)):
                        if xcolumns is not None:
                            record[xcolumns[i]] = c[i]
                        else:
                            record["c" + str(i + 1)] = c[i]

                if name in ["RandomForest"]:
                    record["weightlist"] = model.feature_importances_

                # 返回记录
                yield record
            else:
                yield None

        # ry1=model.predict(x_test)
        # print("test:", ry1, y_test)    


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

