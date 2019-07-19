# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ----- 对处理好的purchaseRecord表进行回归类选择策略 
"""
import numpy as np
from com.bc.data.data_repeatData import repeatData
from com.bc.base.public import logger, public_basePath, baseSql, public, config_ini
# from sklearn.linear_model import LinearRegression,LogisticRegression
# from sklearn.svm import SVC,SVR
# from sklearn import tree
# from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
# from sklearn.ensemble import ExtraTreesClassifier
from multiprocessing import Pool, Process
from sklearn.decomposition import PCA  # 主成分分析
import time
import json

class modelResult(baseSql):
    tablename = "t_modelResult"
    keylist = up_struct = ["model", "keys", "score", "means", "upwin_0", "upwin_r", "keyNum", "createDate"]

# 回归方法
class Stage_regression():
    """
          回归模型类：数据来源 
          参数:
             x_column: x变量列表
             y_column: y目标函数
             show:  数据源是否需要图形化和csv输出
             isArea: x变量是否区间化处理
             isGroup: y变量是否采用Group处理   
    """
    # 模型清单
    modelMap = {
        # "Linear": LinearRegression(),
        # "Logistic":LogisticRegression(),
        # "svc":SVC(),
        # "svr":SVR(),
        # "DecisionTree":tree.DecisionTreeRegressor(),
        # "KNeighbors": KNeighborsRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=40),
        # "AdaBoost":AdaBoostRegressor(n_estimators=50),
        # "Gradien":GradientBoostingRegressor(n_estimators=100)
    }
    x_fieldIndex = ["oddDiff", "interval", "start", "step", "sclassRange", "sclassScope", "purDiff", "purOddRange",
                    "oddFirstDiff"]
    data, x_column, y_column, modelName = None, None, None, None
    time0, modelResult = None, None

    def __init__(self, x_column=["oddDiff", "interval", "purOddRange", "oddFirstDiff"],
                 y_column=["upwin"],
                 show=False, isArea=True, isGroup=True):
        # 初始化数据源
        self.data = repeatData(xlist=x_column,
                               ylist=y_column,
                               show=show, isArea=isArea, isGroup=isGroup)

        self.x_column, self.y_column = x_column, y_column
        self.time0 = time.time()
        # self.modelResult= modelResult()

    # 参数选择
    np_ary = None
    process = 0

    def train(self, process, ary, xc, times, save=False):
        self.x_column, self.process = xc, process
        xi = [self.data.x_fieldIndex.index(item) for item in xc]
        # 单个组训练
        if not ary:
            ary = self.data.get()
            xi = [k for k in range(len(xc))]

        # ary = self.check(ary)
        self.np_ary = ary
        x, y = self.mscale(ary[:, xi]), ary[:, ary.shape[1] - 1]
        # x,y=ary[:,xi],ary[:,ary.shape[1]-1]
        self.applyModel(x, y, times, save)

    def applyModel(self, x, y, times, save):
        if not self.modelResult:
            self.modelResult = modelResult()
        kk = 0
        for i in range(times):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)
            for key in self.modelMap.keys():
                self.modelName, time1 = key, time.time()
                model = self.modelMap[key]
                model.fit(x_train, y_train)
                self.test(model, x_test, y_test)
                if save:
                    self.mapSave(i, model, self.np_ary)
                logger.info(["进程:%s_%s" % (self.process, i), "参数:", self.x_column, "model:", key,
                             "start", time.time() - self.time0, "interface:", time.time() - time1])
                kk += 1

    def test(self, model, x_test, y_test):
        keys = self.x_column
        # 检测函数1，score
        p0 = model.score(x_test, y_test)

        # 检测函数 - 允许0.0001误差
        py = model.predict(x_test)
        p1 = np.mean(abs(py - y_test) < 0.001)
        # logger.info(["%s_mean" % self.modelName,keys,p1])

        print(py[0:100], y_test[0:100])

        # 计算调整后的经胜率
        r0 = np.mean(y_test)
        ry = np.round(py * 1.01, 0) * y_test
        # print(np.where(np.isnan(ry)))

        r1 = (np.mean(ry) + np.mean(ry[ry != 0])) / 2.0
        if np.isnan(r1): r1 = 0.0

        # 写入数据库
        doc = self.modelResult.set(
                [self.modelName, "/".join(keys), p0, p1, r0, r1, len(keys), public.getDatetime()[0:-3]])
        print(doc)
        self.modelResult.insert(doc)

        # 输出
        # self.mat(p0,np.vstack((np.arange(py.shape[0]),py,y_test)).T)
        return p0, p1

    modelfile = ""

    def mapSave(self, which, model, np_ary):
        self.modelfile = public_basePath + "/csv/model/%s_%s.pkl" % (self.modelName, which)
        self.modelsave(model)
        # 生成max_min文件
        area_map = {}
        for i in range(len(self.x_column)):
            key = self.x_column[i]
            area_map[key] = {}
            area_map[key]["min"] = np.min(np.abs(np_ary[:, i]))
            area_map[key]["max"] = np.max(np.abs(np_ary[:, i]))
        self.jsonsave(area_map)

    def predict(self, ary, model):

        pass

    def optimise(self):
        pass

    def pca(self):
        # 主成分分析法降维
        ary = self.data.get()
        dim = len(self.x_column)
        x, y = self.mscale(ary[:, 0:dim]), ary[:, dim]
        # 协方差
        # c=np.cov(x,y)
        # print(c)
        # 主成分
        pca = PCA(n_components=5)
        pca.fit(x)
        print(pca.explained_variance_ratio_)
        xt = pca.fit_transform(x)
        self.applyModel(xt, y, 5, False)


    def mat(self, r, np_ary):
        keys, mat = self.x_column, self.data.matPlot
        self.data.write(np_ary)

        if np_ary.shape[0] > 1000: return None
        # 输出图形
        xl = "/".join(keys)
        mat.x_column, mat.y_column = [xl], ["upwin-result", "test"]
        mat.keys = mat.x_column + mat.y_column
        mat.filename = "%s模型_%s_%s.png" % (self.modelName, np_ary.shape[0], r)
        mat.title = "%s_模型: %s , %s" % (self.modelName, np_ary.shape[0], r)
        mat.draw(np_ary=np_ary)

    def mscale(self, y):
        mscale = preprocessing.MaxAbsScaler()
        mscale.fit(y)
        print(mscale.max_abs_)
        return mscale.transform(y)

    def check(self, ary):

        return ary

    def modelsave(self, model):
        joblib.dump(model, self.modelfile)

    def modelload(self):
        return joblib.load(self.modelfile)

    def jsonsave(self, doc):
        filename = self.modelfile.replace(".pkl", ".txt")
        json.dump(doc, open(filename, "w"))

    def jsonload(self):
        filename = self.modelfile.replace(".pkl", ".txt")
        return json.load(open(filename, 'r'))


import itertools


class action_regression:
    @staticmethod
    def exec_creat():
        # x_fieldIndex=["purOddRange","oddDiff","oddFirstDiff","sclassScope","interval"]
        x_fieldIndex = eval(config_ini.get("fellow.trainkeys", "stage"))
        # print(x_fieldIndex)
        stage = Stage_regression(x_column=x_fieldIndex,
                                 y_column=["upwin"],
                                 show=False, isArea=False, isGroup=False)
        # ary = stage.data.get()
        stage.train(1, None, x_fieldIndex, 5, save=True)

    @staticmethod
    def pool_train(times=3, scope=(1, 6), poolSize=4):
        print("启动多进程mulPool_stat..scope: %s_%s..,poolSize: %s -----" % (scope[0], scope[1], poolSize))
        x_fieldIndex = ["oddDiff", "interval", "start", "step", "sclassRange", "sclassScope", "purDiff", "purOddRange",
                        "oddFirstDiff"]
        stage = Stage_regression(x_column=x_fieldIndex,
                                 y_column=["upwin"],
                                 show=False, isArea=False, isGroup=False)
        ary = stage.data.get()
        p = 0
        pool = Pool(processes=poolSize)
        for k in range(scope[0], scope[1]):
            for xc in list(itertools.combinations(x_fieldIndex, k)):
                if len(xc) == 1: xc = [xc[0]]
                result = pool.apply_async(stage.train, (p, ary, xc, times))
                # stage.train(p,ary,xc,times)
                p += 1
        pool.close()
        pool.join()
        print(result.successful())

    @staticmethod
    def process(times=3, scope=(1, 6), poolSize=4):
        x_fieldIndex = ["oddDiff", "interval", "start", "step", "sclassRange", "sclassScope", "purDiff", "purOddRange",
                        "oddFirstDiff"]
        stage = Stage_regression(x_column=x_fieldIndex,
                                 y_column=["upwin"],
                                 show=False, isArea=False, isGroup=False)
        ary = stage.data.get()
        stage.modelResult = modelResult()
        p_list = []
        i = 0
        for k in range(scope[0], scope[1]):
            for xc in list(itertools.combinations(x_fieldIndex, k)):
                if len(xc) == 1: xc = [xc[0]]
                # result = pool.apply_async(stage.train,(p,ary,xc,times))
                # print(xc)
                p = Process(target=stage.train, args=(i, ary, xc, times))
                # stage.train(p,ary,xc,times)
                p_list.append(p)
                p.start()
                i += 1

        for j in p_list:
            j.join()

    @staticmethod
    def exec_pca():
        # 主成分分析法降维
        xc = ["purOddRange", "oddDiff", "oddFirstDiff", "sclassScope", "interval"]
        stage = Stage_regression(x_column=xc,
                                 y_column=["upwin"],
                                 show=False, isArea=False, isGroup=False)
        stage.pca()


def main():
    sel = [1, 0, 0, 0]
    if sel[0] == 1:
        action_regression.exec_creat()
    if sel[1] == 1:
        s, e, p = (public.getCmdParam("s:e:p:", key) for key in [("-s", 5), ("-e", 8), ("-p", 4)])
        action_regression.pool_train(times=5, scope=(s, e), poolSize=p)
    if sel[2] == 1:
        s, e = (public.getCmdParam("s:e:", key) for key in [("-s", 1), ("-e", 9)])
        action_regression.process(times=5, scope=(s, e), poolSize=12)
    if sel[3] == 1:
        action_regression.exec_pca()


if __name__ == '__main__':
    main()
