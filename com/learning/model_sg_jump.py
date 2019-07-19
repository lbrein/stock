# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
 ----- 对处理好的purchaseRecord表进行回归类选择策略 
"""
import numpy as np

from com.base.public import public, public_basePath
from sklearn import preprocessing

from sklearn.externals import joblib

from sklearn.decomposition import PCA  # 主成分分析
import time
import json
from com.learning.model_simple import modelSimple
from com.object.obj_model import future_train_source, future_model

# 回归方法
class model_sg_jump():
    """
          回归模型类：数据来源 
          参数:
             x_column: x变量列表
             y_column: y目标函数
             show:  数据源是否需要图形化和csv输出
             isArea: x变量是否区间化处理
             isGroup: y变量是否采用Group处理   
    """

    def __init__(self):
        # 初始化数据源
        self.Data = future_train_source()

        self.modelResult = future_model()
        # ['mode', 'diff', 'bias', 'jump_n', 'isup', 'isbig', 'iscross', 'isatr', 'income']

        self.x_fieldIndex = self.Data.keylist
        self.x0, self.x1 = 1, -1
        self.x_column = self.x_fieldIndex[self.x0:self.x1]
        self.y_column = ['income']
        self.modelName = 'sg'

    # 参数选择
    np_ary = None
    process = 0

    def start(self):
        df = self.Data.getdata()

        ary = np.array(df)
        #dim = len(self.x_column) + self.x0
        #x, y =  ary[:, self.x0: dim],    ary[:, dim:]
        x, y = self.mscale(ary[:, self.x0: self.x1]), self.mscale(ary[:, -1:])
        print(self.x_column)

        model = modelSimple()
        for res in model.train([], x, y, mode='all'):
            print(res)

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

    def pca(self, ary):
        # 主成分分析法降维
        dim = len(self.x_column)
        x, y = self.mscale(ary[:, 0:dim]), self.mscale(ary[:, dim])

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


def main():
    obj = model_sg_jump()
    obj.start()

if __name__ == '__main__':
    main()
