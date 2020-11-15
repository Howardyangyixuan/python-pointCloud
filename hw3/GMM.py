# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random, math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

plt.style.use('seaborn')


class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.rznk = None
        self.pk = None
        self.Mu = None
        self.Var = None

    # 屏蔽开始
    # 更新rznk
    def update_rznk(self, data, Mu, Var, pi):
        n = data.shape[0]
        print(Mu)
        # 按类计算概率值,假设协方差矩阵为对角
        pdfs = np.zeros((n, self.n_clusters))
        for i in range(self.n_clusters):
            pdfs[:, i] = pi[i] * multivariate_normal.pdf(data, Mu[i], np.diag(Var[i]))
        rznk = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        return rznk

    # 更新pi
    # pi的和，应为1
    def update_pk(self, rznk):
        pk = rznk.sum(axis=0) / rznk.sum()
        self.pk = pk
        return pk

    # Mu和Var没有约束，随机生成
    # 更新Mu
    def update_Mu(self, data, rznk):
        Mu = np.zeros((self.n_clusters, data.shape[1]))
        for i in range(self.n_clusters):
            Mu[i] = np.average(data, axis=0, weights=rznk[:, i])
        self.Mu = Mu
        return Mu

    # 更新Var
    def update_Var(self, data, Mu, rznk):
        Var = np.zeros((self.n_clusters, data.shape[1]))
        for i in range(self.n_clusters):
            Var[i] = np.average((data - Mu[i]) ** 2, axis=0, weights=rznk[:, i])
        self.Var = Var
        return Var

    # 屏蔽结束

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        # 1.初始化
        pk = np.random.rand(self.n_clusters)
        pk_sum = 0.0
        for i in range(self.n_clusters):
            pk_sum += pk[i]
        pk /= pk_sum
        self.pk = pk
        Mu = np.random.rand(self.n_clusters, data.shape[1])
        self.Mu = Mu
        Var = np.random.rand(self.n_clusters, data.shape[1])
        self.pk = pk
        # 2.迭代
        for iter in range(self.max_iter):
            # 2.1 E step
            rznk = self.update_rznk(data, pk, self.Mu, Var)
            # 2.2 M step
            self.update_Mu(data,rznk)
            self.update_Var(data,Mu,rznk)
            self.update_pk(rznk)


    # 屏蔽结束

    def predict(self, data):
        result = []
        rznk = self.update_rznk(data, self.pk, self.Mu, self.Var)
        print("在各类的概率：")
        print(rznk)
        idx = np.argmax(rznk)
        result.append(idx)
# 屏蔽开始

# 屏蔽结束


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化
