# 文件功能： 实现 K-Means 算法

import numpy as np


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.mk = None

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        # 1.初始化k个中心点
        dimension = data.shape[1]
        n = data.shape[0]
        mk = np.zeros((self.k_, dimension))
        for i in range(self.k_):
            # 实际使用发现，随机生成很容易所有点都在一类，因此使用任选一点
            random = np.random.randint(n)
            mk[i] = data[random]
            # mk[i] = np.random.randn(1, dimension) + 3

        # 2.开始迭代
        for iter in range(self.max_iter_):
            # 每次迭代开始，重置rnk,记录上一次的误差，用于提前结束
            rnk = np.zeros((data.shape[0], self.k_), dtype=int)
            mk_last = mk
            # 2.1 E step
            # 每一个点求解离它最近的mk，将对应的rnk置为1
            print("E step:")
            for i, point in enumerate(data):
                means = []
                for k in range(self.k_):
                    cost = np.linalg.norm(point - mk[k])
                    means.append(cost)
                idx = np.argmin(means)
                rnk[i][idx] = 1
            # 2.2 M step
            print("M step:")
            # 计算新的mk,sum存点的和
            for k in range(self.k_):
                point_sum = np.zeros((1, dimension), dtype=np.float32)
                cnt = 0
                for n in range(data.shape[0]):
                    point_sum += data[n] * rnk[n][k]
                    cnt += rnk[n][k]
                mk[k] = point_sum / cnt

            # 2.3 打印 mk
            print("mk:")
            print(mk)
            self.mk = mk
            if np.linalg.norm(mk - mk_last) <= self.tolerance_:
                print("Finish: Converged. less change. ")
                break

        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        for i in range(self.k_):
            print("%d th catogery is at " % i)
            print(self.mk[i])
        print("predict(E step):")
        for i, point in enumerate(p_datas):
            means = []
            for k in range(self.k_):
                cost = np.linalg.norm(point - self.mk[k])
                means.append(cost)
            idx = np.argmin(means)
            print("Point %d" % i, " belongs to %d" % idx, "th catogery")
            result.append(idx)
        # 屏蔽结束
        return result


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)
