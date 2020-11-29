# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def PCA(data, sort=True):
    # 1.标准化
    x_avg = np.mean(data, axis=0)
    x = data - x_avg
    # 2.求协方差矩阵特征值特征向量
    xt = np.transpose(x)
    h = np.matmul(xt, x)
    eigenvalues, eigenvectors = np.linalg.eig(h)
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvectors


# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    # RANSAC进行地面提取
    # 1.确定参数 r e s p N t
    # r: inlier radio = r
    # e: outlier ratio = 1 - r
    # s: number of sample
    # t: 阈值
    r = 0.5
    e = 1 - r
    s = data.shape[1]
    # 指定置信率p以计算迭代次数N ~ 34
    p = 0.99
    N = int(np.log(1 - p) / np.log(1 - pow((1 - e), s)))
    t = 0.2
    # 2.开始算法
    # 可视化确定平面是否正确
    ax = plt.figure().add_subplot(111, projection='3d')
    for n in range(1):
        # 2.1随机选取1个sample（3个点）对平面进行拟合
        idx = np.random.randint(0, data.shape[0], 3)
        sample = []
        for i in range(data.shape[1]):
            sample.append(data[idx[i]])
            # 绘制点
            ax.scatter(data[idx[i]][0], data[idx[i]][1], data[idx[i]][2], s=100, color='#4daf4a')
        # 获取主方向，即所求平面
        vectors = PCA(np.array(sample))[:, :2]
        # 绘制平面
        leftx = min(data[:, 0])
        rightx = max(data[:, 0])
        lefty = min(data[:, 1])
        righty = max(data[:, 1])
        X = np.arange(leftx, rightx, 5)
        Y = np.arange(lefty, righty, 5)
        X, Y = np.meshgrid(X, Y)
        a1 = vectors[1][0] * vectors[2][1] - vectors[1][1] * vectors[2][0]
        a2 = vectors[0][1] * vectors[2][0] - vectors[0][0] * vectors[2][1]
        b = vectors[0][0] * vectors[1][1] - vectors[0][1] * vectors[1][0]
        Z = a1 * X + a2 * Y + b
        ax.scatter(X, Y, Z, s=2, color='#ff7f00', alpha=0.2)
        # 原始点云
        # ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color='#377eb8', alpha=0.3)
        plt.show()
    # 计算所有点到平面距离，根据t确定该模型对应的内点数和内点率
    # 当内点率大于r或迭代次数大于N时，结束迭代，否则重复2 - 3
    # 内点数最多的模型即为所求

    # 屏蔽结束

    # print('origin data points num:', data.shape[0])
    # print('segmented data points num:', segmengted_cloud.shape[0])
    # return segmengted_cloud


# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始

    # 屏蔽结束

    return clusters_index


# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection='3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()


def main():
    root_dir = '../kitti'  # 数据集路径
    cat = os.listdir(root_dir)
    print(cat)
    cat = cat[0:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        # plot_clusters(origin_points, np.zeros(origin_points.shape[0], dtype=int))
        ground_segmentation(data=origin_points)
        # segmented_points = ground_segmentation(data=origin_points)
        # cluster_index = clustering(segmented_points)

        # plot_clusters(segmented_points, cluster_index)


if __name__ == '__main__':
    main()
