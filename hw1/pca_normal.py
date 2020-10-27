# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d
import os
import numpy as np


# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def numpy_read_txt(pc_txt_path):
    data = np.genfromtxt(pc_txt_path, delimiter=',')
    # 原数据中,有法向量,只取前3个
    pc = data[:, :3]
    return pc


def visualize_pc(pc, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if normals:
        pcd.normals = normals
    o3d.visualization.draw_geometries([pcd])


def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    pc_txt_path = "./test.txt"
    points = numpy_read_txt(pc_txt_path)

    # 显示原始点云
    visualize_pc(points)

    # 显示点数
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 2]  # 点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    # o3d.visualization.draw_geometries([point_cloud_o3d])

    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(points)
    normals = []
    # 作业2
    # 屏蔽开始

    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    visualize_pc(points,normals)


if __name__ == '__main__':
    main()
