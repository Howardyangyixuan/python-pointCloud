# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d
import numpy as np
import pcio as pcio


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

    return eigenvalues, eigenvectors





def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    pc_txt_path = "./test.txt"
    points = pcio.numpy_read_txt(pc_txt_path)

    # 显示原始点云
    pcio.visualize_pc(points)

    # 显示点数
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向,这里取2个
    w, v = PCA(points)
    point_cloud_vector = v[:, :2]  # 点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # PCA
    pca_encoder = np.matmul(points, point_cloud_vector)
    pca_decoder = np.matmul(pca_encoder, np.transpose(point_cloud_vector))
    # 显示PCA后结果
    pcio.visualize_pc(pca_decoder)

    # 循环计算每个点的法向量
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = []
    # 作业2
    # 屏蔽开始
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
    for i in range(points.shape[0]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 20)
        # print(idx)
        pointsnn = np.asarray(pcd.points)[idx[1:], :]
        eigenvalues, eigenvectors = PCA(pointsnn)
        normal = eigenvectors[-1]
        normals.append(normal)

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    pcio.visualize_pc(points, normals)


if __name__ == '__main__':
    main()
