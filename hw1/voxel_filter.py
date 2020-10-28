# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d
import numpy as np
import pcio


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    border = []
    D = []
    index = []
    for i in range(3):
        border.append([np.max(point_cloud[:, i]), np.min(point_cloud[:, i])])
        D.append(((border[i][0] - border[i][1]) / leaf_size))

    for i in range(point_cloud.shape[0]):
        tmp = []
        for j in range(3):
            tmp.append(np.floor(point_cloud[i][j] / D[j]))
        h = int(tmp[0] + tmp[1] * leaf_size + tmp[2] * leaf_size * leaf_size)
        index.append([h, i])
    index.sort()
    mark = index[0][0]
    pcs = []
    pcs.append(index[0][1])
    for i in range(len(index)):
        if index[i][0] != mark:
            idx = np.random.randint(0, len(pcs))
            filtered_points.append(point_cloud[pcs[idx]])
            pcs = []
            mark = index[i][0]
            pcs.append(index[i][1])
        else:
            pcs.append(index[i][1])

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = "./test.txt"
    points = pcio.numpy_read_txt(file_name)
    # pcio.visualize_pc(points)

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(points, 50.0)
    # 显示滤波后的点云
    pcio.visualize_pc(filtered_cloud)


if __name__ == '__main__':
    main()
