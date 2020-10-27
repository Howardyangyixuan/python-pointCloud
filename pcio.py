import numpy as np
import open3d as o3d


def numpy_read_txt(pc_txt_path):
    data = np.genfromtxt(pc_txt_path, delimiter=',')
    # 原数据中,有法向量,只取前3个
    pc = data[:, :3]
    return pc


def visualize_pc(pc, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd])
