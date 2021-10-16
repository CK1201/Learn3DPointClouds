# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
import math
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, random_select = True):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    data = np.asarray(point_cloud)
    # print(data.shape)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    # print(data_min)
    # print(data_max)
    D = np.floor((data_max - data_min) / leaf_size)
    h = np.floor((data - data_min) / leaf_size)
    # print(h)
    H = h[:, 0] + h[:, 1] * D[0] + h[:, 2] * D[0] * D[1]
    # print(hh)
    idx = np.argsort(H)
    H_sorted = np.sort(H)
    # print(H_sorted)
    # print(idx)
    if(random_select):
        last = -1
    else:
        last = H_sorted[0]
    start_idx = 0
    for i in range(idx.size):
        if(random_select):
            if(H_sorted[i] != last):
                last = H_sorted[i]
                filtered_points.append(data[idx[i],:])
        else:
            if (H_sorted[i] != last):
                last = H_sorted[i]
                # print(data[idx[start_idx: i], :].mean(axis=0))
                filtered_points.append(data[idx[start_idx: i], :].mean(axis=0))
                start_idx = i

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    cat_index = 30 # 物体编号，范围是0-39，即对应数据集中40个物体
    sample_num = 2000
    root_dir_off = '..\\3DDataset\\ModelNet40'  # 数据集路径
    cat = os.listdir(root_dir_off)
    filename_off = os.path.join(root_dir_off, cat[cat_index], 'train', cat[cat_index] + '_0001.off')
    mesh = o3d.io.read_triangle_mesh(filename_off)
    pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sample_num)
    # o3d.visualization.draw_geometries([pcd], width=800, height=800)  # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(pcd.points, 10.0, random_select=False)
    # print(filtered_cloud)
    pcd.points = o3d.utility.Vector3dVector(filtered_cloud)
    # print(pcd)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([pcd], width=800, height=800)

if __name__ == '__main__':
    main()
