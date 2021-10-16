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

    data = data - np.mean(data, axis=0)
    if correlation:
        H = np.corrcoef(data.T)
    else:
        H = np.cov(data.T)
    m, n = data.shape
    H = data.T.dot(data) / (m - 1)
    # print(H)
    eigenvalues, eigenvectors = np.linalg.eig(H)

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def main():
    # 指定点云路径
    cat_index = 2 # 物体编号，范围是0-39，即对应数据集中40个物体
    sample_num = 2000
    root_dir_off = '..\\3DDataset\\ModelNet40'  # 数据集路径
    cat = os.listdir(root_dir_off)
    filename_off = os.path.join(root_dir_off, cat[cat_index], 'train', cat[cat_index] + '_0001.off')

    # 加载原始点云
    mesh = o3d.io.read_triangle_mesh(filename_off)
    pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sample_num)
    o3d.visualization.draw_geometries([pcd], width=800, height=800) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = np.asarray(pcd.points)
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 0] #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector.round(2))
    # TODO: 此处只显示了点云，还没有显示PCA


    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = []
    # 作业2
    # 屏蔽开始

    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
    for i in range(points.shape[0]):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(points[i], 1) # find the points within 1
        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[i], 10) # find nearest 10 points
        # print(idx)
        data = np.asarray(pcd.points)[idx[1:], :]
        w, v = PCA(data)
        # print("normal vector of point ",i ,": ",v[:, -1].round(2))
        normals.append(v[:, -1])

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    print(normals.shape)
    # TODO: 此处把法向量存放在了normals中
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd],point_show_normal = True, width=800, height=800)


if __name__ == '__main__':
    main()
