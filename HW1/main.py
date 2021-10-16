import open3d as o3d
import numpy as np
import os

def listdir(path, list_name, dir_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name, dir_name)
        elif os.path.splitext(file_path)[1]=='.off':
            dir_name.append(os.path.split(file_path)[0])
            list_name.append(file_path)

def off2pcd(path_off, sample_num):
    list_name = []
    dir_name = []
    listdir(path_off, list_name, dir_name)
    for dir in dir_name:
        dir = dir.replace('ModelNet40', 'ModelNet40_PCD')
        folder = os.path.exists(dir)
        if not folder:
            os.makedirs(dir)
    for file in list_name:
        file_new = file.replace('ModelNet40', 'ModelNet40_PCD')
        file_new = file_new.replace('off', 'pcd')
        # file_new = file_new.replace("\\", '/')
        # print(file)
        print(file_new)
        mesh = o3d.io.read_triangle_mesh(file)
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points = sample_num)
        o3d.io.write_point_cloud(file_new, pcd)

def PCA(data):
    m, n = data.shape
    num = 100
    # data -= data.mean(axis = 1)
    # print(data.shape)
    # H = np.dot(data.T, data) / (m - 1)
    # eigenvalue, eigenvector = np.linalg.eig(H)
    # print(eigenvector.shape)
    # data_pca = np.zeros((m, n))
    # data_pca[:, 0: 3] = np.dot(data, eigenvector[:, 0: 3].reshape(3, 3))
    # data_pca = np.dot(data, eigenvector)


    data -= data.mean(axis=1).reshape(m, 1)
    H = np.dot(data, data.T) / (m - 1)
    # print(H.shape)
    eigenvalue, eigenvector = np.linalg.eig(H)
    temp = eigenvector[:, 0: num].sum(axis = 1)
    print(temp.shape)
    data_pca = np.zeros((m, n))
    for i in range(n):
    #     print(i)
        data_pca[:, i] = data[:, i] * temp;
    return data_pca


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "..\\3DDataset\\ModelNet40\\"
    sample_num = 1000

    # file = "sofa"
    # mesh = o3d.io.read_triangle_mesh(os.path.join(path, file, "train", file + "_0001.off"))  # 加载mesh
    # pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sample_num)
    # o3d.visualization.draw_geometries([pcd], width=800, height=800)

    data = np.zeros((sample_num * 3, 1))
    for file in os.listdir(path):
        # print(file)
        mesh = o3d.io.read_triangle_mesh(os.path.join(path, file, "train", file + "_0003.off"))
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points = sample_num)
        # o3d.visualization.draw_geometries([pcd], width=800, height=800)
        temp = np.asarray(pcd.points).reshape(sample_num * 3, 1)
        data = np.concatenate([data, temp], axis = 1)
        # print(data[:10, :])
        # data_pca = PCA(data)
        # print(data_pca[:5, :])
        # pcd_pca = o3d.geometry.PointCloud()
        # pcd_pca.points = o3d.utility.Vector3dVector(data_pca)
        # o3d.visualization.draw_geometries([pcd_pca], point_show_normal=True, width=800, height=800)
        # break

        # pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    data = data[:, 1:]
    # print(data.shape)
    data_pca = PCA(data)
    # print(data_pca.shape)
    pcd_pca = o3d.geometry.PointCloud()
    data_new = np.zeros((1000, 3))
    i = 0
    for file in os.listdir(path):
        print(file)
        data_new[:, 0] = data_pca[0: sample_num, i]
        data_new[:, 1] = data_pca[sample_num: sample_num * 2, i]
        data_new[:, 2] = data_pca[sample_num * 2: sample_num * 3, i]
        pcd_pca.points = o3d.utility.Vector3dVector(data_new)
        o3d.visualization.draw_geometries([pcd_pca], point_show_normal=True, width=800, height=800)
        i += 1

