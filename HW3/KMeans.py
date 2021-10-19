# 文件功能： 实现 K-Means 算法

import numpy as np
import matplotlib.pyplot as plt

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter


    def fit(self, data):
        # 作业1
        # 屏蔽开始
        iter = 0
        row_rand_array = np.arange(data.shape[0])
        np.random.shuffle(row_rand_array)
        self.clusters_center = data[row_rand_array[0: self.k_]]
        old_center = self.clusters_center.copy()
        # print(self.clusters_center)
        label = np.zeros(data.shape[0])
        while(iter <= self.max_iter_):
            iter += 1
            for i in range(data.shape[0]):
                dist = np.zeros(self.k_)
                for k in range(self.k_):
                    dist[k] = np.linalg.norm(self.clusters_center[k] - data[i])
                # print(np.where(dist == np.min(dist))[0])
                label[i] = np.where(dist == np.min(dist))[0]
            # print(label)

            dist_center = np.zeros(self.k_)
            for k in range(self.k_):
                self.clusters_center[k] = np.mean(data[label == k])
                dist_center[k] = np.linalg.norm(self.clusters_center[k] - old_center[k])
            # print(old_center)
            # print(self.clusters_center)
            # print(dist_center)
            if dist_center.all() < self.tolerance_:
                break
            else:
                old_center = self.clusters_center.copy()
        # print("origin label: ", label)
        # 屏蔽结束

    def predict(self, p_datas):
        result = np.zeros(p_datas.shape[0])
        # 作业2
        # 屏蔽开始
        dist = np.zeros(self.k_)
        for i in range(p_datas.shape[0]):
            for k in range(self.k_):
                dist[k] = np.linalg.norm(self.clusters_center[k] - p_datas[i])
            result[i] = np.where(dist == np.min(dist))[0]
        # 屏蔽结束
        return result

    def get_centroids(self):
        return self.clusters_center

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    # print(x)
    n_clusters = 2
    k_means = K_Means(n_clusters = n_clusters)
    k_means.fit(x)

    category = k_means.predict(x)

    # visualize:
    color = ['red', 'blue', 'green', 'cyan', 'magenta']
    labels = [f'Cluster{k:02d}' for k in range(n_clusters)]
    for k in range(n_clusters):
        plt.scatter(x[category == k][:, 0], x[category == k][:, 1], c=color[k], label=labels[k])

    centroids = k_means.get_centroids()
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='grey', marker='P', label='Centroids')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('KMeans Testcase')
    plt.show()

