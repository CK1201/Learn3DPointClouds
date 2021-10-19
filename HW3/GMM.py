# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter = 50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    
    # 屏蔽开始
    # 更新W
    

    # 更新pi

        
    # 更新Mu


    # 更新Var


    # 屏蔽结束
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始
        self.pi = np.ones(self.n_clusters) / self.n_clusters
        self.cov = np.array([np.diag(np.ones(data.shape[1])) for i in range(self.n_clusters)])
        # print(self.var)
        row_rand_array = np.arange(data.shape[0])
        np.random.shuffle(row_rand_array)
        self.mu = data[row_rand_array[0: self.n_clusters]]
        N_nk = np.zeros((self.n_clusters, data.shape[0]))
        iter = 0
        # print(self.mu)
        while iter < self.max_iter:
            iter += 1

            for k in range(self.n_clusters):
                N_nk[k] = multivariate_normal.pdf(data, mean = self.mu[k], cov = self.cov[k])
            N_nk = np.dot(np.diag(self.pi.ravel()), N_nk)
            N_nk /= np.sum(N_nk, axis=0)
            N_k = np.sum(N_nk, axis = 1)
            self.mu = np.asarray([np.dot(N_nk[k], data) / N_k[k] for k in range(self.n_clusters)])
            self.cov = np.asarray([np.dot((data - self.mu[k]).T, np.dot(np.diag(N_nk[k].ravel()), data - self.mu[k]))/N_k[k] for k in range(self.n_clusters)])
            self.pi = N_k / data.shape[0]

            # for i in range(data.shape[0]):
            #     for j in range(self.n_clusters):
            #         # print(self.var[j])
            #         # print((data[i] - self.mu[j]).reshape(data.shape[1], 1))
            #         # print(np.linalg.inv(self.var[j]))
            #         # print(np.dot((data[i] - self.mu[j]), np.linalg.inv(self.var[j])).dot((data[i] - self.mu[j]).reshape(data.shape[1], 1)))
            #         # print(np.dot((data[i] - self.mu[j]), np.linalg.inv(self.var[j])).dot(data[i] - self.mu[j]))
            #         N_nk[i, j] = 1 / (2 * math.pi) ** (data.shape[1] / 2) / np.linalg.det(self.var[j]) ** (1 / 2) * exp(-1 / 2 * np.dot((data[i] - self.mu[j]), np.linalg.inv(self.var[j])).dot((data[i] - self.mu[j]).reshape(data.shape[1], 1)))
            #         sum_N_nk[i] += N_nk[i, j]
            #         N_k[j] += N_nk[i, j]
            #     for j in range(self.n_clusters):
            #         gamma_nk[i, j] = self.pi[j] * N_nk[i, j] / sum_N_nk[i]

            # for k in range(self.n_clusters):
            #     # print(gamma_nk[:, k])
            #     self.mu[k] = 0
            #     self.var[k] = 0
            #     for i in range(data.shape[0]):
            #         self.mu[k] += 1 / N_k[k] * data[i] * gamma_nk[i, k]
            #         self.var[k] += 1 / N_k[k] * gamma_nk[i, k] * np.dot((data[i] - self.mu[k]).reshape(data.shape[1], 1), (data[i] - self.mu[k]).reshape(1, data.shape[1]))
            #         # print((data[i] - self.mu[k]).reshape(data.shape[1], 1))
            #         # print((data[i] - self.mu[k]).reshape(1, data.shape[1]))
            #         # print(np.dot((data[i] - self.mu[k]).reshape(data.shape[1], 1), (data[i] - self.mu[k]).reshape(1, data.shape[1])))
            #     self.pi[k] = N_k[k] / data.shape[0]
            # print(self.mu)
            # print(self.var)
            # print(self.pi)
        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        N_nk = np.zeros((self.n_clusters, data.shape[0]))
        for k in range(self.n_clusters):
            N_nk[k] = multivariate_normal.pdf(data, mean=self.mu[k], cov=self.cov[k])
        N_nk = np.dot(np.diag(self.pi.ravel()), N_nk)
        N_nk /= np.sum(N_nk, axis=0)
        result_set = np.argmax(N_nk, axis=0)
        return result_set
        # 屏蔽结束

    def get_mu(self):
        return self.mu

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s = 5)
    plt.scatter(X2[:, 0], X2[:, 1], s = 5)
    plt.scatter(X3[:, 0], X3[:, 1], s = 5)
    # plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)
    # print(X)
    gmm = GMM(n_clusters = 3)
    gmm.fit(X)
    # cat = gmm.predict(X)
    # print(cat)
    category = gmm.predict(X)

    # visualize:
    color = ['red', 'blue', 'green', 'cyan', 'magenta']
    labels = [f'Cluster{k:02d}' for k in range(3)]

    for k in range(3):
        plt.scatter(X[category == k][:, 0], X[category == k][:, 1], c=color[k], label=labels[k])

    mu = gmm.get_mu()
    plt.scatter(mu[:, 0], mu[:, 1], s=300, c='grey', marker='P', label='Centroids')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('GMM Testcase')
    plt.show()

    

