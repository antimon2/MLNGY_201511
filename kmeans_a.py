# -*- coding: utf-8 -*-
import numpy as np
data = np.loadtxt('CodeIQ_data.txt', delimiter=' ')

# i番目のデータ（i≧0）を取り出す方法：
# data[i] または data[i, :]
# x成分（1列目）・y成分（2列目）をそれぞれ取り出す方法：
# data_x = data[:, 0]
# data_y = data[:, 1]


import matplotlib.pyplot as plt
# plt.scatter(data[:,0], data[:,1])


def init_centroids(X, K):
    # centroids = [] # ← numpy.ndarray でもOK
    # # X から K個 のデータをランダムに選択する処理を適切に実装してください。
    # # ヒント：Python 標準の random モジュールを利用するか、numpy.random モジュールを利用します。
    # return centroids
    randidx = np.random.choice(range(len(X)), K, replace=False)
    return X[randidx, :]


def find_closest_centroids(X, centroids):
    K = len(centroids)
    # idxs = [] # ← numpy.ndarray でもOK、適当な値で初期化するのもOK
    # for i, x in enumerate(X):
    #     pass
    #     # ここを適切に実装してください。
    #     # 方針：X の点 x が centroids のどれに『近いか』を判定し、idxs[i] にそのインデックスを格納する。
    #     # 計算方法は先述の例を参照。
    idxs = [0 for _ in X]
    for i, x in enumerate(X):
        t = x - centroids[0]
        l0 = sum(n**2 for n in t)
        for k in range(1, K):
            t = x - centroids[k]
            l = sum(n**2 for n in t)
            if l < l0:
                l0 = l
                idxs[i] = k
    
    return idxs


def compute_centroids(X, idxs, K):
    # m, n = "X の行数（＝データ数）", "X の列数（＝次元数、今回の場合 2）" # 適切に実装してください。
    # centroids = [] # ← numpy.ndarray でもOK
    # # ここを適切に実装してください。
    # # ヒント1：numpy.ndarray を使用している場合、m, n の取得は `np.shape` 関数が利用できます。
    # # ヒント2：インデックス（0〜K-1）ごとにデータを抽出して、平均を取ればOK。
    m, n = np.shape(X)
    centroids = np.array([[0.0 for _i in range(n)] for _k in range(K)])
    ws = [0.0 for _ in range(K)]
    for i in range(m):
        j = idxs[i]
        centroids[j] += X[i]
        ws[j] += 1.0

    for j in range(K):
        if ws[j] > 0:
            centroids[j] /= ws[j]

    return centroids


def run_kmeans(X, initial_centroids, max_iters=10):
    m, n = "X の行数（＝データ数）", "X の列数（＝次元数、今回の場合 2）" # 適切に実装してください。
    K = len(initial_centroids)
    centroids = initial_centroids
    prev_centroids = centroids
    idxs = []

    for _ in range(max_iters):
        idxs = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idxs, K)
        # これだけでもOKですが、必要に応じて、ここに「終了条件」を入れてもOK。
        if (centroids == prev_centroids).all():
            break

        prev_centroids = centroids

    return centroids, idxs


def disp_result(X, idxs):
    a_idxs = np.array(idxs)
    data_0 = X[a_idxs == 0, :]
    data_1 = X[a_idxs == 1, :]
    data_2 = X[a_idxs == 2, :]
    
    plt.plot(data_0[:,0], data_0[:,1], "c.")
    plt.plot(data_1[:,0], data_1[:,1], "m.")
    plt.plot(data_2[:,0], data_2[:,1], "y.")
    return plt.show()

# def disp_result(X, idxs):
#     data_0 = [v for i, v in enumerate(X) if idxs[i] == 0]
#     data_1 = [v for i, v in enumerate(X) if idxs[i] == 1]
#     data_2 = [v for i, v in enumerate(X) if idxs[i] == 2]
    
#     data_0x, data_0y = map(list, zip(*data_0))
#     data_1x, data_1y = map(list, zip(*data_1))
#     data_2x, data_2y = map(list, zip(*data_2))

#     plt.plot(data_0x, data_0y, "c.")
#     plt.plot(data_1x, data_1y, "m.")
#     plt.plot(data_2x, data_2y, "y.")
#     return plot.show()

# def disp_result(X, idxs):
#     for i, x in enumerate(X):
#         print("%.2f %.2f %d" % (x[0], x[1], idxs[i]))


def run(X, K):
    initial_centroids = init_centroids(X, K)
    centroids, idxs = run_kmeans(X, initial_centroids)
    disp_result(X, idxs)


if __name__ == "__main__":
    run(data, 3)
