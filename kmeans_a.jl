using Gadfly

# データ読込
data = open(readdlm, "./CodeIQ_data.txt", "r")

K = 3
X = data;

# クラスタ重心初期化
function initCentroids(X, K)
    randidx = randperm(size(X, 1))
    X[randidx[1:K], :]
end

# クラスタ割り付けステップ
function findClosestCentroids(X, centroids)
    K = size(centroids, 1)
    map(1:size(X, 1)) do i
        x = vec(X[i, :])
        ts = [x - vec(centroids[j,:]) for j=1:K]
        ls = [dot(t, t) for t=ts]
        indmin(ls)
    end
end

# 重心移動ステップ
function computeCentroids(X, idxs, K)
    m, n = size(X)  # m 不使用
    centroids = zeros(K, n)
    for j = 1:K
        centroids[j, :] = mean(X[idxs .== j, :], 1) # ← idxs が j（=1,2,…,K）の行のみ抽出して、列ごとに平均値を算出
    end
    centroids
end

# ループ実施
function runKMeans(X, initialCentroids, max_iters=10)
    m, n = size(X)
    K = size(initialCentroids, 1)
    centroids = initialCentroids
    prev_centroids = centroids
    idxs = zeros(m, 1)

    for i = 1:max_iters
        idxs = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idxs, K)
        if centroids == prev_centroids
            break
        end
        prev_centroids = centroids
    end

    return centroids, idxs
end

# 結果表示
function dispResult(X, idxs)
    plotdata = [X [string(i) for i=idxs];]
    plot(x=plotdata[:,1], y=plotdata[:,2], color=plotdata[:,3])
end

# 実行
function main()
    initial_centroids = initCentroids(X, K)
    centroids, idxs = runKMeans(X, initial_centroids)
    dispResult(X, idxs)
end

main()
