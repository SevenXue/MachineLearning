import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3.0, 4.0, (2, 10))

x = np.hstack((cluster1, cluster2, cluster3)).T

'''
plt.scatter(x[:,0], x[:,1])
plt.xlabel('x1')
plt.ylabel('x2')

plt.show()
'''

# 测试9种不同聚类中心数量下，每种情况的聚类质量，并作图
K = range(1,10)
meanddistortions = []

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    meanddistortions.append(sum(np.min(cdist(x, kmeans.cluster_centers_, 'euclidean'), axis=1))/x.shape[0])

plt.plot(K, meanddistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')

plt.show()