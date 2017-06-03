from sklearn.cluster import KMeans
import numpy as np
def kmeans(cluster,input):
    kmeans = KMeans(n_clusters=cluster, random_state=0).fit(input)
    return kmeans.cluster_centers_


#test1
input_array1 = np.array([[1,2,3,4,5],[11,2,3,3,4],[1,3,4,5,31],[11,2,3,4,5]])
res = kmeans(2,input_array1)
print(res)

#test2 failed
input_array2=np.array([np.matrix([[1, 2], [3, 4]]),np.matrix([[1, 2], [3, 4]]),np.matrix([[1, 2], [3, 4]]),np.matrix([[1, 2], [3, 4]]),np.matrix([[1, 2], [3, 4]])])
res = kmeans(2,input_array2)
print(res)
