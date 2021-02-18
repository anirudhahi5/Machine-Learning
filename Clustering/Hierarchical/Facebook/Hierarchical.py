#KMeans
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"Facebook_data.csv")

data.isnull().sum()

# Fill in with most frequent category ( Mode )
data['Share'].fillna(data['Share'].mode()[0], inplace = True)
data['Interactions'].fillna(data['Interactions'].mode()[0], inplace = True)

data1 = data[['Comment', 'Like', 'Share', 'Interactions']]

X = data1.values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Using Dendogram to find the optimum number of cluster
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()
#From graph  we can go for number of clusters = 5

#Fitting Hierarchial Clustering to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5 , affinity = 'euclidean' , linkage = 'ward') # Ward's minimizes sum-within cluster
y_hc = hc.fit_predict(X)

data['h_clust'] = y_hc
data.replace({'h_clust' : { 0 : 'Red' , 1 : 'Orange' , 2 : 'Blue' , 3 : 'Green' , 4 : 'Purple'}},inplace = True)
data['h_clust'].value_counts()

#Visualizing the clusters
plt.scatter(X[y_hc == 0 , 0] , X[y_hc == 0 , 1], s = 100 , c = 'Red' , label = 'Cluster1')
plt.scatter(X[y_hc == 1 , 0] , X[y_hc == 1 , 1], s = 100 , c = 'Orange' , label = 'Cluster2')
plt.scatter(X[y_hc == 2 , 0] , X[y_hc == 2 , 1], s = 100 , c = 'Blue' , label = 'Cluster3')
plt.scatter(X[y_hc == 3 , 0] , X[y_hc == 3 , 1], s = 100 , c = 'Green' , label = 'Cluster4')
plt.scatter(X[y_hc== 4 , 0] , X[y_hc == 4 , 1], s = 100 , c = 'Purple' , label = 'Cluster5')

plt.legend()