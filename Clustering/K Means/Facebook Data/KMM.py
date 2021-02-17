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
plt.scatter(data1['Comment'], data1['Like'], data1['Share'],data1['Interactions'])



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Using elbow method to a dataset
from sklearn.cluster import KMeans
list = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i , random_state = 1)
    kmeans.fit(X)
    list.append(kmeans.inertia_)
plt.plot(range(1,11), list,marker ="o")
plt.title('The Elbow method')
plt.xlabel('Number of cluster')
plt.ylabel('Within cluster distance')

#Fitting K-means to dataset
kmeans = KMeans(n_clusters = 5 , random_state = 1)
y_kmeans = kmeans.fit_predict(X)

data['kmeans'] = y_kmeans
data.replace({'kmeans' : { 0 : 'Red' , 1 : 'Orange' , 2 : 'Blue' , 3 : 'Green' , 4 : 'Purple'}},inplace = True)
data['kmeans'].value_counts()

#Visualizing the clusters
plt.scatter(X[y_kmeans == 0 , 0] , X[y_kmeans == 0 , 1], s = 100 , c = 'Red' , label = 'Cluster1')
plt.scatter(X[y_kmeans == 1 , 0] , X[y_kmeans == 1 , 1], s = 100 , c = 'Orange' , label = 'Cluster2')
plt.scatter(X[y_kmeans == 2 , 0] , X[y_kmeans == 2 , 1], s = 100 , c = 'Blue' , label = 'Cluster3')
plt.scatter(X[y_kmeans == 3 , 0] , X[y_kmeans == 3 , 1], s = 100 , c = 'Green' , label = 'Cluster4')
plt.scatter(X[y_kmeans == 4 , 0] , X[y_kmeans == 4 , 1], s = 100 , c = 'Purple' , label = 'Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0] , kmeans.cluster_centers_[: , 1] , s = 300 ,c ='yellow' , label ='centroids')
plt.legend()
