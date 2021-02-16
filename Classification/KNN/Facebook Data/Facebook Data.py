import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"Facebook_data.csv")

data.isnull().sum()

# Fill in with most frequent category ( Mode )
data['Share'].fillna(data['Share'].mode()[0], inplace = True)
data['Interactions'].fillna(data['Interactions'].mode()[0], inplace = True)




data1 = data.iloc[: , 1: -1]


X = data1.values
y = data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y,test_size = 0.25 , random_state = 10)

# Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train , y_train)
y_pred = knn.predict(X_test)

print('Training Accuracy : {:3f}'.format(knn.score(X_train, y_train)))
print('Testing Accuracy : {:3f}'.format(knn.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred, labels = ("Photo","Status"))
print(cm)

#Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(knn,X,y,cv=10)                 # data is not scaled
print('{:3f}'.format(accuracies.mean()))

#PIpeline
from sklearn.pipeline import make_pipeline
clf = make_pipeline(sc , knn)
accuracies = cross_val_score(clf,X,y,cv = 10)
print('{:3f}'.format(accuracies.mean()))

#Elbow methid to get optimum value of k

neighbors = range(1,20)
k_score = []
for n in neighbors:
    knn1 = KNeighborsClassifier(n_neighbors = n)
    clf1 = make_pipeline(sc, knn1)
    accuracies1 = cross_val_score(clf1,X,y, cv = 10)
    k_score.append(1 - accuracies1.mean())
    print('{:3f}'.format(1 - accuracies1.mean()))
    
plt.plot(neighbors,k_score)
plt.ylabel('Error')
plt.xlabel('Number of Neighbors')

#KNN model
from sklearn.neighbors import KNeighborsClassifier
knn1 = KNeighborsClassifier(n_neighbors = 4)
clf1 = make_pipeline(sc , knn1)
accuracies1 = cross_val_score(clf1,X,y, cv = 10)
print('{:3f}'.format( accuracies1.mean()))