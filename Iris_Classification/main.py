from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

n_neigh = 7
weights = 'uniform'

iris = datasets.load_iris()

X = iris.data[:,:2]
y = iris.target

clf = neighbors.KNeighborsClassifier(n_neigh, weights=weights)
clf.fit(X,y) #x = dados, y = supervisão

#print(clf.predict([[3.3,1]]))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 1/3, random_state= 42)
clf2 = neighbors.KNeighborsClassifier(n_neigh, weights=weights)
clf2.fit(X_train, y_train)

print(accuracy_score(y_test, clf2.predict(X_test))*100)
print(clf2.predict([[6.2,3.4]]))

plt.figure(figsize=(10,8))

sns.scatterplot(x=X[:,0], y=X[:,1], hue=(iris.target_names[y]), palette=(["orange", "cyan", "blue"]), edgecolor="black")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[0])
plt.show()
