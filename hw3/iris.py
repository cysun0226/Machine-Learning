import sys
import math
import numpy as np
from numpy import array
from sklearn import tree
from random import shuffle
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

iris_file = str(sys.argv[1])
# test_file = str(sys.argv[2])
train_file = open(iris_file, 'r', encoding='UTF-8')

# main
target_names = ['setosa', 'versicolor', 'virginica']
target_names = np.array(target_names)

feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
feature_names = np.array(feature_names)

lines = []
target = []
data = []
iris = {}

# read iris.data
for line in train_file.readlines():
    if line=='\n':
        continue
    lines.append(line)
# shuffle and pick the training set
shuffle(lines)
train_set = lines[0:math.floor(0.7*len(lines))]
test_set = lines[math.floor(0.7*len(lines))+1:len(lines)]

for line in train_set:
    row = line.strip('\n')
    row = row.split(',')
    iClass = row.pop()
    row = list(map(float, row))
    row = array(row)
    data.append(row)
    target.append(iClass)
    # if iClass=='Iris-setosa':
    #     target.append('Iris-setosa')
    # elif iClass=='Iris-versicolor':
    #     target.append('Iris-versicolor')
    # else: # Iris-virginica
    #     target.append('Iris-virginica')

iris['target_names'] = array(target_names)
iris['target'] = array(target)
iris['feature_names'] = array(feature_names)
iris['data'] = array(data)

# build decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris['data'], iris['target'])

# test
total = len(test_set)
correct = 0

print('test_set = %d' % total)

for line in test_set:
    row = line.strip('\n')
    row = row.split(',')
    iClass = row.pop()
    row = list(map(float, row))
    iPredict = clf.predict([array(row)])

    if iClass==iPredict[0]:
        correct += 1
        print('[correct] iClass = ' + iClass + ' | Predict = ' + iPredict[0])
    else:
        print('[ wrong ] iClass = ' + iClass + ' | Predict = ' + iPredict[0])

acc = correct / total
print('decision tree accuracy = %f' % acc)

# KD-tree
knn = KNeighborsClassifier(n_neighbors=10, algorithm="kd_tree").fit(iris['data'],iris['target'])
print(iris['target'][0])
print(knn.predict_proba([iris['data'][0]]))
print(iris['target'][1])
print(knn.predict_proba([iris['data'][1]]))
# nbrs = NearestNeighbors(n_neighbors=5, algorithm="kd_tree").fit(iris['data'])
# kdt = KDTree(iris['data'], metric='euclidean')
# ind = kdt.query([iris['data'][0]], k=5, return_distance=False)
# print(ind)

#
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# >>> kdt = KDTree(X, leaf_size=30, metric='euclidean')
