import sys
import math
import numpy as np
from numpy import array
from sklearn import tree
from random import shuffle
# from sklearn.neighbors import KDTree
# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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
test_set = lines[math.floor(0.7*len(lines)):len(lines)]

for line in train_set:
    row = line.strip('\n')
    row = row.split(',')
    iClass = row.pop()
    row = list(map(float, row))
    row = array(row)
    data.append(row)
    target.append(iClass)

iris['target_names'] = array(target_names)
iris['target'] = array(target)
iris['feature_names'] = array(feature_names)
iris['data'] = array(data)

# print result header
test_data = []
test_iClass = []
total = len(test_set)
correct = 0

print('\n\n===== Iris data set =====\n')
print('train_set = %d' % len(train_set))
print('test_set = %d' % total)

# build decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris['data'], iris['target'])

# test
print('\n\n----- decision tree -----\n')

for line in test_set:
    row = line.strip('\n')
    row = row.split(',')
    iClass = row.pop()
    test_iClass.append(iClass)
    row = list(map(float, row))
    test_data.append([array(row)])
    iPredict = clf.predict([array(row)])

    if iClass==iPredict[0]:
        correct += 1
        print('[correct] iClass = {0:15}'.format(iClass) + ' | Predict = ' + iPredict[0])
    else:
        print('[ wrong ] iClass = {0:15}'.format(iClass) + ' | Predict = ' + iPredict[0])

acc = correct / total
print('\ndecision tree accuracy = %f' % acc)

# KNN
print('\n\n----- kNN -----\n')
knn = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree").fit(iris['data'],iris['target'])
# test
correct = 0

for i in range(len(test_set)):
    iPredict = knn.predict(test_data[i])
    if test_iClass[i]==iPredict[0]:
        correct += 1
        print('[correct] iClass = {0:15}'.format(test_iClass[i]) + ' | Predict = ' + iPredict[0])
    else:
        print('[ wrong ] iClass = {0:15}'.format(test_iClass[i]) + ' | Predict = ' + iPredict[0])

acc = correct / total
print('\nkNN accuracy = %f' % acc)

# naïve Bayes
print('\n\n----- naïve Bayes -----\n')
nb_clf = GaussianNB()
nb_clf.fit(iris['data'],iris['target'])

# test
correct = 0
for i in range(len(test_set)):
    iPredict = nb_clf.predict(test_data[i])
    if test_iClass[i]==iPredict[0]:
        correct += 1
        print('[correct] iClass = {0:15}'.format(test_iClass[i]) + ' | Predict = ' + iPredict[0])
    else:
        print('[ wrong ] iClass = {0:15}'.format(test_iClass[i]) + ' | Predict = ' + iPredict[0])

acc = correct / total
print('\nnaïve Bayes accuracy = %f' % acc)
