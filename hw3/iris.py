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
train_file = open(iris_file, 'r', encoding='UTF-8')

# main
target_names = ['setosa', 'versicolor', 'virginica']
target_names = np.array(target_names)

feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
feature_names = np.array(feature_names)

lines = []
tree_avg_sum = 0
knn_avg_sum = 0
nb_avg_sum = 0

# read iris.data
for line in train_file.readlines():
    if line=='\n':
        continue
    lines.append(line)

# print result header
print('\n\n===== Iris data set =====\n')
print('train_set = %d' % math.floor(0.7*len(lines)))
print('test_set = %d' % (len(lines) - math.floor(0.7*len(lines))))

time = int(input("\nexecute time: "))
print_result = input("print detailed results?(y/n): ")

for x in range(time):
    print('\n\n===== test %d =====' % (x+1))

    # shuffle and pick the training set
    target = []
    data = []
    iris = {}
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


    test_data = []
    test_iClass = []
    total = len(test_set)
    correct = 0

    # build decision tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris['data'], iris['target'])

    # test
    if print_result == 'y':
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
            if print_result == 'y':
                print('[correct] iClass = {0:15}'.format(iClass) + ' | Predict = ' + iPredict[0])
        else:
            if print_result == 'y':
                print('[ wrong ] iClass = {0:15}'.format(iClass) + ' | Predict = ' + iPredict[0])

    acc = correct / total
    tree_avg_sum += acc
    print('\ndecision tree accuracy = %f' % acc)

    # KNN
    if print_result == 'y':
        print('\n\n----- kNN -----\n')
    knn = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree").fit(iris['data'],iris['target'])
    # test
    correct = 0

    for i in range(len(test_set)):
        iPredict = knn.predict(test_data[i])
        if test_iClass[i]==iPredict[0]:
            correct += 1
            if print_result == 'y':
                print('[correct] iClass = {0:15}'.format(test_iClass[i]) + ' | Predict = ' + iPredict[0])
        else:
            if print_result == 'y':
                print('[ wrong ] iClass = {0:15}'.format(test_iClass[i]) + ' | Predict = ' + iPredict[0])

    acc = correct / total
    knn_avg_sum += acc
    print('\nkNN accuracy = %f' % acc)

    # naïve Bayes
    if print_result == 'y':
        print('\n\n----- naïve Bayes -----\n')
    nb_clf = GaussianNB()
    nb_clf.fit(iris['data'],iris['target'])

    # test
    correct = 0
    for i in range(len(test_set)):
        iPredict = nb_clf.predict(test_data[i])
        if test_iClass[i]==iPredict[0]:
            correct += 1
            if print_result == 'y':
                print('[correct] iClass = {0:15}'.format(test_iClass[i]) + ' | Predict = ' + iPredict[0])
        else:
            if print_result == 'y':
                print('[ wrong ] iClass = {0:15}'.format(test_iClass[i]) + ' | Predict = ' + iPredict[0])

    acc = correct / total
    nb_avg_sum += acc
    print('\nnaïve Bayes accuracy = %f' % acc)
print()

# avg test result
print('\n\n===== test results =====')
print('\ndecision tree avg accuracy = %f' % (tree_avg_sum / time))
print('\nkNN avg accuracy = %f' % (knn_avg_sum / time))
print('\nnaïve Bayes avg accuracy = %f' % (nb_avg_sum / time))
print()
