import sys
import math
import numpy as np
from numpy import array
from sklearn import tree
from random import shuffle

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
print('accuracy = %f' % acc)

# iPredict = clf.predict([array([6.2,2.6,4.8,1.7])])
# print(iPredict)
# iPredict = clf.predict([array([5.1,3.5,1.4,0.1])])
# print(iPredict)
# iPredict = clf.predict([array([5.6,2.85,3.55,1.3])])
# print(iPredict)


# print(iris['target_names'])
# print(iris['target'])
# print(iris['feature_names'])
# print(iris['data'])
#
# print((iris['data'][0][0]))
