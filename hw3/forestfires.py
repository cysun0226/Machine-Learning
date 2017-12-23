import csv
import math
import sys
import numpy as np
from numpy import array
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

forestfire_file = str(sys.argv[1])
lines = []
data = []
target = []
forestfire = {}
forestfire_nb = {}
cat_data = []
cont_data = []

def month_to_int(month):
    return {
        'jan': 0,
        'feb': 1,
        'mar': 2,
        'apr': 3,
        'may': 4,
        'jun': 5,
        'jul': 5,
        'aug': 4,
        'sep': 3,
        'oct': 2,
        'nov': 1,
        'dec': 0,
    }[month]

def day_to_int(day):
    return {
        'mon': 0,
        'tue': 1,
        'wed': 2,
        'thu': 3,
        'fri': 4,
        'sat': 5,
        'sun': 6,
    }[day]

def int_to_day(num):
    return {
        0 : 'mon',
        1 : 'tue',
        2 : 'wed',
        3 : 'thu',
        4 : 'fri',
        5 : 'sat',
        6 : 'sun',
    }[num]

# main
# read forestfires.csv
with open(forestfire_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        lines.append(row)
feature_names = lines[0]

# shuffle and pick the training set
lines = lines[1:len(lines)]
shuffle(lines)
train_set = lines[0:math.floor(0.7*len(lines))]
test_set = lines[math.floor(0.7*len(lines)):len(lines)]
day_list = []
month_list = []

# train set
for line in train_set:
    d = []
    cd = []
    d.append(int(line[0])) # X
    d.append(int(line[1])) # Y
    d.append(month_to_int(line[2])) # month
    d.append(day_to_int(line[3])) # day
    d.append(float(line[4])) # FFMC
    d.append(float(line[5])) # DMC
    d.append(float(line[6])) # DC
    d.append(float(line[7])) # ISI
    d.append(float(line[8])) # temp
    d.append(float(line[9])) # RH
    d.append(float(line[10])) # wind
    d.append(float(line[11])) # rain
    data.append(d)
    db = list(d)
    del db[2]
    del db[2]
    cont_data.append(db)
    cd.append(month_to_int(line[2]))
    cd.append(day_to_int(line[3]))
    cat_data.append(cd)
    target.append(float(line[12]))

forestfire['target'] = array(target)
forestfire['feature_names'] = array(feature_names)
forestfire['data'] = array(data)
forestfire_nb['target'] = array(target)
forestfire_nb['cont_data'] = array(cont_data)
forestfire_nb['cat_data'] = array(cat_data)

# test set
test_data = []
testNB_cat_data = []
testNB_cont_data = []
test_target = []
for line in test_set:
    d = []
    cd = []
    d.append(int(line[0])) # X
    d.append(int(line[1])) # Y
    d.append(month_to_int(line[2])) # month
    d.append(day_to_int(line[3])) # day
    d.append(float(line[4])) # FFMC
    d.append(float(line[5])) # DMC
    d.append(float(line[6])) # DC
    d.append(float(line[7])) # ISI
    d.append(float(line[8])) # temp
    d.append(float(line[9])) # RH
    d.append(float(line[10])) # wind
    d.append(float(line[11])) # rain
    test_data.append([array(d)])
    db = list(d)
    del db[2]
    del db[2]
    testNB_cont_data.append(db)
    cd.append(month_to_int(line[2]))
    cd.append(day_to_int(line[3]))
    testNB_cat_data.append(cd)
    test_target.append(float(line[12]))

# print result header
total = len(test_set)
correct = 0

print('\n\n===== Forestfire data set =====\n')
print('train_set = %d' % len(train_set))
print('test_set = %d' % total)

# build decision tree
tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(forestfire['data'], forestfire['target'])

# test
print('\n\n----- decision tree -----\n')
acc_sum = 0

for i in range(len(test_set)):
    aPredict = tree_regressor.predict(test_data[i])
    logTest = 0 if test_target[i] == 0 else math.log10(test_target[i])
    predTest = 0 if aPredict[0] == 0 else math.log10(aPredict[0])
    acc = abs(logTest - predTest)
    acc_sum += acc
    # print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))

tree_acc = acc_sum / total
print('\ndecision tree predict diff = %f' % tree_acc)

# kNN
# normalize
knn_train = {}
knn_train = dict(forestfire)
knn_data = []

for d in range(len(forestfire['data'][0])):
    feature = []
    for r in range(len(forestfire['data'])):
        feature.append(forestfire['data'][r][d])
    knn_data.append(feature)

for f in range(len(forestfire['data'][0])):
    knn_data[f] = normalize([knn_data[f]])

for d in range(len(forestfire['data'][0])):
    for r in range(len(forestfire['data'])):
        knn_train['data'][r][d] = knn_data[d][0][r]

# kNN regressor
neiRgr = KNeighborsRegressor(n_neighbors=10)
neiRgr.fit(knn_train['data'], knn_train['target'])

# test
print('\n\n----- kNN -----\n')
acc_sum = 0

for i in range(len(test_set)):
    aPredict = neiRgr.predict(test_data[i])
    logTest = 0 if test_target[i] == 0 else math.log10(test_target[i])
    predTest = 0 if aPredict[0] == 0 else math.log10(aPredict[0])
    acc = abs(logTest - predTest)
    acc_sum += acc
    # print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))

tree_acc = acc_sum / total
print('\nkNN predict diff = %f' % tree_acc)

# naïve Bayes
# gaussian NB model on the continuous part
gNB = GaussianNB()
y_train = np.asarray(forestfire_nb['target'], dtype="|S6")
gNB.fit(forestfire_nb['cont_data'], y_train)

# multinomial NB model on the categorical part with Laplace smoothing
mNB = MultinomialNB(alpha=1.0)
mNB.fit(forestfire_nb['cat_data'], y_train)

# get predict for two NB model
hy_pred = []
for i in range(len(forestfire_nb['cont_data'])):
    gp = gNB.predict([forestfire_nb['cont_data'][i]])
    mp = mNB.predict([forestfire_nb['cat_data'][i]])
    g = float(gp[0].decode('UTF-8'))
    m = float(mp[0].decode('UTF-8'))
    p = []
    p.append(g)
    p.append(m)
    hy_pred.append(p)
hy_pred = array(hy_pred)

hNB = GaussianNB()
hNB.fit(hy_pred, y_train)

# test
print('\n\n----- naïve Bayes -----\n')
acc_sum = 0

for i in range(len(test_set)):
    gp = gNB.predict([testNB_cont_data[i]])
    mp = mNB.predict([testNB_cat_data[i]])
    g = float(gp[0].decode('UTF-8'))
    m = float(mp[0].decode('UTF-8'))
    p = [g,m]
    hp = hNB.predict([p])
    h = float(hp[0].decode('UTF-8'))

    logTest = 0 if test_target[i] == 0 else math.log10(test_target[i])
    predTest = 0 if h == 0 else math.log10(h)
    acc = abs(logTest - predTest)
    acc_sum += acc
    # print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))

tree_acc = acc_sum / total
print('\nnaïve Bayes predict diff = %f' % tree_acc)

print()
# for i in range(len(test_set)):
# gPredict = gNB.predict([testNB_cont_data[i]])
# print(gPredict[0].decode('UTF-8'))
# print(float(gPredict[0].decode('UTF-8')))
    # mPredict = mNB.predict([testNB_cat_data[i]])

    # np.hstack((gPredict,mPredict))
    # print(gPredict[0].decode('UTF-8'))
    # logTest = 0 if test_target[i] == 0 else math.log10(test_target[i])
    # predTest = 0 if aPredict[0] == 0 else math.log10(aPredict[0])
    # acc = abs(logTest - predTest)
    # acc_sum += acc
    # print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))

# tree_acc = acc_sum / total
# print('\nkNN predict diff = %f' % tree_acc)
