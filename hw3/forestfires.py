import csv
import math
import sys
import numpy as np
from numpy import array
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()

forestfire_file = str(sys.argv[1])
lines = []
data = []
target = []
forestfire = {}

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

# train set
for line in train_set:
    d = []
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
    target.append(float(line[12]))

forestfire['target'] = array(target)
forestfire['feature_names'] = array(feature_names)
forestfire['data'] = array(data)

# test set
test_data = []
test_target = []
for line in test_set:
    d = []
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
# p = tree_regressor.predict([forestfire['data'][0]])

# test
print('\n\n----- decision tree -----\n')
acc_sum = 0

for i in range(len(test_set)):
    aPredict = tree_regressor.predict(test_data[i])

    x=True if 'a'=='a' else False
    logTest = 0 if test_target[i] == 0 else math.log10(test_target[i])
    predTest = 0 if aPredict[0] == 0 else math.log10(aPredict[0])
    acc = abs(logTest - predTest)
    acc_sum += acc
    print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))

tree_acc = acc_sum / total
print('\ndecision tree predict diff = %f' % tree_acc)
