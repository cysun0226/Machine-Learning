import csv
import math
import operator
from random import shuffle

test_file = open('test.csv', 'w', newline='')
train_file = open('train_set.csv', 'w', newline='')

class_num = {'cp': 0, 'im': 0, 'pp': 0, 'imU': 0, 'om': 0, 'omL': 0, 'imL': 3, 'imS': 3}
data = []
train_set = []
test_set = []
header = [ 'index',	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

f_test = csv.writer(test_file, quoting = csv.QUOTE_ALL, delimiter=',')
f_test.writerow(header)

f_train = csv.writer(train_file, quoting = csv.QUOTE_ALL, delimiter=',')
f_train.writerow(header)

# read origin train file
with open('train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    idx = 0
    for row in spamreader:
        if idx != 0:
            data.append(row)
        idx += 1


# shuffle
data_set = [r for r in data]
shuffle(data_set)

# pick up test_set
test_num = 0
for r in data_set:
    r = r[0].split(',')
    #
    # if class_num[r[11]] < 4:
    #     class_num[r[11]] += 1
    #     test_set.append(r)
    # else:
    train_set.append(r)


# write file
print('test_set = %d' % len(test_set))
print('train_set = %d' % len(train_set))

lack = 20 - len(test_set)
for i in range(lack):
    test_set.append(train_set.pop())

for r in test_set:
    f_test.writerow(r)

for r in train_set:
    f_train.writerow(r)

test_file.close()
train_file.close()
