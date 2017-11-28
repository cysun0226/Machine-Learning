import csv
import math
import operator
from operator import itemgetter
import sys

# train data
train = []
tree_leaf = 0
train_file = str(sys.argv[1])
test_file = str(sys.argv[2])
# k = int(sys.argv[3])

# struct
class Node(object):
    def __init__(self):
        self.parent = None
        self.leftChild = None
        self.rightChild = None
        self.dim = None
        self.pivot = None
        self.ecoli = None
        self.childType = None

    def getLeftChild(self):
        return self.leftChild

    def getRightChild(self):
        return self.rightChild

class Leaf(object):
    def __init__(self):
        self.ecoli = None
        self.ecoliClass = None
        self.childType = None
        self.parent = None
    def __str__(self):
        out = 'leaf' + '\n'
        out += self.ecoli
        return out


class Ecoli(object):
    def __init__(self):
        self.index = None
        self.name = None
        self.attr = []
        self.ecoliClass = None
    def __str__(self):
        if self.index < 10:
            idx = str(self.index) + '  '
        if self.index >= 10 and self.index < 100:
            idx = str(self.index) + ' '
        if self.index >= 100:
            idx = str(self.index)
        if len(self.name) != 10:
            n = self.name + ' '
        else:
            n = self.name
        out = 'index: ' + idx + ' | name: ' + n + ' | attr:'
        for i in range(len(self.attr)):
            out = out + ' ' + format(self.attr[i], '.2f')
        out = out + ' | class: ' + self.ecoliClass
        return out

# mathmathetic function
def mean(seq):
    sum = 0;
    for x in range(len(seq)):
        sum += seq[x]
    return sum / len(seq)

def variance(seq):
    m = mean(seq)
    var_sum = 0
    for i in range(len(seq)):
        diff = seq[i] - m
        var_sum += diff*diff
    return var_sum / len(seq)

def median(seq):
    seq.sort()
    size = len(seq)
    if size % 2 == 0:
        m = (seq[size//2] + seq[size//2-1])/2
    if size % 2 == 1:
        m = seq[(size-1)//2]
    return m

def euclidean_distance(p1, p2):
    e_sum = 0
    for i in range(len(p1.attr)):
        diff = p1.attr[i] - p2.attr[i]
        e_sum += diff*diff
    return(math.sqrt(e_sum))

def kNN_classify(heap):
    class_num = {'cp': 0, 'im': 0, 'pp': 0, 'imU': 0, 'om': 0, 'omL': 0, 'imL': 0, 'imS': 0}
    for e in heap:
        class_num[e.ecoliClass] += 1
    #print(max(class_num.iteritems(), key=operator.itemgetter(1)))
    #print(max(class_num, key=class_num.get))
    sorted_class = sorted(class_num.items(), key=operator.itemgetter(1, 0))
    sorted_class.reverse()
    # if sorted_class[0][1] == sorted_class[1][1] and sorted_class[1][0] == 'cp':
    #     sorted_class[0][0] = 'cp'
    return sorted_class[0][0]

# main
# read train.csv
with open(train_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        if(row[0] != 'index'):
        	data = Ecoli()
        	data.index = int(row[0])
        	data.name = row[1]
        	for x in range(1,9+1):
        		data.attr.append(float(row[x+1]))
        	data.ecoliClass = row[11]
        	train.append(data)

test = []
with open(test_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        if(row[0] != 'index'):
        	data = Ecoli()
        	data.index = int(row[0])
        	data.name = row[1]
        	for x in range(1,9+1):
        		data.attr.append(float(row[x+1]))
        	data.ecoliClass = row[11]
        	test.append(data)

near = []
min_dist = math.inf
dist_list = []
for r in test:
    dist = []
    for t in train:
        p = []
        d = euclidean_distance(r, t)
        p.append(t)
        p.append(d)
        dist.append(p)
    dist.sort(key=itemgetter(1))
    dist_list.append(dist)

correct = 0

# 1
for i in range(len(test)):
    if (dist_list[i][0][0].ecoliClass == test[i].ecoliClass):
    #if (nearest_neighbor(root, r).ecoliClass == r.ecoliClass):
        correct += 1
print('KNN accuracy: %f' % (correct/len(test)))
print(dist_list[0][0][0].index)
print(dist_list[1][0][0].index)
print(dist_list[2][0][0].index)
#print(root.__class__.__name__)
print('')



# 5
correct = 0
for r in range(len(test)):
    h = []
    for i in range(5):
        h.append(dist_list[r][i][0])
    if kNN_classify(h) == test[r].ecoliClass:
        correct += 1
    #print('%d = ' % r, end='')
    #print(kNN_classify(h))
print('KNN accuracy: %f' % (correct/len(test)))
#for t in range(3):
for t in range(3):
    h = []
    d = []
    for i in range(5):
        h.append(dist_list[t][i][0])
        # h.append(dist_list[t][i])
    for i in h:
        # print('dist = %f || ' % i[1], end='')
        print(i.index, end='')
        #print(i[0], end='')
        print(' ', end='')
        # print('')
    print('')

print('')

# 10
correct = 0
for r in range(len(test)):
    h = []
    for i in range(10):
        h.append(dist_list[r][i][0])
    if kNN_classify(h) == test[r].ecoliClass:
        correct += 1
print('KNN accuracy: %f' % (correct/len(test)))
for t in range(3):
    h = []
    for i in range(10):
        h.append(dist_list[t][i][0])
    for i in h:
        print(i.index, end='')
        print(' ', end='')
    print('')

print('')
# 100
correct = 0
for r in range(len(test)):
    h = []
    for i in range(100):
        h.append(dist_list[r][i][0])
    if kNN_classify(h) == test[r].ecoliClass:
        correct += 1
print('KNN accuracy: %f' % (correct/len(test)))
for t in range(3):
    h = []
    for i in range(100):
        h.append(dist_list[t][i][0])
    for i in h:
        print(i.index, end='')
        print(' ', end='')
    print('')
