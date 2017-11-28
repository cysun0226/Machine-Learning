import csv
import math
import operator
import sys
from heapq import *

# train data
train = []
tree_leaf = 0
train_file = str(sys.argv[1])
test_file = str(sys.argv[2])
heap = []

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

class NN(object):
    def __init__(self):
        self.dist = None
        self.ecoli = None
    def __str__(self):
        out = 'dist: ' + str(self.dist) + ' || '
        out = out + str(self.ecoli)
        return out
    def __lt__(self, other):
        return self.dist >= other.dist


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


# KD tree
def build_KD_tree(train):
    # find the dimension that has most largest variance
    attr_num = len(train[0].attr)
    select_attr = []
    max_var = 0
    for d in range(attr_num):
        attr = []
        for i in range(len(train)):
            attr.append(train[i].attr[d])
        var = variance(attr)
        if var >= max_var:
            max_var = var
            dim = d
            select_attr = attr
    # set the mid of the dimension as pivot
    #pivot = median(select_attr)
    new_node = Node()
    train.sort(key=lambda x: x.attr[dim])
    if len(train) % 2 == 0:
        pivot = (train[len(train)//2].attr[dim] + train[len(train)//2-1].attr[dim])/2
        new_node.ecoli = train[len(train)//2]
        m = len(train)//2
    if len(train) % 2 == 1:
        pivot = train[(len(train)-1)//2].attr[dim]
        new_node.ecoli = train[(len(train)-1)//2]
        m = (len(train)-1)//2

    new_node.pivot = pivot
    new_node.dim = dim

    # separate data by pivot
    left_train = []
    right_train = []
    for i in range(len(train)):
        if train[i].attr[dim] <= pivot:
        #if i < m:
            left_train.append(train[i])
        else:
            right_train.append(train[i])

    if len(right_train) == 0:
        # print('right = 0')
        # print('dim = %d' % dim)
        # print('left = %d' % len(left_train))
        p = 0
        l = len(left_train)
        for i in range(l-1, -1, -1):
            # print(left_train[i])
            if left_train[i].attr[dim] == pivot:
                p += 1
                right_train.append(left_train[l-p])
        for i in range(p):
            left_train.pop()

    if len(left_train) == 1:
        new_leaf = Leaf()
        new_leaf.ecoli = left_train[0]
        new_node.leftChild = new_leaf
    else:
        new_node.leftChild = build_KD_tree(left_train)
    # print('type(new_node.leftChild) = ' + new_node.leftChild.__class__.__name__)
    new_node.leftChild.parent = new_node
    new_node.leftChild.childType = 'leftChild'

    # print('right_train = ', end='')
    # print(len(right_train))
    if len(right_train) == 1:
        new_leaf = Leaf()
        new_leaf.ecoli = right_train[0]
        new_node.rightChild = new_leaf
    else:
        new_node.rightChild = build_KD_tree(right_train)
    # print('type(new_node.rightChild) = ' + new_node.rightChild.__class__.__name__)
    new_node.rightChild.parent = new_node
    new_node.rightChild.childType = 'rightChild'

    return new_node

# all the subset is same
def class_num(train):
    subset = {}
    for i in range(len(train)):
        if train[i].ecoliClass not in subset:
            subset[train[i].ecoliClass] = 1
    return len(subset)

# find nearest neighbor
# query_tree
def query_tree(root, target):
    # while isinstance(root, Node):
    while root.__class__.__name__ == 'Node':
        if target.attr[root.dim] <= root.pivot:
            root = root.getLeftChild()
        else:
            root = root.getRightChild()
    return root

# shortest distance in leaf
def leaf_nearest(leaf, target):
    min_dist = math.inf
    near_leaf = [ min_dist, leaf.leaves[0]]
    for i in len(leaf.leaves):
        dist = euclidean_distance(leaf.leaves[i], target)
        if  dist < min_dist:
            min_dist = dist
            near_leaf[0] = min_dist
            near_leaf[1] = leaf.leaves[i]
    return near_leaf

# nearest neighbor
def nearest_neighbor(root, target):
    cur_node = root
    path = []
    while cur_node.__class__.__name__ == 'Node':
        path.append(cur_node)
        if target.attr[cur_node.dim] <= cur_node.pivot:
            cur_node = cur_node.getLeftChild()
        else:
            cur_node = cur_node.getRightChild()
    leaf = cur_node
    min_dist = euclidean_distance(leaf.ecoli, target)
    nearest = leaf.ecoli

    # compare min_dist with the parent node
    while len(path) != 0:
        back = path.pop()
        b_node = Ecoli()
        for i in range(len(target.attr)):
            b_node.attr.append(target.attr[i])
        b_node.attr[back.dim] = back.pivot # hyperPlain
        b_dist = euclidean_distance(b_node, target)
        if b_dist <= min_dist:
            if target.attr[back.dim] > back.pivot:
                cur_node = back.getLeftChild()
            else:
                cur_node = back.getRightChild()
            min_ecoli = traversal(cur_node, target)
            n_dist = euclidean_distance(min_ecoli, target)
            if n_dist < min_dist:
                min_dist = n_dist
                nearest = min_ecoli

    return nearest


def traversal(root, target):
    if root.__class__.__name__ == 'Node':
        e1 = traversal(root.getLeftChild(), target)
        e2 = traversal(root.getRightChild(), target)
        d1 = euclidean_distance(e1, target)
        d2 = euclidean_distance(e2, target)
        if d1 < d2:
            return e1
        else:
            return e2
    else:
        return root.ecoli


def kNN_heap(root, target, k):
    cur_node = root
    path = []
    global heap
    heap = []
    while cur_node.__class__.__name__ == 'Node':
        path.append(cur_node)
        if target.attr[cur_node.dim] <= cur_node.pivot:
            cur_node = cur_node.getLeftChild()
        else:
            cur_node = cur_node.getRightChild()
    leaf = cur_node
    new_nbr = NN()
    min_dist = euclidean_distance(leaf.ecoli, target)
    nearest = leaf.ecoli
    new_nbr.dist = min_dist
    new_nbr.ecoli = nearest
    heappush(heap, new_nbr)

    # while len(heap) < k:
    #     back = path.pop()
    #     if cur_node.childType == 'leftChild':
    #         trav_add_nbr(back.getRightChild(), target)
    #     else:
    #         trav_add_nbr(back.getLeftChild(), target)
    # # remove the redundant leaves
    # while len(heap) > k:
    #     heappop(heap)

    while len(path) != 0:
        back = path.pop()
        b_node = Ecoli()
        for i in range(len(target.attr)):
            b_node.attr.append(target.attr[i])
        b_node.attr[back.dim] = back.pivot # hyperPlain
        b_dist = euclidean_distance(b_node, target)
        if b_dist <= heap[0].dist or len(heap) < k:
            if target.attr[back.dim] > back.pivot:
                trav_add_nbr(back.getLeftChild(), target)
            else:
                trav_add_nbr(back.getRightChild(), target)
            # remove the remain leaves
            while len(heap) > k:
                heappop(heap)

    heap.sort()
    return heap

def kNN_classify(heap):
    class_num = {'cp': 0, 'im': 0, 'pp': 0, 'imU': 0, 'om': 0, 'omL': 0, 'imL': 0, 'imS': 0}
    for e in heap:
        class_num[e.ecoli.ecoliClass] += 1
    #print(max(class_num.iteritems(), key=operator.itemgetter(1)))
    #print(max(class_num, key=class_num.get))
    sorted_class = sorted(class_num.items(), key=operator.itemgetter(1, 0))
    sorted_class.reverse()

    return sorted_class[0][0]

def trav_add_nbr(root, target):
    global heap
    if root.__class__.__name__ == 'Node':
        trav_add_nbr(root.getLeftChild(), target)
        trav_add_nbr(root.getRightChild(), target)
    else:
        same = 0
        for e in heap:
            if e.ecoli == root.ecoli:
                same += 1
        if same == 0:
            new_nbr = NN()
            new_nbr.dist = euclidean_distance(root.ecoli, target)
            new_nbr.ecoli = root.ecoli
            heappush(heap, new_nbr)
        return





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

# build KD tree
root = build_KD_tree(train)
# read test file
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


correct = 0

for r in test:
    if (kNN_classify(kNN_heap(root, r, 1)) == r.ecoliClass):
        correct += 1

# 1
print('KNN accuracy: %f' % (correct/len(test)))
#print(nearest_neighbor(root, test[0]).index)
#print(nearest_neighbor(root, test[1]).index)
#print(nearest_neighbor(root, test[2]).index)
print(kNN_heap(root, test[0], 1)[0].ecoli.index)
print(kNN_heap(root, test[1], 1)[0].ecoli.index)
print(kNN_heap(root, test[2], 1)[0].ecoli.index)

print('')

# 5
correct = 0
for r in test:
    if (kNN_classify(kNN_heap(root, r, 5)) == r.ecoliClass):
        correct += 1
print('KNN accuracy: %f' % (correct/len(test)))

for d in range(3):
    h = kNN_heap(root, test[d], 5)
    h.reverse()
    out = ''
    for i in range(5):
        out += str(h[i].ecoli.index) + ' '
    print(out)

print('')

# 10
correct = 0
for r in test:
    if (kNN_classify(kNN_heap(root, r, 10)) == r.ecoliClass):
        correct += 1
print('KNN accuracy: %f' % (correct/len(test)))

for d in range(3):
    h = kNN_heap(root, test[d], 10)
    h.reverse()
    out = ''
    for i in range(10):
        out += str(h[i].ecoli.index) + ' '
    print(out)

print('')

# 100
correct = 0
for r in test:
    if (kNN_classify(kNN_heap(root, r, 100)) == r.ecoliClass):
        correct += 1
print('KNN accuracy: %f' % (correct/len(test)))

for d in range(3):
    h = kNN_heap(root, test[d], 100)
    h.reverse()
    out = ''
    for i in range(100):
        out += str(h[i].ecoli.index) + ' '
    print(out)
