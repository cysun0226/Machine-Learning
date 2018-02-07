import starSignEnum
import csv
import math
import sys
import numpy as np
from numpy import array
from random import shuffle
import operator

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import normalize

star_file = str(sys.argv[1])
lines = []
data = []
target = []

# 計算各星座的特質
def calAttr(signAttr, signData):
    for i in range(1, 13): # 12 個星座
        for a in range(1, 32): # 32 題
            signAttr[starSignEnum.getSignName(i)][starSignEnum.getAttrName(a)] /= len(signData[starSignEnum.getSignName(i)])

# 列出每個星座前n高的特質
def highestAttr(signAttr, n):
    for i in range(1, 13): # 12 個星座
        sorted_a = sorted(signAttr[starSignEnum.getSignName(i)].items(), key=operator.itemgetter(1))
        sorted_a.reverse()
        print('---' + starSignEnum.getSignName(i) + '---\n')
        a = 0
        for attr in range(32):
            if (sorted_a[attr][0] == '專情') or (sorted_a[attr][0] == '重視友情'):
                continue
            a += 1
            print(sorted_a[attr])
            if a == 5:
                break
        print('\n\n')

# 每個特質最高與最低的n個星座
def maxAttrSign(signAttr, n):
    for a in range(1, 32):
        attr = {}
        for s in range(1, 13):
            attr[starSignEnum.getSignName(s)] = signAttr[starSignEnum.getSignName(s)][starSignEnum.getAttrName(a)]

        sorted_attr = sorted(attr.items(), key=operator.itemgetter(1))
        sorted_attr.reverse()

        print('--- ' + starSignEnum.getAttrName(a) + '---')
        print('\n前三名\n')
        for i in range(n):
            print(sorted_attr[i])
        print('\n後三名\n')
        for i in range(n):
            print(sorted_attr[11-i])
        print('\n\n')


starSignData = {}
starSignAttr = {}
starData = {}
stddev = {}
otherSignData = {}
otherSignAttr = {}

a = {}
li = {}
for i in range(1, 32):
    a[starSignEnum.getAttrName(i)] = 0
    li[starSignEnum.getAttrName(i)] = list()

for i in range(1, 13):
    starSignData[starSignEnum.getSignName(i)] = []
    starSignAttr[starSignEnum.getSignName(i)] = dict(a)
    otherSignData[starSignEnum.getSignName(i)] = []
    otherSignAttr[starSignEnum.getSignName(i)] = dict(a)
    stddev[starSignEnum.getSignName(i)] = dict(a)
    starData[starSignEnum.getSignName(i)] = dict(li)


# main
tree_avg_sum = 0
knn_avg_sum = 0
nb_avg_sum = 0

# read star.csv
with open(star_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 0
    for row in spamreader:
        if i == 0: # header
            row = row[0].split(',')
            lines.append(row)
            i += 1
        else:
            row = row[2].split(',')
            lines.append(row)

feature_names = lines[0]


# print result header
del feature_names[0] # remove time stamp
del feature_names[len(feature_names)-2] # remove commands
# print(feature_names)
print('\n\n===== star-sign data set =====\n')
print('資料總數 = %d 份\n' % len(lines))
total_num = len(lines)
other_total = 0

# lines
del lines[0]
i = 0
for line in lines:
    del line[0] # remove time
    del line[len(line)-2]

i = 0
for line in lines:
    if len(line) != 33:
        # print('line[%d] has incorrect element num' % i)
        total_num -= 1
        continue

    d = {}
    d['分析對象'] = line[0]
    d['星座'] = line[1]
    d['耐性'] = int(line[2])
    d['脾氣暴躁'] = int(line[3])
    d['幼稚'] = int(line[4])
    d['頑固'] = int(line[5])
    d['心思細膩'] = int(line[6])
    d['保守'] = int(line[7])
    d['冷靜'] = int(line[8])
    d['樂觀'] = int(line[9])
    d['活潑'] = int(line[10])
    d['公正'] = int(line[11])
    d['優柔寡斷'] = int(line[12])
    d['強勢'] = int(line[13])
    d['浪漫'] = int(line[14])
    d['過度理想化'] = int(line[15])
    d['斤斤計較'] = int(line[16])
    d['心機重'] = int(line[17])
    d['完美主義'] = int(line[18])
    d['愛計仇'] = int(line[19])
    d['與眾不同'] = int(line[20])
    d['愛面子'] = int(line[21])
    d['有魅力'] = int(line[22])
    d['正義感'] = int(line[23])
    d['重視友情'] = int(line[24])
    d['專情'] = int(line[25])
    d['愛哭'] = int(line[26])
    d['顧家'] = int(line[27])
    d['體貼'] = int(line[28])
    d['情緒化'] = int(line[29])
    d['口才'] = int(line[30])
    d['創意'] = int(line[31])
    d['潔癖'] = int(line[32])
    data.append(d)
    i += 1

# 將資料依星座分類並加總特質
for row in data:
    for i in range(1, 32):
        starSignAttr[row['星座']][starSignEnum.getAttrName(i)] += row[starSignEnum.getAttrName(i)]
        # starData[row['星座']][starSignEnum.getAttrName(i)].append(row[starSignEnum.getAttrName(i)])
        if row['分析對象'] == '他人':
            otherSignAttr[row['星座']][starSignEnum.getAttrName(i)] += row[starSignEnum.getAttrName(i)]

    for i in range(1, 13):
        if(row['星座'] == starSignEnum.getSignName(i)):
            starSignData[starSignEnum.getSignName(i)].append(row)
            if row['分析對象'] == '他人':
                otherSignData[starSignEnum.getSignName(i)].append(row)
                other_total += 1

starData = {}
for i in range(1, 13):
    starData[starSignEnum.getSignName(i)] = {}
    for a in range(1, 32):
        starData[starSignEnum.getSignName(i)][starSignEnum.getAttrName(a)] = []

for d in range(len(data)):
    for i in range(1, 32):
        tmp = data[d][starSignEnum.getAttrName(i)]
        starData[data[d]['星座']][starSignEnum.getAttrName(i)].append(tmp)
    # print(starData)
# starData['射手座']['樂觀'] = [0, 0, 0]

# 印出各星座資料數量
# for i in range(1, 13):
#     print(starSignEnum.getSignName(i) + ' = %d' % len(starSignData[starSignEnum.getSignName(i)]))

# 計算各星座的特質
# calAttr(starSignAttr, starSignData)

# 計算標準差
# print(starData['雙魚座']['樂觀'])
# # print(tst)
# print(starData['射手座']['樂觀'])

for i in range(1, 13):
    for a in range(1, 32):
        # print(starData[starSignEnum.getSignName(i)][starSignEnum.getAttrName(a)])
        stddev[starSignEnum.getSignName(i)][starSignEnum.getAttrName(a)] = np.std(array(starData[starSignEnum.getSignName(i)][starSignEnum.getAttrName(a)]))

# 列出每個星座的標準差
print('\n\n===== 標準差 =====\n\n')
for s in range(1, 13):
    print('- ' + starSignEnum.getSignName(s) + ' -\n')
    # print(stddev[starSignEnum.getSignName(s)])
    sorted_attr = sorted(stddev[starSignEnum.getSignName(s)].items(), key=operator.itemgetter(1))
    sorted_attr.reverse()
    for a in range(0, 31):
        space = '  '
        for i in range(6-len(sorted_attr[a][0])):
            space += '  '
        print('%s%s%s' % (sorted_attr[a][0], space, format(sorted_attr[a][1], '0.3f') ) )

    print('\n\n')


print('\n\n')

# # 列出每個星座前五高的特質
# highestAttr(starSignAttr, 5)
#
# # 每個特質最高與最低的星座
# maxAttrSign(starSignAttr, 3)
#
# # 他人眼中的星座
# print('\n=== 他人眼中的xx座 ===\n')
#
# print('資料總數 = %d 份\n' % other_total)
#
# for i in range(1, 13):
#     print(starSignEnum.getSignName(i) + ' = %d' % len(otherSignData[starSignEnum.getSignName(i)]))
#
# # 計算各星座的特質
# calAttr(otherSignAttr, otherSignData)
#
# print('\n\n')
#
# # 列出每個星座前三高的特質
# highestAttr(otherSignAttr, 5)
#
# # 每個特質最高與最低的星座
# maxAttrSign(otherSignAttr, 3)
