import csv
import math
import sys
import numpy as np
from numpy import array
from random import shuffle

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import normalize


star_file = str(sys.argv[1])
lines = []
data = []
target = []
starSign = {}

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
print(feature_names)
print('\n\n===== star-sign data set =====\n')
print('資料總數 = %d 份\n' % len(lines))
total_num = len(lines)
# print('train_set = %d' % math.floor(0.9*len(lines)))
# print('test_set = %d' % (len(lines) - math.floor(0.1*len(lines))))

# sys.stderr.write("\nexecute time: ")
# time = int(input("\nexecute time: "))
# sys.stderr.write("\nprint detailed results?(y/n): ")
# print_result = input("print detailed results?(y/n): ")
# sys.stderr.write("\n\n--- process status ---\n\n")


# for x in range(time):
#     print('\n\n===== test %d =====' % (x+1))
#
#     # shuffle and pick the training set
#     lines = lines[1:len(lines)]
#     shuffle(lines)
#     train_set = lines[0:math.floor(0.7*len(lines))]
#     test_set = lines[math.floor(0.7*len(lines)):len(lines)]
#     day_list = []
#     month_list = []
#

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

# Aries
Aries = []
for row in data:
    if(row['星座'] == '牡羊座'):
        Aries.append(row)

# 


# for d in data:
#     print(d)

    # d = []
    # cd = []
    # d.append(int(line[0])) # X
    # d.append(int(line[1])) # Y
    # d.append(month_to_int(line[2])) # month
    # d.append(day_to_int(line[3])) # day
    # d.append(float(line[4])) # FFMC
    # d.append(float(line[5])) # DMC
    # d.append(float(line[6])) # DC
    # d.append(float(line[7])) # ISI
    # d.append(float(line[8])) # temp
    # d.append(float(line[9])) # RH
    # d.append(float(line[10])) # wind
    # d.append(float(line[11])) # rain
    # data.append(d)
    # db = list(d)
    # del db[2]
    # del db[2]
    # cont_data.append(db)
    # cd.append(month_to_int(line[2]))
    # cd.append(day_to_int(line[3]))
    # cat_data.append(cd)
    # target.append(float(line[12]))
    # if float(line[12]) == 0:
    #     target.append(-10) # area
    # else:
    #     target.append(math.log10(float(line[12]))) # area

# forestfire['target'] = array(target)
# forestfire['feature_names'] = array(feature_names)
# forestfire['data'] = array(data)
# forestfire_nb['target'] = array(target)
# forestfire_nb['cont_data'] = array(cont_data)
# forestfire_nb['cat_data'] = array(cat_data)
#
#     # test set
#     test_data = []
#     testNB_cat_data = []
#     testNB_cont_data = []
#     test_target = []
#     for line in test_set:
#         d = []
#         cd = []
#         d.append(int(line[0])) # X
#         d.append(int(line[1])) # Y
#         d.append(month_to_int(line[2])) # month
#         d.append(day_to_int(line[3])) # day
#         d.append(float(line[4])) # FFMC
#         d.append(float(line[5])) # DMC
#         d.append(float(line[6])) # DC
#         d.append(float(line[7])) # ISI
#         d.append(float(line[8])) # temp
#         d.append(float(line[9])) # RH
#         d.append(float(line[10])) # wind
#         d.append(float(line[11])) # rain
#         test_data.append([array(d)])
#         db = list(d)
#         del db[2]
#         del db[2]
#         testNB_cont_data.append(db)
#         cd.append(month_to_int(line[2]))
#         cd.append(day_to_int(line[3]))
#         testNB_cat_data.append(cd)
#         test_target.append(float(line[12]))
#         # if float(line[12]) == 0:
#         #     test_target.append(-10) # area
#         # else:
#         #     test_target.append(math.log10(float(line[12]))) # area
#
#
#     total = len(test_set)
#     correct = 0
#
#     # build decision tree
#     tree_regressor = DecisionTreeRegressor(random_state=0)
#     tree_regressor.fit(forestfire['data'], forestfire['target'])
#
#     # test
#     if print_result=='y':
#         print('\n\n----- decision tree -----\n')
#     acc_sum = 0
#
#     for i in range(len(test_set)):
#         aPredict = tree_regressor.predict(test_data[i])
#         logTest = -1 if test_target[i] == 0 else math.log10(test_target[i])
#         predTest = -1 if aPredict[0] == 0 else math.log10(aPredict[0])
#         # logTest = test_target[i]
#         # predTest = aPredict[0]
#         acc = abs(logTest - predTest)
#         acc_sum += acc
#         if print_result=='y':
#             print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))
#
#     tree_acc = acc_sum / total
#     tree_avg_sum += tree_acc
#     print('\ndecision tree predict diff = %f' % tree_acc)
#
#     # kNN
#     # normalize
#     knn_train = {}
#     knn_train = dict(forestfire)
#     knn_data = []
#
#     for d in range(len(forestfire['data'][0])):
#         feature = []
#         for r in range(len(forestfire['data'])):
#             feature.append(forestfire['data'][r][d])
#         knn_data.append(feature)
#
#     for f in range(len(forestfire['data'][0])):
#         knn_data[f] = normalize([knn_data[f]])
#
#     for d in range(len(forestfire['data'][0])):
#         for r in range(len(forestfire['data'])):
#             knn_train['data'][r][d] = knn_data[d][0][r]
#
#     # kNN regressor
#     neiRgr = KNeighborsRegressor(n_neighbors=5)
#     neiRgr.fit(knn_train['data'], knn_train['target'])
#
#     # test
#     if print_result=='y':
#         print('\n\n----- kNN -----\n')
#     acc_sum = 0
#
#     for i in range(len(test_set)):
#         aPredict = neiRgr.predict(test_data[i])
#         logTest = -1 if test_target[i] == 0 else math.log10(test_target[i])
#         predTest = -1 if aPredict[0] == 0 else math.log10(aPredict[0])
#         # logTest = test_target[i]
#         # predTest = aPredict[0]
#         acc = abs(logTest - predTest)
#         acc_sum += acc
#         if print_result=='y':
#             print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))
#
#     knn_acc = acc_sum / total
#     knn_avg_sum += knn_acc
#     print('\nkNN predict diff = %f' % knn_acc)
#
#     # naïve Bayes
#     # gaussian NB model on the continuous part
#     gNB = GaussianNB()
#     y_train = np.asarray(forestfire_nb['target'], dtype="|S6")
#     gNB.fit(forestfire_nb['cont_data'], y_train)
#
#     # multinomial NB model on the categorical part with Laplace smoothing
#     mNB = MultinomialNB(alpha=1.0)
#     mNB.fit(forestfire_nb['cat_data'], y_train)
#
#     # get predict for two NB model
#     hy_pred = []
#     for i in range(len(forestfire_nb['cont_data'])):
#         gp = gNB.predict([forestfire_nb['cont_data'][i]])
#         mp = mNB.predict([forestfire_nb['cat_data'][i]])
#         g = float(gp[0].decode('UTF-8'))
#         m = float(mp[0].decode('UTF-8'))
#         p = [g,m]
#         hy_pred.append(p)
#     hy_pred = array(hy_pred)
#
#     hNB = GaussianNB()
#     hNB.fit(hy_pred, y_train)
#
#     # test
#     if print_result=='y':
#         print('\n\n----- naïve Bayes -----\n')
#     acc_sum = 0
#
#     for i in range(len(test_set)):
#         gp = gNB.predict([testNB_cont_data[i]])
#         mp = mNB.predict([testNB_cat_data[i]])
#         g = float(gp[0].decode('UTF-8'))
#         m = float(mp[0].decode('UTF-8'))
#         p = [g,m]
#         hp = hNB.predict([p])
#         h = float(hp[0].decode('UTF-8'))
#
#         logTest = -1 if test_target[i] == 0 else math.log10(test_target[i])
#         predTest = -1 if h == 0 else math.log10(h)
#         # logTest = test_target[i]
#         # predTest = h
#         acc = abs(logTest - predTest)
#         acc_sum += acc
#         if print_result=='y':
#             print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))
#
#     nb_acc = acc_sum / total
#     nb_avg_sum += nb_acc
#     print('\nnaïve Bayes predict diff = %f' % nb_acc)
#     print()
#
#     # process line
#     # print('time = %d' % time)
#     # print('x = %d' % x)
#
#     i_per = math.floor(((x+1)/time)*100)
#     k = i_per + 1
#     pl = '['+ '#'*(i_per//2)+' '*((100-k)//2) + ']'
#     sys.stderr.write('\r'+pl+' (%s%%)'%(i_per))
#     sys.stderr.flush()
#     # i_print = math.floor((i/time)*50)
#     # sys.stderr.write('\r')
#     # sys.stderr.write("[%-50s] %.2f%%" % ('='*i_print, i_per))
#     # sys.stderr.flush()
#
# 	# end = time.time()
# 	# elapsed = end - start
# 	# elapsed = elapsed / 60
#
#
#
# 	# sys.stdout.write('Time taken: ')
# 	# sys.stdout.write("%d" % elapsed)
# 	# sys.stdout.write(' min.\t')
#
#
#
# # avg test result
# sys.stderr.write('\n\n\n===== test results =====')
# sys.stderr.write('\n\ndecision tree avg diff = %f' % (tree_avg_sum / time))
# sys.stderr.write('\n\nkNN avg diff = %f' % (knn_avg_sum / time))
# sys.stderr.write('\n\nnaïve Bayes avg diff = %f' % (nb_avg_sum / time))
# sys.stderr.write('\n\n')
#
# print('\n\n===== test results =====')
# print('\ndecision tree avg diff = %f' % (tree_avg_sum / time))
# print('\nkNN avg diff = %f' % (knn_avg_sum / time))
# print('\nnaïve Bayes avg diff = %f' % (nb_avg_sum / time))
# print()
#
#
# # for i in range(len(test_set)):
# # gPredict = gNB.predict([testNB_cont_data[i]])
# # print(gPredict[0].decode('UTF-8'))
# # print(float(gPredict[0].decode('UTF-8')))
#     # mPredict = mNB.predict([testNB_cat_data[i]])
#
#     # np.hstack((gPredict,mPredict))
#     # print(gPredict[0].decode('UTF-8'))
#     # logTest = 0 if test_target[i] == 0 else math.log10(test_target[i])
#     # predTest = 0 if aPredict[0] == 0 else math.log10(aPredict[0])
#     # acc = abs(logTest - predTest)
#     # acc_sum += acc
#     # print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))
#
# # tree_acc = acc_sum / total
# # print('\nkNN predict diff = %f' % tree_acc)
