import csv
import math
import sys
import numpy as np
from numpy import array
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import cross_val_score
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
        'jul': 6,
        'aug': 7,
        'sep': 8,
        'oct': 9,
        'nov': 10,
        'dec': 11,
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
tree_avg_sum = 0
knn_avg_sum = 0
nb_avg_sum = 0

# read forestfires.csv
with open(forestfire_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        lines.append(row)
feature_names = lines[0]

# print result header
print('\n\n===== Forestfire data set =====\n')
print('train_set = %d' % math.floor(0.7*len(lines)))
print('test_set = %d' % (len(lines) - math.floor(0.7*len(lines))))

sys.stderr.write("\nexecute time: ")
time = int(input("\nexecute time: "))
sys.stderr.write("\nprint detailed results?(y/n): ")
print_result = input("print detailed results?(y/n): ")
sys.stderr.write("\n\n--- process status ---\n\n")

for x in range(time):
    print('\n\n===== test %d =====' % (x+1))

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
        # if float(line[12]) == 0:
        #     target.append(-10) # area
        # else:
        #     target.append(math.log10(float(line[12]))) # area

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
        # if float(line[12]) == 0:
        #     test_target.append(-10) # area
        # else:
        #     test_target.append(math.log10(float(line[12]))) # area


    total = len(test_set)
    correct = 0

    # build decision tree
    tree_regressor = DecisionTreeRegressor(random_state=0)
    tree_regressor.fit(forestfire['data'], forestfire['target'])

    # test
    if print_result=='y':
        print('\n\n----- decision tree -----\n')
    acc_sum = 0

    for i in range(len(test_set)):
        aPredict = tree_regressor.predict(test_data[i])
        logTest = -1 if test_target[i] == 0 else math.log10(test_target[i])
        predTest = -1 if aPredict[0] == 0 else math.log10(aPredict[0])
        # logTest = test_target[i]
        # predTest = aPredict[0]
        acc = abs(logTest - predTest)
        acc_sum += acc
        if print_result=='y':
            print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))

    tree_acc = acc_sum / total
    tree_avg_sum += tree_acc
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
    neiRgr = KNeighborsRegressor(n_neighbors=3)
    neiRgr.fit(knn_train['data'], knn_train['target'])

    # test
    if print_result=='y':
        print('\n\n----- kNN -----\n')
    acc_sum = 0

    for i in range(len(test_set)):
        aPredict = neiRgr.predict(test_data[i])
        logTest = -1 if test_target[i] == 0 else math.log10(test_target[i])
        predTest = -1 if aPredict[0] == 0 else math.log10(aPredict[0])
        # logTest = test_target[i]
        # predTest = aPredict[0]
        acc = abs(logTest - predTest)
        acc_sum += acc
        if print_result=='y':
            print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))

    knn_acc = acc_sum / total
    knn_avg_sum += knn_acc
    print('\nkNN predict diff = %f' % knn_acc)

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
        p = [g,m]
        hy_pred.append(p)
    hy_pred = array(hy_pred)

    hNB = GaussianNB()
    hNB.fit(hy_pred, y_train)

    # test
    if print_result=='y':
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

        logTest = -1 if test_target[i] == 0 else math.log10(test_target[i])
        predTest = -1 if h == 0 else math.log10(h)
        # logTest = test_target[i]
        # predTest = h
        acc = abs(logTest - predTest)
        acc_sum += acc
        if print_result=='y':
            print('test area = %8.2f | Predict area= %8.2f | diff = %f' % (test_target[i],aPredict[0], acc))

    nb_acc = acc_sum / total
    nb_avg_sum += nb_acc
    print('\nnaïve Bayes predict diff = %f' % nb_acc)
    print()

    # process line
    # print('time = %d' % time)
    # print('x = %d' % x)

    i_per = math.floor(((x+1)/time)*100)
    k = i_per + 1
    pl = '['+ '#'*(i_per//2)+' '*((100-k)//2) + ']'
    sys.stderr.write('\r'+pl+' (%s%%)'%(i_per))
    sys.stderr.flush()
    # i_print = math.floor((i/time)*50)
    # sys.stderr.write('\r')
    # sys.stderr.write("[%-50s] %.2f%%" % ('='*i_print, i_per))
    # sys.stderr.flush()

	# end = time.time()
	# elapsed = end - start
	# elapsed = elapsed / 60



	# sys.stdout.write('Time taken: ')
	# sys.stdout.write("%d" % elapsed)
	# sys.stdout.write(' min.\t')



# avg test result
sys.stderr.write('\n\n\n===== test results =====')
sys.stderr.write('\n\ndecision tree avg diff = %f' % (tree_avg_sum / time))
sys.stderr.write('\n\nkNN avg diff = %f' % (knn_avg_sum / time))
sys.stderr.write('\n\nnaïve Bayes avg diff = %f' % (nb_avg_sum / time))
sys.stderr.write('\n\n')

print('\n\n===== test results =====')
print('\ndecision tree avg diff = %f' % (tree_avg_sum / time))
print('\nkNN avg diff = %f' % (knn_avg_sum / time))
print('\nnaïve Bayes avg diff = %f' % (nb_avg_sum / time))
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
