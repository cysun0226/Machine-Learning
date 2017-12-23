import csv
import math
import sys
import numpy as np

forestfire_file = str(sys.argv[1])
lines = []
data = []
target = []


# main
# read forestfires.csv
with open(forestfire_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        lines.append(row)
feature_names = np.array(lines[0])

for i in range(1,len(lines)):
    d = []
    d.append(int(lines[i][0])) # X
    d.append(int(lines[i][1])) # Y
    d.append(lines[i][2]) # month
    d.append(lines[i][3]) # day
    d.append(float(lines[i][4])) # FFMC
    d.append(float(lines[i][5])) # DMC
    d.append(float(lines[i][6])) # DC
    d.append(float(lines[i][7])) # ISI
    d.append(float(lines[i][8])) # temp
    d.append(float(lines[i][9])) # RH
    d.append(float(lines[i][10])) # wind
    d.append(float(lines[i][11])) # rain
    data.append(d)
    target.append(float(lines[i][12]))
