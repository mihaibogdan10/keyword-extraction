import csv
import numpy as np
import pylab as pl
import time
from collections import defaultdict
import pickle

DUMP_FILE = 'data.pickle'

try:
    with open(DUMP_FILE, 'rb') as dump_file:
        data = pickle.load(dump_file)
except IOError:
    keywords = defaultdict(int)
    for index, row in enumerate(csv.reader(open("Train_no_duplicates.csv"))):
        if index == 0:
            column_names = row
        else:
            title = row[1]
            description = row[2]
            tags = row[3].split(' ')
            for tag in tags:
                keywords[tag] += 1

    data = {'questions_no' : index}
    data['tag_list'] = sorted([(value, key) for key, value in keywords.iteritems()], reverse = True)

    with open(DUMP_FILE, 'wb') as dump_file:
        pickle.dump(data, dump_file)


def plot_tag_distribution(x, y):
    x = np.array(x)
    y = np.array(y)
    #pl.title('Tag distribution')
    pl.xlabel('Indicele etichetei (sortate dupa frecventa)')
    pl.ylabel('Frecventa etichetei')
    pl.grid(True)
    pl.plot(x, y)


occurence = [count * 1.0 / data['questions_no'] for count, tag in data['tag_list']]
index = range(1, len(data['tag_list']) + 1)
plot_tag_distribution(index[:200], occurence[:200])

# Plot accuracy evolution with runtime
#accuracy, runtime = zip(*stats['runtime_history'])
#plot_accuracy(runtime, accuracy, 212, 'runtime (s)')

#pl.show()

total_tags = sum([count for count, tag in data['tag_list']])
relative_occurence = [count * 1.0 / total_tags for count, tag in data['tag_list']]


def get_percentile(pct):
    freq_sum = 0
    
    for i in xrange(len(relative_occurence)):
        freq_sum += relative_occurence[i]
        if freq_sum > pct:
            return i, freq_sum


print get_percentile(0.90)
print data['tag_list'][:5]
