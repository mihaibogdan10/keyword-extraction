"""Python script to analyze tags: distribution, etc.
"""

import csv
import time
import pickle

from collections import defaultdict
from utils.module_utils import TAGS_DUMP_FILE, TRAIN_FILE, plot_distribution

def get_percentile(pct, array):
    freq_sum = 0
    
    for i in xrange(len(array)):
        freq_sum += array[i]
        if freq_sum > pct:
            return i, freq_sum

def get_tags_data():
    try:
        with open(TAGS_DUMP_FILE, 'rb') as dump_file:
            data = pickle.load(dump_file)
    except IOError:
        keywords = defaultdict(int)
        for index, row in enumerate(csv.reader(open(TRAIN_FILE))):
            title = row[1]
            description = row[2]
            tags = row[3].split(' ')
            for tag in tags:
                keywords[tag] += 1

        data = {'questions_no' : index}
        data['tag_list'] = sorted([(value, key) for key, value in keywords.iteritems()], reverse = True)

        with open(TAGS_DUMP_FILE, 'wb') as dump_file:
            pickle.dump(data, dump_file)


    occurence = [count * 1.0 / data['questions_no'] for count, tag in data['tag_list']]
    index = range(1, len(data['tag_list']) + 1)
    plot_distribution(index[:200], occurence[:200],
                      'Indicele etichetei (sortate dupa frecventa)',
                      'Frecventa etichetei')

    total_tags = sum([count for count, tag in data['tag_list']])
    relative_occurence = [count * 1.0 / total_tags for count, tag in data['tag_list']]

    print get_percentile(0.40, relative_occurence)
    print data['tag_list'][2000:2005]

if __name__ == "__main__":
    get_tags_data()
