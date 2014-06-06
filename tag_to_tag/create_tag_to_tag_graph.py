import csv
import pickle
import random

from itertools import combinations
from collections import defaultdict
from utils.module_utils import TRAIN_FILE, DOCUMENTS_NO, TAG_GRAPH_FILE, print_overwrite

if __name__ == "__main__":
    tag_graph = defaultdict(dict)
    tag_counts = defaultdict(int)

    for index, row in enumerate(csv.reader(open(TRAIN_FILE))):
        #the last 10% of training data is used to construct a test file
        if index >= DOCUMENTS_NO * 0.9:
            break

        if index % 10000 == 0:
            print_overwrite("Traversing the train file: " + str(int(index * 100 / (0.9 * DOCUMENTS_NO))) + " %")
         
        tags = set(row[3].split(' '))
        for tag in tags:
            tag_counts[tag] += 1

        for tag1, tag2 in combinations(tags, 2):
            if tag2 in tag_graph[tag1]:
                tag_graph[tag1][tag2] += 1
            else:
                tag_graph[tag1][tag2] = 1

    
    dump_dict = {'tag_graph': tag_graph, 'tag_counts': tag_counts}
    with open(TAG_GRAPH_FILE, 'wb') as tag_graph_file:
        pickle.dump(dump_dict, tag_graph_file)
