import csv
import pickle
from collections import defaultdict
from utils.module_utils import OVA_TRAIN_FILE, TRAIN_FILE, QUESTIONS_NO, OVA_TAGS_NO, TAGS_DUMP_FILE, print_overwrite

def create_ova_train_file():
    with open(TAGS_DUMP_FILE, 'rb') as tags_dump_file:
        tag_list = pickle.load(tags_dump_file)['tag_list']
    ova_tag_set = set([tag for count, tag in tag_list[: OVA_TAGS_NO + 1]])

    #create a train file that has a balanced number of occurences for the top tags
    writer = csv.writer(open(OVA_TRAIN_FILE, "w"))
    written_rows = 0

    count = defaultdict(int)
    for index, row in enumerate(csv.reader(open(TRAIN_FILE))):
        #the last 10% of training data is used to construct a test file
        if index >= QUESTIONS_NO * 0.9:
            break

        if index % 10000 == 0:
            print_overwrite("Traversing the train file: " + str(int(index * 100 / (0.9 * QUESTIONS_NO))) + " %")

        keywords = set(row[3].split(' '))
        ova_counts = [count[keyword] for keyword in keywords & ova_tag_set]
        if ova_counts and min(ova_counts) < 1000: 
            writer.writerow(row)
            written_rows += 1
            for keyword in keywords:
                count[keyword] += 1

    print "Created train file with", written_rows, "questions."

if __name__ == "__main__":
    create_ova_train_file()
