import csv
import pickle
from collections import defaultdict
from utils.module_utils import OVA_TRAIN_FILE, TRAIN_FILE, QUESTIONS_NO, OVA_TAGS_NO, TAGS_DUMP_FILE, print_overwrite

def create_ova_train_file():
    with open(TAGS_DUMP_FILE, 'rb') as tags_dump_file:
        tag_list = pickle.load(tags_dump_file)['tag_list']
    ova_tag_set = set([tag for count, tag in tag_list[: OVA_TAGS_NO + 1]])

    #create a train file that has a balanced number of occurences for the top tags
    test_writer = csv.writer(open(OVA_TRAIN_FILE, "w"))

    used_questions = set()
    count = defaultdict(int)
    for index, row in enumerate(csv.reader(open(TRAIN_FILE))):
        if index >= QUESTIONS_NO * 0.9:
            break

        if index % 10000 == 0:
            print_overwrite("First traversal of the train file: " + str(int(index * 100 / (0.9 * QUESTIONS_NO))) + " %")

        #the last 10% of training data is used to construct a test file
            keywords = set(row[3].split(' '))
            ova_counts = [count[keyword] for keyword in keywords & ova_tag_set]
            if ova_counts and min(ova_counts) < 500:
                for keyword in keywords:
                    count[keyword] += 1
                    used_questions.add(index)

    data = []
    for index, row in enumerate(csv.reader(open(TRAIN_FILE))):
        if index >= QUESTIONS_NO * 0.9:
            break

        if index % 10000 == 0:
            test_writer.writerows(data)
            data = []
            print_overwrite("Second traversal of the train file: " + str(int(index * 100 / (0.9 * QUESTIONS_NO))) + " %")
        
        if index in used_questions:
            data.append(row)

    test_writer.writerows(data)
    print "Created train file with", len(used_questions), "questions."

if __name__ == "__main__":
    create_ova_train_file()
