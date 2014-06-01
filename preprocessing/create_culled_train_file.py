import csv
import pickle
import random

from utils.module_utils import TAGS_DUMP_FILE, CULLED_TRAIN_FILE, CULLED_TRAIN_FILE_SIZE, \
    DOCUMENTS_NO, TRAIN_FILE, OVA_TAGS_NO, print_overwrite

if __name__ == "__main__":
    with open(TAGS_DUMP_FILE, 'rb') as tags_dump_file:
        tag_list = pickle.load(tags_dump_file)['tag_list']

    tags_set = set([tag for count, tag in tag_list[: OVA_TAGS_NO + 1]])
    tags_examples = {tag: 100 for tag in tags_set}

    #create a train file that has at least 100 examples for each tag
    #also, randomly add some documents, in order to preserve tag distribution

    writer = csv.writer(open(CULLED_TRAIN_FILE, "w"))
    written_rows = 0
    skipped_documents = set()
    
    for index, row in enumerate(csv.reader(open(TRAIN_FILE))):
        #the last 10% of training data is used to construct a test file
        if index >= DOCUMENTS_NO * 0.9:
            break

        if index % 10000 == 0:
            print_overwrite("Traversing the train file: " + str(int(index * 100 / (0.9 * DOCUMENTS_NO))) + " %")
         
        keywords = set(row[3].split(' '))
        if keywords & tags_set:
            writer.writerow(row)
            written_rows += 1

            for keyword in keywords:
                if keyword in tags_examples:
                    tags_examples[keyword] -= 1
                    if tags_examples[keyword] == 0:
                        tags_set -= set([keyword])
        else:
            skipped_documents.add(index)

    
    print "All tags will appear at least 100 times in the first", written_rows, "documents"
    random_documents_no = CULLED_TRAIN_FILE_SIZE - written_rows
    random_documents = sorted(random.sample(skipped_documents, random_documents_no), reverse = True)

   
    for index, row in enumerate(csv.reader(open(TRAIN_FILE))):
        if index % 10000 == 0:
            print_overwrite("Traversing the train file: " + str(int(index * 100 / (0.9 * DOCUMENTS_NO))) + " %")

        if index == random_documents[-1]:
            writer.writerow(row)
            random_documents.pop()
            written_rows += 1
            
            if not random_documents:
                break

    print "A total of", written_rows, "documents was picked for the culled training file."
