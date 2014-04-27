import csv

TRAIN_FILE = "Train_no_duplicates.csv"
TEST_FILE  = "Test_file.csv"
TAGS_DUMP_FILE = 'tag_data.pickle'
OVA_DUMP_FILE = 'ova_classifiers.pickle'

# We will feed the classifier with mini-batches of 50 documents
# This means we have at most 50 docs in memory at any time.
MINIBATCH_SIZE = 50

def iter_minibatchs(input_file, transformer, positive_class=None):
    """Generator of minibatchs of examples, returns a tuple x, y.
    """

    size = MINIBATCH_SIZE
    corpus = [None] * size
    keywords = list(corpus)
    for index, row in enumerate(csv.reader(open(input_file))):
        title = row[1]
        #description = row[2]
        tags = row[3].split(' ')
        
        if positive_class is None:
            keywords[index % size] = tags
        else:
            keywords[index % size] = int(positive_class in tags)
        corpus[index % size] = title

        if index % size == size - 1:
            yield (transformer.transform(corpus), keywords)

def create_test_file():
    test_writer = csv.writer(open(TEST_FILE, "w"))
    data = []
    for index, row in enumerate(csv.reader(open(TRAIN_FILE))):
        if index % 100000 == 0:
            print index / 100000
        if index < 5.5 * 10 ** 6:
            pass
        else:
            data.append(row)

    test_writer.writerows(data)
