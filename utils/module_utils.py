import numpy as np
import pylab as pl
import csv
from sys import stdout

TRAIN_FILE = "data/Train_no_duplicates.csv"
OVA_TRAIN_FILE = "data/Train_ova.csv"
TEST_FILE  = "data/Test_file.csv"
TAGS_DUMP_FILE = "data/tag_data.pickle"
OVA_DUMP_FILE = "data/ova_classifiers.pickle"
QUESTIONS_NO = 6034195
OVA_TAGS_NO = 100

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

def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate

@static_var("prev_msg_size", 0)
def print_overwrite(current_msg):
    print ' ' * print_overwrite.prev_msg_size  + '\r',
    stdout.flush()

    print_overwrite.prev_msg_size = len(current_msg)
    print current_msg + '\r',
    stdout.flush()


def plot_distribution(x, y, xlabel, ylabel):
    x = np.array(x)
    y = np.array(y)
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.grid(True)
    pl.plot(x, y)
    pl.show
