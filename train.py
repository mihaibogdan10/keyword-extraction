import pylab as pl
import numpy as np
import pickle

from positive_class_classifier import PositiveClassClassifier
from module_utils import DUMP_FILE, TEST_FILE


######################################################################
# Plot results
######################################################################

def plot_accuracy(x, y, plot_placement, x_legend):
    """Plot accuracy as a function of x."""
    x = np.array(x)
    y = np.array(y)
    pl.subplots_adjust(hspace=0.5)
    pl.subplot(plot_placement)
    pl.title('Classification accuracy as a function of %s' % x_legend)
    pl.xlabel('%s' % x_legend)
    pl.ylabel('Accuracy')
    pl.grid(True)
    pl.plot(x, y)

def iter_minibatchs():
    """Generator of minibatchs of examples, returns a tuple x, y.
    """

    size = 50
    corpus = [None] * size
    keywords = list(corpus)
    for index, row in enumerate(csv.reader(open(TEST_FILE))):
        title = row[1]
        #description = row[2]
        tags = row[3].split(' ')
        
        keywords[index % size] = int(self.positive_class in tags)
        corpus[index % size] = title

        if index % size == size - 1:
            yield (transformer.transform(corpus), np.asarray(keywords))



def train_PCC():
    TAG_CANDIDATES_NO = 2

    with open(DUMP_FILE, 'rb') as dump_file:
        tag_list = pickle.load(dump_file)['tag_list']

    
    for i in xrange(TAG_CANDIDATES_NO):
        count, tag = tag_list[i]
        classifier = PositiveClassClassifier('python')
        classifier.train()
        tag_list[i] = (count, tag, classifier)

    

    """
    pl.figure(1)

    # Plot accuracy evolution with #examples
    accuracy, n_examples = zip(*classifier.stats['accuracy_history'])
    plot_accuracy(n_examples, accuracy, 211, "training examples (#)")

    # Plot accuracy evolution with runtime
    accuracy, runtime = zip(*classifier.stats['runtime_history'])
    plot_accuracy(runtime, accuracy, 212, 'runtime (s)')

    pl.show()
    """

if __name__ == "__main__": 
    train_PCC()
