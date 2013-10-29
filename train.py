import csv
import numpy as np
import pylab as pl
import time

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer 
from sklearn.linear_model.stochastic_gradient import SGDClassifier

DOCUMENTS_COUNT = 20000

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        allowed_chars = '[0-9]|[a-z]|[\$\#\%\&\*\-]'
        self.tkn = RegexpTokenizer( r'(?:{0})*[a-z]+(?:{0})*'.format(allowed_chars))
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in self.tkn.tokenize(doc)]

hvectorizer = HashingVectorizer(tokenizer=LemmaTokenizer(),
                                n_features=2 ** 21,
                                stop_words='english',
                                lowercase=True,
                                non_negative=True)

# Create an online classifier i.e. supporting `partial_fit()`
classifier = SGDClassifier()

# Here we propose to learn a binary classification between the positive class
# and all other documents
all_classes = np.array([0, 1])
positive_class = 'javascript'



def iter_minibatchs(size, transformer=hvectorizer):
    """Generator of minibatchs of examples, returns a tuple X, y.
    """

    corpus = []
    keywords = []
    for index, row in enumerate(csv.reader(open("Train_no_duplicates.csv"))):
        if index == 0:
            column_names = row 
        elif len(corpus) == size:
            yield (transformer.transform(corpus), np.asarray(keywords, dtype=int))
            corpus = []
            keywords = []
        else:
            title = row[1]
            description = row[2]
            tags = row[3].split(' ')
            if positive_class in tags:
                keywords.append(1)
            else:
                keywords.append(0)
            corpus.append(title)

# structure to track accuracy history
stats = {'n_train': 0, 'n_test': 0, 'n_train_pos': 0, 'n_test_pos': 0,
         'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
         'runtime_history': [(0, 0)]}

# We will feed the classifier with mini-batches of 100 documents; this means
# we have at most 100 docs in memory at any time.
MINIBATCH_SIZE = 100
minibatch_iterator = iter_minibatchs(MINIBATCH_SIZE)

# First we hold out a number of examples to estimate accuracy
TEST_BATCHES_NO = 10
TRAIN_BATCHES_NO = 90

X_test, y_test = minibatch_iterator.next()
for index in xrange(TEST_BATCHES_NO - 1):
    X_test_temp, y_test_temp = minibatch_iterator.next()
    print X_test.asarray()
    X_test = np.concatenate([X_test, X_test_temp])
    y_test = np.concatenate([y_test, y_test_temp])
    
stats['n_test'] += len(y_test)
stats['n_test_pos'] += sum(y_test)

print("Test set is %d documents (%d positive)" % (len(y_test), sum(y_test)))

def progress(stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
    s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % stats
    s += "accuracy: %(accuracy).3f " % stats
    s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
    return s

# Main loop : iterate on mini-batchs of examples
for i, (X_train, y_train) in enumerate(minibatch_iterators):
    # update estimator with examples in the current mini-batch
    classifier.partial_fit(X_train, y_train, classes=all_classes)
    # accumulate test accuracy stats
    stats['n_train'] += X_train.shape[0]
    stats['n_train_pos'] += sum(y_train)
    stats['accuracy'] = classifier.score(X_test, y_test)
    stats['accuracy_history'].append((stats['accuracy'], stats['n_train']))
    stats['runtime_history'].append((stats['accuracy'],
                                     time.time() - stats['t0']))
    if i % 10 == 0:
        print(progress(stats))

###############################################################################
# Plot results
###############################################################################


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

pl.figure(1)

# Plot accuracy evolution with #examples
accuracy, n_examples = zip(*stats['accuracy_history'])
plot_accuracy(n_examples, accuracy, 211, "training examples (#)")

# Plot accuracy evolution with runtime
accuracy, runtime = zip(*stats['runtime_history'])
plot_accuracy(runtime, accuracy, 212, 'runtime (s)')

pl.show()
