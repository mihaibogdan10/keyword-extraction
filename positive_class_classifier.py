import csv
import numpy as np
import time

from sklearn.feature_extraction.text import HashingVectorizer 
from sklearn.linear_model.stochastic_gradient import SGDClassifier

from module_utils import TRAIN_FILE, TEST_FILE
from lemma_tokenizer import LemmaTokenizer


class PositiveClassClassifier(object):
    hvectorizer = HashingVectorizer(tokenizer=LemmaTokenizer(),
                                    n_features=2 ** 21,
                                    stop_words='english',
                                    lowercase=True,
                                    non_negative=True)
 
    all_classes = np.array([0, 1])
    
    def __init__(self, positive_class):
        # Create an online classifier i.e. supporting `partial_fit()`
        self.classifier = SGDClassifier()

        # Here we propose to learn a binary classification between the positive class
        # and all other documents
        self.positive_class = positive_class

        # structure to track accuracy history
        self.stats = {'n_train': 0, 'n_test': 0, 'n_train_pos': 0, 'n_test_pos': 0,
             'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
             'runtime_history': [(0, 0)]}


    def iter_minibatchs(self, size, input_file, transformer=hvectorizer):
        """Generator of minibatchs of examples, returns a tuple X, y.
        """

        corpus = []
        keywords = []
        for index, row in enumerate(csv.reader(open(input_file))):
            if index == 0:
                column_names = row 
            elif len(corpus) == size:
                yield (transformer.transform(corpus), np.asarray(keywords))
                corpus = []
                keywords = []
            else:
                title = row[1]
                description = row[2]
                tags = row[3].split(' ')
                if self.positive_class in tags:
                    keywords.append(1)
                else:
                    keywords.append(0)
                corpus.append(title)

    def progress(self):
        """Report progress information, return a string."""
        duration = time.time() - self.stats['t0']
        s = "%(n_train)6d train docs (%(n_train_pos)6d positive) " % self.stats
        s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % self.stats
        s += "accuracy: %(accuracy).6f " % self.stats
        s += "in %.2fs (%5d docs/s)" % (duration, self.stats['n_train'] / duration)
        return s

    def train(self):

        # We will feed the classifier with mini-batches of 50 documents; this means
        # we have at most 50 docs in memory at any time.
        MINIBATCH_SIZE = 50
        TRAIN_BATCHES_NO = 50
        minibatch_iterator = self.iter_minibatchs(MINIBATCH_SIZE, TRAIN_FILE)

        # First we hold out a number of examples to estimate accuracy
        X_test, y_test = self.iter_minibatchs(1000, TRAIN_FILE).next()
        self.stats['n_test'] += len(y_test)
        self.stats['n_test_pos'] += sum(y_test)

        print("Test set is %d documents (%d positive)" % (len(y_test), sum(y_test)))
 
        # Main loop : iterate on mini-batchs of examples
        for i, (X_train, y_train) in enumerate(minibatch_iterator):
            # update estimator with examples in the current mini-batch
            self.classifier.partial_fit(X_train, y_train, classes=self.all_classes)

            # accumulate test accuracy stats
            self.stats['n_train'] += X_train.shape[0]
            self.stats['n_train_pos'] += sum(y_train)
            self.stats['accuracy'] = self.classifier.score(X_test, y_test)
            self.stats['accuracy_history'].append((self.stats['accuracy'], 
                                                   self.stats['n_train']))
            self.stats['runtime_history'].append((self.stats['accuracy'],
                                                  time.time() - self.stats['t0']))
            if i % 10 == 0:
                print(self.progress())
            if i >= TRAIN_BATCHES_NO:
                break
