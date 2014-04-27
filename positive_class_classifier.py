import numpy as np
import time

from sklearn.feature_extraction.text import HashingVectorizer 
from sklearn.linear_model.stochastic_gradient import SGDClassifier

from module_utils import TRAIN_FILE, TEST_FILE, iter_minibatchs
from lemma_tokenizer import LemmaTokenizer


class PositiveClassClassifier(object):
    hvectorizer = HashingVectorizer(tokenizer = LemmaTokenizer(),
                                    n_features = 2 ** 19,
                                    stop_words = 'english',
                                    lowercase = True,
                                    non_negative = True)
 
    all_classes = np.array([0, 1])
    
    def __init__(self, positive_class):
        # Create an online classifier i.e. supporting `partial_fit()`
        self.classifier = SGDClassifier(loss = 'log')

        # Here we propose to learn a binary classification of the positive class
        # and all other documents
        self.positive_class = positive_class

        # structure to track accuracy history
        self.stats = {'n_train': 0, 'n_train_pos': 0, 'accuracy': 0.0, 
            'accuracy_history': [(0, 0)], 't0': time.time(), 
            'runtime_history': [(0, 0)]}

    def progress(self):
        """Report progress information, return a string."""
        duration = time.time() - self.stats['t0']
        s = "%(n_train)6d train docs (%(n_train_pos)6d positive) " % self.stats
        s += "accuracy: %(accuracy).6f " % self.stats
        s += "in %.2fs (%5d docs/s)" % (duration, self.stats['n_train'] / duration)
        return s

    def train(self):
        TRAIN_BATCHES_NO = 100
        minibatch_iterator = iter_minibatchs(TRAIN_FILE, self.hvectorizer, self.positive_class)
 
        # Main loop : iterate on mini-batchs of examples
        for i, (x_train, y_train) in enumerate(minibatch_iterator):
            # update estimator with examples in the current mini-batch
            self.classifier.partial_fit(x_train, y_train, classes=self.all_classes)

            # accumulate test accuracy stats
            self.stats['n_train'] += x_train.shape[0]
            self.stats['n_train_pos'] += sum(y_train)
            self.stats['accuracy'] = self.score()
            self.stats['accuracy_history'].append((self.stats['accuracy'], 
                                                   self.stats['n_train']))
            self.stats['runtime_history'].append((self.stats['accuracy'],
                                                  time.time() - self.stats['t0']))
            if i % 10 == 0:
                print(self.progress())
            if i >= TRAIN_BATCHES_NO - 1:
                break

    def score(self): 
        TEST_BATCHES_NO = 20
        minibatch_iterator = iter_minibatchs(TEST_FILE, self.hvectorizer, self.positive_class)
        score = 0
        
        for i, (x_test, y_test) in enumerate(minibatch_iterator):
            y_test = np.asarray(y_test)
            score += self.classifier.score(x_test, y_test)

            if i >= TEST_BATCHES_NO - 1:
                break

        return score / TEST_BATCHES_NO
