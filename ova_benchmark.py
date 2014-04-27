import pylab as pl
import numpy as np
import pickle
import os.path

from itertools import izip
from positive_class_classifier import PositiveClassClassifier
from module_utils import TAGS_DUMP_FILE, OVA_DUMP_FILE, TEST_FILE, iter_minibatchs

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

def train_PCCs():
    if os.path.isfile(OVA_DUMP_FILE):
        with open(OVA_DUMP_FILE, 'rb') as ova_classifiers:
            PCCs = pickle.load(ova_classifiers)
            return PCCs

    #the ova_classifiers weren't yet trained -> train them
    TAG_CANDIDATES_NO = 100

    with open(TAGS_DUMP_FILE, 'rb') as tags_dump_file:
        tag_list = pickle.load(tags_dump_file)['tag_list']

    PCCs = []
    for i in xrange(TAG_CANDIDATES_NO):
        count, tag = tag_list[i]
        print "Training positive class classifier for the tag:", tag
        classifier = PositiveClassClassifier(tag)
        classifier.train()
        PCCs.append(classifier)

    with open(OVA_DUMP_FILE, 'wb') as ova_classifiers:
        pickle.dump(PCCs, ova_classifiers)
    return PCCs

def benchmark_OVA(PCCs): 
    minibatch_iterator = iter_minibatchs(TEST_FILE, PositiveClassClassifier.hvectorizer)
    
    TEST_BATCHES_NO = 1
    tp = fp = fn = 0

    for batch_no, (x_test, y_test) in enumerate(minibatch_iterator):
        predictions = [[] for x in xrange(x_test.size)]
        for pcc_index, pcc in enumerate(PCCs):
            confidence_scores = list(pcc.classifier.decision_function(x_test))
            pcc_predictions = list(pcc.classifier.predict(x_test))
            for i, (confidence_score, pcc_prediction) in enumerate(izip(confidence_scores, pcc_predictions)):
                predictions[i].append((pcc_prediction, confidence_score, pcc_index))

        for i in xrange(len(predictions)):
            predictions[i].sort(reverse = True)

        for i in xrange(len(y_test)):
            predicted_tags = predictions[i][:3]
            for prediction in predicted_tags:
                if PCCs[prediction[2]].positive_class in y_test[i]:
                    tp += 1
                else:
                    fp += 1

            for output_tag in y_test[i]:
                if output_tag not in predicted_tags:
                    fn += 1

        if batch_no >= TEST_BATCHES_NO - 1:
            break

    print "true positives:", tp, "false positives:", fp, "false negatives:", fn
    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    print "precision:", precision
    print "recall:", recall
    print "F1 score:", 2 * precision * recall / (precision + recall)

if __name__ == "__main__": 
    PCCs = train_PCCs()
    benchmark_OVA(PCCs)
