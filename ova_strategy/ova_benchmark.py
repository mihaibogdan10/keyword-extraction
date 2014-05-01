import pylab as pl
import numpy as np
import pickle
import os.path

from itertools import izip
from positive_class_classifier import PositiveClassClassifier
from utils.module_utils import TAGS_DUMP_FILE, OVA_DUMP_FILE, TRAIN_FILE, TEST_FILE, \
    DOCUMENTS_NO, OVA_TAGS_NO, iter_documents, iter_minibatches, print_overwrite

def train_PCCs():
    if os.path.isfile(OVA_DUMP_FILE):
        with open(OVA_DUMP_FILE, 'rb') as ova_classifiers:
            PCCs = pickle.load(ova_classifiers)
            return PCCs

    #the ova_classifiers weren't yet trained -> train them
    with open(TAGS_DUMP_FILE, 'rb') as tags_dump_file:
        tag_list = pickle.load(tags_dump_file)['tag_list']

    PCCs = {}
    tag_count = {}
    classifier_tags_to_train = set()
    range_len = min(OVA_TAGS_NO, len(tag_list))
    
    for i in xrange(range_len):
        count, tag = tag_list[i]
        tag_count[tag] = {'positives' : min(count, 1000), 'negatives' : min(count, 1000)}
        classifier = PositiveClassClassifier(tag)
        PCCs[tag] = classifier
        classifier_tags_to_train.add(tag)

    documents_iterator = iter_documents(TRAIN_FILE, PositiveClassClassifier.hvectorizer)

    for doc_no, (x_train, y_train) in enumerate(documents_iterator):
        print_overwrite("Batch training ... " + '(' + "%.3f" % (doc_no * 100.0 / DOCUMENTS_NO) + ' %)')
        
        #positive training for PCCs whose tags appear in y_train
        for tag in y_train:
            if tag in PCCs:
                PCC = PCCs[tag]
                PCC.classifier.partial_fit(x_train, [1], classes = PCC.all_classes)
                tag_count[tag]['positives'] -= 1

        tags_with_complete_negative_training = []

        #negative training for all the others
        for tag in classifier_tags_to_train:
            if tag not in y_train:
                PCC = PCCs[tag]
                PCC.classifier.partial_fit(x_train, [0], classes = PCC.all_classes)
                tag_count[tag]['negatives'] -= 1
                if tag_count[tag]['negatives'] == 0:
                    tags_with_complete_negative_training.append(tag)

        for tag in tags_with_complete_negative_training:
            classifier_tags_to_train.remove(tag)

    with open(OVA_DUMP_FILE, 'wb') as ova_classifiers:
        pickle.dump(PCCs.values(), ova_classifiers)
    return PCCs.values()

def benchmark_OVA(PCCs): 
    minibatch_iterator = iter_minibatches(TEST_FILE, PositiveClassClassifier.hvectorizer)
    
    TEST_BATCHES_NO = 100
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
