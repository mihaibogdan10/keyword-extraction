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

    title_PCCs = {}
    description_PCCs = {}
    tag_count = {}
    classifier_tags_to_train = set()
    range_len = min(OVA_TAGS_NO, len(tag_list))
    
    for i in xrange(range_len):
        count, tag = tag_list[i]
        tag_count[tag] = {'positives' : min(count, 1000), 'negatives' : min(count, 1000)}
        title_PCCs[tag] = PositiveClassClassifier(tag)
        description_PCCs[tag] = PositiveClassClassifier(tag)
        classifier_tags_to_train.add(tag)

    documents_iterator = iter_documents(TRAIN_FILE, PositiveClassClassifier.hvectorizer)

    for doc_no, (x_title_train, x_description_train, y_train) in enumerate(documents_iterator):
        print_overwrite("Batch training ... " + '(' + "%.3f" % (doc_no * 100.0 / DOCUMENTS_NO) + ' %)')
        
        #positive training for PCCs whose tags appear in y_train
        for tag in y_train:
            if tag in title_PCCs:
                PCC = title_PCCs[tag]
                PCC.classifier.partial_fit(x_title_train, [1], classes = PCC.all_classes)
                PCC = description_PCCs[tag]
                PCC.classifier.partial_fit(x_description_train, [1], classes = PCC.all_classes)
                tag_count[tag]['positives'] -= 1

        tags_with_complete_negative_training = []

        #negative training for all the others
        for tag in classifier_tags_to_train:
            if tag not in y_train: 
                PCC = title_PCCs[tag]
                PCC.classifier.partial_fit(x_title_train, [0], classes = PCC.all_classes)
                PCC = description_PCCs[tag]
                PCC.classifier.partial_fit(x_description_train, [0], classes = PCC.all_classes)
                
                tag_count[tag]['negatives'] -= 1
                if tag_count[tag]['negatives'] == 0:
                    tags_with_complete_negative_training.append(tag)

        for tag in tags_with_complete_negative_training:
            classifier_tags_to_train.remove(tag)

    with open(OVA_DUMP_FILE, 'wb') as ova_classifiers:
        pickle.dump((title_PCCs.values(), description_PCCs.values()), ova_classifiers)
    return (title_PCCs.values(), description_PCCs.values())

def benchmark_OVA(title_PCCs, description_PCCs): 
    documents_iterator = iter_documents(TEST_FILE, PositiveClassClassifier.hvectorizer)
    
    tp = fp = fn = 0

    for x_title_test, x_description_test, y_test in documents_iterator:
        predictions = []
        for PCC in title_PCCs:
            confidence_score = PCC.classifier.decision_function(x_title_test)
            pcc_prediction = PCC.classifier.predict(x_title_test)
            predictions.append((pcc_prediction, confidence_score, PCC.positive_class))

        #for PCC in description_PCCs:
        #    confidence_score = PCC.classifier.decision_function(x_description_test)
        #    pcc_prediction = PCC.classifier.predict(x_description_test)
        #    predictions.append((pcc_prediction, confidence_score, PCC.positive_class))


        predictions.sort(reverse = True)
        predicted_tags = set()

        #print len(set([p[2] for p in predictions[:5]]) & set(y_test))

        for prediction in predictions:
            if len(predicted_tags) < 3 and prediction[1][0] > 1.5:
                predicted_tags.add(prediction[2])
            else:
                break

        for predicted_tag in predicted_tags:
            if predicted_tag in y_test:
                tp += 1
            else:
                fp += 1

        for output_tag in y_test:
            if output_tag not in predicted_tags:
                fn += 1

    print "true positives:", tp, "false positives:", fp, "false negatives:", fn
    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    print "precision:", precision
    print "recall:", recall
    print "F1 score:", 2 * precision * recall / (precision + recall)

if __name__ == "__main__": 
    title_PCCs, description_PCCs = train_PCCs()
    benchmark_OVA(title_PCCs, description_PCCs)
