import pylab as pl
import numpy as np
import pickle
import os.path
import multiprocessing as mp

from itertools import izip
from positive_class_classifier import PositiveClassClassifier
from utils.module_utils import TAGS_DUMP_FILE, OVA_DUMP_FILE, TRAIN_FILE, TEST_FILE, \
    DOCUMENTS_NO, OVA_TAGS_NO, iter_documents, iter_minibatches, print_overwrite

from time import time
def train_PCCs():
    def train_PCCs_worker(worker_id, tag_list, pipe, return_queue): 
        title_PCCs = {}
        description_PCCs = {}
        tag_count = {}
        classifier_tags_to_train = set()

        for count, tag in tag_list:
            tag_count[tag] = {'positives' : min(count, 5000), 'negatives' : min(count, 5000)}
            title_PCCs[tag] = PositiveClassClassifier(tag)
            #description_PCCs[tag] = PositiveClassClassifier(tag)
            classifier_tags_to_train.add(tag)

        start = time()
        train_time = 0
        read_time, read_start = 0, time()

        while True:
            read_start = time()
            pipe_data = pipe.recv()
            read_time += time() - read_start
            train_start = time()

            if pipe_data == 'STOP':
                break
            
            doc_no, x_title_train, x_description_train, y_train = pipe_data

            if doc_no % 500 == 0:
                print_overwrite("Batch training: worker " + str(worker_id) + " ... " + '(' + "%.3f" % (doc_no * 100.0 / DOCUMENTS_NO) + ' %)')
            
            #positive training for PCCs whose tags appear in y_train
            for tag in y_train:
                if tag in title_PCCs:
                    PCC = title_PCCs[tag]
                    PCC.classifier.partial_fit(x_title_train, [1], classes = PCC.all_classes)
                    #PCC = description_PCCs[tag]
                    #PCC.classifier.partial_fit(x_description_train, [1], classes = PCC.all_classes)
                    tag_count[tag]['positives'] -= 1

            tags_with_complete_negative_training = []

            #negative training for all the others
            for tag in classifier_tags_to_train:
                if tag not in y_train and tag_count[tag]['negatives'] > tag_count[tag]['positives']: 
                    PCC = title_PCCs[tag]
                    PCC.classifier.partial_fit(x_title_train, [0], classes = PCC.all_classes)
                    #PCC = description_PCCs[tag]
                    #PCC.classifier.partial_fit(x_description_train, [0], classes = PCC.all_classes)
                    
                    tag_count[tag]['negatives'] -= 1
                    if tag_count[tag]['negatives'] == 0:
                        tags_with_complete_negative_training.append(tag)

            for tag in tags_with_complete_negative_training:
                classifier_tags_to_train.remove(tag)

            train_time += time() - train_start

        print worker_id, "took", train_time, "seconds to train"
        print worker_id, "took", read_time, "seconds to read"
        print worker_id, "took", time() - start, "seconds to complete"
        return_queue.put((title_PCCs.values(), description_PCCs.values()))

    if os.path.isfile(OVA_DUMP_FILE):
        with open(OVA_DUMP_FILE, 'rb') as ova_classifiers:
            PCCs = pickle.load(ova_classifiers)
            return PCCs

    #the ova_classifiers weren't yet trained -> train them
    with open(TAGS_DUMP_FILE, 'rb') as tags_dump_file:
        tag_list = pickle.load(tags_dump_file)['tag_list'][: OVA_TAGS_NO]

    workers_no = 2 * mp.cpu_count()
    worker_load = OVA_TAGS_NO / workers_no

    jobs = []
    q = mp.Queue()
    parent_pipes = []
    for i in xrange(workers_no):
        worker_tag_list = tag_list[worker_load * i : worker_load * (i + 1)]
        child_pipe, parent_pipe = mp.Pipe(False) #False then the pipe is unidirectional
        parent_pipes.append(parent_pipe)
        p = mp.Process(target = train_PCCs_worker, args = (i, worker_tag_list, child_pipe, q))
        p.start()
        jobs.append(p)
    
    documents_iterator = iter_documents(TRAIN_FILE, PositiveClassClassifier.hvectorizer)
    for doc_no, (x_train_title, x_train_description, y_train) in enumerate(documents_iterator):
        for pipe in parent_pipes:
            pipe.send((doc_no, x_train_title, x_train_description, y_train))

    for pipe in parent_pipes:
        pipe.send('STOP')

    title_PCCs, description_PCCs = [], []
    for i in xrange(len(jobs)):
        worker_title_PCCs, worker_description_PCCs = q.get()
        title_PCCs += worker_title_PCCs
        description_PCCs += worker_description_PCCs

    for p in jobs:
        p.join()

    with open(OVA_DUMP_FILE, 'wb') as ova_classifiers:
        pickle.dump((title_PCCs, description_PCCs), ova_classifiers)
    return (title_PCCs, description_PCCs)


def benchmark_OVA_worker(documents): 
    tp = fp = fn = 0

    for x_title_test, x_description_test, y_test in documents:
        predictions = []
        for PCC in title_PCCs:
            confidence_score = PCC.classifier.decision_function(x_title_test)
            #For the binary case, confidence_score > 0 means this class would be predicted
            if confidence_score > 1.5:
                predictions.append((confidence_score, PCC.positive_class))

        #for PCC in description_PCCs:
        #    confidence_score = PCC.classifier.decision_function(x_description_test)
        #    pcc_prediction = PCC.classifier.predict(x_description_test)
        #    predictions.append((pcc_prediction, confidence_score, PCC.positive_class))


        predictions.sort(reverse = True)
        predicted_tags = set()

        for prediction in predictions:
            if len(predicted_tags) < 3:
                predicted_tags.add(prediction[1])
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
    
    return (tp, fp, fn)

def benchmark_OVA(title_PCCs, description_PCCs): 
    d = {'tp' : 0, 'fp' : 0, 'fn' : 0}

    def add_result(result):
        tp_inc, fp_inc, fn_inc = result
        d['tp'] += tp_inc
        d['fp'] += fp_inc
        d['fn'] += fn_inc

    documents_iterator = iter_documents(TEST_FILE, PositiveClassClassifier.hvectorizer)
    pool = mp.Pool(processes = 2 * mp.cpu_count())
    documents = []

    for document in documents_iterator:
        documents.append(document)
        if len(documents) == 25:
            pool.apply_async(benchmark_OVA_worker, (documents,), callback = add_result)
            documents = []

    pool.apply_async(benchmark_OVA_worker, (documents,), callback = add_result)
    pool.close()
    pool.join()

    print "true positives:", d['tp'], "false positives:", d['fp'], "false negatives:", d['fn']
    precision = d['tp'] * 1.0 / (d['tp'] + d['fp'])
    recall = d['tp'] * 1.0 / (d['tp'] + d['fn'])
    print "precision:", precision
    print "recall:", recall
    print "F1 score:", 2 * precision * recall / (precision + recall)


if __name__ == "__main__": 
    title_PCCs, description_PCCs = train_PCCs()
    benchmark_OVA(title_PCCs, description_PCCs)
