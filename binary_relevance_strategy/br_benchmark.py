from __future__ import division

import numpy as np
import pickle
import os.path
import multiprocessing as mp

from itertools import izip
from time import time
from collections import defaultdict
from positive_class_classifier import PositiveClassClassifier
from utils.module_utils import TAGS_DUMP_FILE, OVA_DUMP_FILE, CULLED_TRAIN_FILE, TEST_FILE, \
    OVA_TAGS_NO, iter_documents, print_overwrite

def train_PCCs():
    def train_PCCs_worker(worker_id, tag_list, pipe, return_queue): 
        title_PCCs = {}
        description_PCCs = {}
        tag_count = {}
        classifier_tags_to_train = set()

        for count, tag in tag_list:
            tag_count[tag] = {'positives' : min(count, 2000), 'negatives' : min(count, 2000)}
            title_PCCs[tag] = PositiveClassClassifier(tag)
            description_PCCs[tag] = PositiveClassClassifier(tag)
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
                print_overwrite("Batch training: worker " + str(worker_id) + " ... " + str(doc_no) + ' docs')
            
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
                if tag not in y_train and tag_count[tag]['negatives'] > tag_count[tag]['positives']: 
                    PCC = title_PCCs[tag]
                    PCC.classifier.partial_fit(x_title_train, [0], classes = PCC.all_classes)
                    PCC = description_PCCs[tag]
                    PCC.classifier.partial_fit(x_description_train, [0], classes = PCC.all_classes)
                    
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
            title_PCCs, description_PCCs = pickle.load(ova_classifiers)
            return title_PCCs, description_PCCs 

    #the ova_classifiers weren't yet trained -> train them
    with open(TAGS_DUMP_FILE, 'rb') as tags_dump_file:
        tag_list = pickle.load(tags_dump_file)['tag_list'][: OVA_TAGS_NO]

    workers_no = mp.cpu_count()
    worker_load = OVA_TAGS_NO // workers_no

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
    
    documents_iterator = iter_documents(CULLED_TRAIN_FILE, PositiveClassClassifier.hvectorizer)
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


def benchmark_OVA_worker(documents, threshold_1, threshold_2): 
    tp = fp = fn = subsetacc = oneerror = rloss = 0
    tag_tp, tag_fp, tag_fn = [defaultdict(int) for x in xrange(3)]

    for x_title_test, x_description_test, y_test in documents:
        tp_temp = fp_temp = fn_temp = 0
        title_predictions, description_predictions = [], []
        
        for PCC in title_PCCs:
            confidence_score = PCC.classifier.decision_function(x_title_test)
            #For the binary case, confidence_score > 0 means this class would be predicted
            if confidence_score > threshold_1:
                title_predictions.append((float(confidence_score), PCC.positive_class))

        for PCC in description_PCCs:
            confidence_score = PCC.classifier.decision_function(x_description_test)
            if confidence_score > threshold_2:
                description_predictions.append((float(confidence_score), PCC.positive_class))

        title_predictions.sort(reverse = True)
        description_predictions.sort(reverse = True)

        predicted_tags = [prediction[1] for prediction in title_predictions[:5]]
        predicted_tags += [prediction[1] for prediction in description_predictions[:(5 - len(predicted_tags))]]    

        for predicted_tag in predicted_tags:
            if predicted_tag in y_test:
                tp += 1
                tp_temp += 1
                tag_tp[predicted_tag] += 1
            else:
                fp += 1
                fp_temp += 1
                tag_fp[predicted_tag] += 1

        for output_tag in y_test:
            if output_tag not in predicted_tags:
                fn += 1
                fn_temp += 1
                tag_fn[output_tag] += 1
    
        #subsetacc is used for the subset accuracy measure
        subsetacc += int(tp == len(y_test))
   
        #oneerror is used for the one error measure
        if len(predicted_tags) > 0:
            oneerror += int(predicted_tags[0] not in y_test)

        #rloss is user for the rank loss measure
        irelevant_tags_no = OVA_TAGS_NO - len(y_test)
        relevant_tags_no  = len(y_test) 
        rloss +=  fn_temp * len(predicted_tags) / (relevant_tags_no * irelevant_tags_no)

    return (tp, fp, fn, subsetacc, oneerror, rloss, tag_tp, tag_fp, tag_fn)

def benchmark_OVA(title_PCCs, description_PCCs, threshold_1, threshold_2): 
    d = {'tp' : 0, 'fp' : 0, 'fn' : 0, 'subsetacc': 0, 'oneerror': 0, 'rloss': 0} 
    tag_tp, tag_fp, tag_fn = [defaultdict(int) for x in xrange(3)]

    def add_result(result):
        tp_inc, fp_inc, fn_inc, subsetacc_inc, oneerror_inc, rloss_inc, tag_tp_inc, tag_fp_inc, tag_fn_inc = result
        d['tp'] += tp_inc
        d['fp'] += fp_inc
        d['fn'] += fn_inc
        d['subsetacc'] += subsetacc_inc
        d['oneerror'] += oneerror_inc
        d['rloss'] += rloss_inc

        for tag in tag_tp_inc:
            tag_tp[tag] += tag_tp_inc[tag]

        for tag in tag_fp_inc:
            tag_fp[tag] += tag_fp_inc[tag]

        for tag in tag_fn_inc:
            tag_fn[tag] += tag_fn_inc[tag]

    documents_iterator = iter_documents(TEST_FILE, PositiveClassClassifier.hvectorizer)
    pool = mp.Pool(processes = mp.cpu_count())
    documents = []
    documents_no = 0

    for document in documents_iterator:
        documents.append(document)
        documents_no += 1
        if len(documents) == 50:
            result = pool.apply_async(benchmark_OVA_worker, (documents, threshold_1, threshold_2))
            #If the remote call raised an exception then that exception will be reraised by get()
            add_result(result.get())
            documents = []

    result = pool.apply_async(benchmark_OVA_worker, (documents, threshold_1, threshold_2))
    add_result(result.get())
    pool.close()
    pool.join()

    print "true positives:", d['tp'], "false positives:", d['fp'], "false negatives:", d['fn']
    precision = d['tp'] / (d['tp'] + d['fp'])
    recall = d['tp'] / (d['tp'] + d['fn'])
    print "precision:", precision
    print "recall:", recall
    print "F1 score:", 2 * precision * recall / (precision + recall)
    print "Subset accuracy:", d['subsetacc'] / documents_no
    print "Hloss:", (d['fp'] + d['fn']) / (documents_no * OVA_TAGS_NO)
    print "Oneerror:", d['oneerror'] / documents_no
    print "Rloss:", d['rloss'] / documents_no
    
    tags_prec = sorted([(tag, tag_tp[tag] / (tag_tp[tag] + tag_fp[tag])) for tag in tag_tp])
    tags_rec  = sorted([(tag, tag_tp[tag] / (tag_tp[tag] + tag_fn[tag])) for tag in tag_tp])
    print len(tag_tp), len(tag_fp), len(tag_fn)
    tags_F1 = 2 * sum(prec[1] * rec[1] / (prec[1] + rec[1]) for prec, rec in izip(tags_prec, tags_rec)) / OVA_TAGS_NO
    print "Tags F1 score:", tags_F1

    return 2 * precision * recall / (precision + recall)

if __name__ == "__main__": 
    title_PCCs, description_PCCs = train_PCCs()

    #a = []
    #for i in xrange(1, 16):
    ####max_F1, max_t1, max_t2 = 0, 0, 0
    ####for threshold_1 in [x / 2 for x in xrange(6, 13)]:
    ####    for threshold_2 in [x / 2 for x in xrange(14, 20)]:
    F1 = benchmark_OVA(title_PCCs, description_PCCs, 5, 7)
    ####        if F1 > max_F1:
    ####            max_F1, max_t1, max_t2 = F1, threshold_1, threshold_2

    ####print max_F1, max_t1, max_t2
    #    a.append((i * 100, f1))

    #print a
