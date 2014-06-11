from __future__ import division

import numpy as np
import pickle
import os.path
import multiprocessing as mp

from collections import defaultdict
from binary_relevance_strategy.positive_class_classifier import PositiveClassClassifier
from sklearn.tree import DecisionTreeClassifier
from utils.module_utils import CULLED_TRAIN_FILE, TEST_FILE, DT_DUMP_FILE, \
    TAGS_DUMP_FILE, iter_documents, print_overwrite

def train_DTs(tag_id):
    if os.path.isfile(DT_DUMP_FILE):
        with open(DT_DUMP_FILE, 'rb') as dt_file:
            title_dt, description_dt = pickle.load(dt_file)
            return dt

    #the decision trees weren't yet trained -> train them
    documents_iterator = iter_documents(CULLED_TRAIN_FILE, PositiveClassClassifier.hvectorizer)
    title_X, description_X, y = [], [], []

    for doc_no, (x_train_title, x_train_description, y_train) in enumerate(documents_iterator):
        title_X.append(x_train_title.toarray()[0])
        description_X.append(x_train_description.toarray()[0])
        y.append([tag_id[tag] for tag in y_train])
    
    title_dt, description_dt = [DecisionTreeClassifier(max_depth = 5) for x in xrange(2)] 
    title_dt.fit(title_X, y)
    description_dt.fit(description_X, y)
    return title_dt, description_dt
    
def benchmark_DT_worker(documents): 
    tp = fp = fn = subsetacc = oneerror = rloss = 0
    tag_tp, tag_fp, tag_fn = [defaultdict(int) for x in xrange(3)]

    for x_title_test, x_description_test, y_test in documents:
        title_predictions = title_dt.predict(x_title_test.toarray()[0])
        description_predictions = description_dt.predict(x_description_test.toarray()[0])
        
        print title_predictions
        print description_predictions
        print [tag_id[tag] for tag in y_test]
        predicted_tags = []

        for predicted_tag in predicted_tags:
            if predicted_tag in y_test:
                tp += 1
                tag_tp[predicted_tag] += 1
            else:
                fp += 1
                tag_fp[predicted_tag] += 1

        for output_tag in y_test:
            if output_tag not in predicted_tags:
                fn += 1
                tag_fn[output_tag] += 1
    
        #subsetacc is used for the subset accuracy measure
        subsetacc += int(tp == len(y_test))
   
        #rloss is user for the rank loss measure
        irelevant_tags_no = len(tag_id) - len(y_test)
        relevant_tags_no  = len(y_test) 

    return (tp, fp, fn, subsetacc, tag_tp, tag_fp, tag_fn)  

def benchmark_DT(title_dt, description_dt, tag_id):
    d = {'tp' : 0, 'fp' : 0, 'fn' : 0, 'subsetacc': 0} 
    tag_tp, tag_fp, tag_fn = [defaultdict(int) for x in xrange(3)]

    def add_result(result):
        tp_inc, fp_inc, fn_inc, subsetacc_inc, tag_tp_inc, tag_fp_inc, tag_fn_inc = result
        d['tp'] += tp_inc
        d['fp'] += fp_inc
        d['fn'] += fn_inc
        d['subsetacc'] += subsetacc_inc

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
        if len(documents) == 25:
            result = pool.apply_async(benchmark_DT_worker, (documents,))
            #If the remote call raised an exception then that exception will be reraised by get()
            add_result(result.get())
            documents = []

    result = pool.apply_async(benchmark_DT_worker, (documents,))
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
    print "Hloss:", (d['fp'] + d['fn']) / (documents_no * len(tag_id))
    
    tags_prec = sorted([(tag, tag_tp[tag] / (tag_tp[tag] + tag_fp[tag])) for tag in tag_tp])
    tags_rec  = sorted([(tag, tag_tp[tag] / (tag_tp[tag] + tag_fn[tag])) for tag in tag_tp])
    print len(tag_tp), len(tag_fp), len(tag_fn)
    tags_F1 = 2 * sum(prec[1] * rec[1] / (prec[1] + rec[1]) for prec, rec in izip(tags_prec, tags_rec)) / len(tag_id)
    print "Tags F1 score:", tags_F1

    return 2 * precision * recall / (precision + recall)

if __name__ == "__main__":
    with open(TAGS_DUMP_FILE, 'rb') as tags_dump_file:
        tag_list = [tag for count, tag in pickle.load(tags_dump_file)['tag_list']]
    
    tag_id, index = {}, 1
    for tag in tag_list:
        tag_id[tag] = index
        index += 1

    title_dt, description_dt = train_DTs(tag_id)
    benchmark_DT(title_dt, description_dt, tag_id)
