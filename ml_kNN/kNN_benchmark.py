from __future__ import division

import numpy as np
import pickle
import os.path
import multiprocessing as mp

from itertools import izip
from collections import defaultdict
from scipy.sparse import vstack
from binary_relevance_strategy.positive_class_classifier import PositiveClassClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from utils.module_utils import CULLED_TRAIN_FILE, TEST_FILE, TAGS_NO, iter_documents

def benchmark_kNN_worker(documents): 
    tp = fp = fn = subsetacc = oneerror = rloss = 0
    tag_tp, tag_fp, tag_fn = [defaultdict(int) for x in xrange(3)]

    for x_title_test, x_description_test, y_test in documents:
        title_neighbours = title_neigh.kneighbors(title_tfidf.transform(x_title_test))
        description_neighbours = description_neigh.kneighbors(description_tfidf.transform(x_description_test))

        tag_votes = defaultdict(int)
        for index in title_neighbours[1][0]:
            tag_list = y[index]
            for tag in tag_list[0]:
                tag_votes[tag] += 2

        for index in description_neighbours[1][0]:
            tag_list = y[index]
            for tag in tag_list[0]:
                tag_votes[tag] += 1

        sorted_tag_votes = sorted([(votes, tag) for tag, votes in tag_votes.iteritems()], reverse = True)
        predicted_tags = [tag for votes, tag in sorted_tag_votes[:3]]

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
        irelevant_tags_no = TAGS_NO - len(y_test)
        relevant_tags_no  = len(y_test) 

    return (tp, fp, fn, subsetacc, tag_tp, tag_fp, tag_fn)  

def benchmark_kNN(title_neigh, description_neigh, y, title_tfidf, description_tfidf):
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
            result = pool.apply_async(benchmark_kNN_worker, (documents,), callback = add_result)
            #If the remote call raised an exception then that exception will be reraised by get()
            #add_result(result.get())
            documents = []

    result = pool.apply_async(benchmark_kNN_worker, (documents,), callback = add_result)
    #add_result(result.get())
    pool.close()
    pool.join()

    print "true positives:", d['tp'], "false positives:", d['fp'], "false negatives:", d['fn']
    precision = d['tp'] / (d['tp'] + d['fp'])
    recall = d['tp'] / (d['tp'] + d['fn'])
    print "precision:", precision
    print "recall:", recall
    print "F1 score:", 2 * precision * recall / (precision + recall)
    print "Subset accuracy:", d['subsetacc'] / documents_no
    print "Hloss:", (d['fp'] + d['fn']) / documents_no
    
    tags_prec = sorted([(tag, tag_tp[tag] / (tag_tp[tag] + tag_fp[tag])) for tag in tag_tp])
    tags_rec  = sorted([(tag, tag_tp[tag] / (tag_tp[tag] + tag_fn[tag])) for tag in tag_tp])
    print len(tag_tp), len(tag_fp), len(tag_fn)
    tags_F1 = 2 * sum(prec[1] * rec[1] / (prec[1] + rec[1]) for prec, rec in izip(tags_prec, tags_rec)) / TAGS_NO
    print "Tags F1 score:", tags_F1

    return 2 * precision * recall / (precision + recall)

if __name__ == "__main__": 
    title_tfidf = TfidfTransformer()
    description_tfidf = TfidfTransformer()
    documents_iterator = iter_documents(CULLED_TRAIN_FILE, PositiveClassClassifier.hvectorizer)

    for doc_no, (x_train_title, x_train_description, y_train) in enumerate(documents_iterator):
        if doc_no == 0:
            x_title, x_description, y = x_train_title, x_train_description, [y_train]
        else:
            x_title = vstack([x_title, x_train_title])
            x_description = vstack([x_description, x_train_description])
            y.append([y_train])


    #Note: fitting on sparse input will override the setting of the `algorithm` parameter, using brute force
    title_neigh = NearestNeighbors(n_neighbors = 13)
    title_neigh.fit(title_tfidf.fit_transform(x_title))
    
    description_neigh = NearestNeighbors(n_neighbors = 13)
    description_neigh.fit(description_tfidf.fit_transform(x_description))

    benchmark_kNN(title_neigh, description_neigh, y, title_tfidf, description_tfidf)

