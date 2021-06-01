import pandas as pd
import numpy as np
import heapq
import math
import time

import gmpy2
from gmpy2 import mpz
import re

from sklearn import tree

import cProfile

from random import randint, sample, seed

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold # import KFold
# Read in the dataset
monk1 = pd.DataFrame(pd.read_csv('../data/preprocessed/monk1-train.csv',sep=";"))
monkog = pd.DataFrame(pd.read_csv('../data/preprocessed/monks-1.train'))


def test_accuracy_onefold(file, lambs):
    """
    Run CART and OSDT
    use all data, only training accuracy
    :param X:
    :param y:
    :param lambs:
    :param file_CART:
    :param file_OSDT:
    :return:
    """
    #with open(file_CART, 'a+') as f:
        #f.write(";".join(["fold", "lamb", "nleaves", "trainaccu_CART", "testaccu_CART"]) + '\n')
    #with open(file_OSDT, 'a+') as f:
        #f.write(";".join(["fold", "lamb", "nleaves", "trainaccu_OSDT", "testaccu_OSDT", "totaltime", "time_c", "leaves_c"]) + '\n')
    for lamb in lambs:

        #file_train = file + '.train' + str(1) + '.csv'
        #file_test = file + '.test' + str(1) + '.csv'

        #file_train = file
        #file_test = file

        file_train = file + '.train'
        file_test = file + '.test'

        data_train = pd.DataFrame(pd.read_table(file_train,sep=" "))
        data_test = pd.DataFrame(pd.read_table(file_test,sep=" "))
        #data_train = pd.DataFrame(pd.read_csv(file_train, sep=";"))
        #data_test = pd.DataFrame(pd.read_csv(file_test, sep=";"))

        data_train = data_train.iloc[:, 1:-1]
        data_test = data_test.iloc[:, 1:-1]

        X_train = data_train.values[:, 1:]
        y_train = data_train.values[:, 0]

        #X_train = data_train.values[:, :-1]
        #y_train = data_train.values[:, -1]

        X_test = data_test.values[:, 1:]
        y_test = data_test.values[:, 0]

        #X_test = data_test.values[:, :-1]
        #y_test = data_test.values[:, -1]


        # CART
        # clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=max(math.ceil(lamb * 2 * len(y_train)), 2),
        #                                   min_samples_leaf=math.ceil(lamb * len(y_train)),
        #                                   max_leaf_nodes=math.floor(1 / (2 * lamb)),
        #                                   min_impurity_decrease=lamb
        #                                   )
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        # fn = ['a1_1', 'a1_2', 'a2_1', 'a2_2', 'a3_1' , 'a4_1' ,'a4_2' , 'a5_1', 'a5_2', 'a5_3', 'a6_1']
        # #fn = ['a1', 'a2','a3', 'a4', 'a5', 'a6']
        # cn = ['0', '1']
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
        # tree.plot_tree(clf,
        #                feature_names=fn,
        #                class_names=cn,
        #                filled=True);

        #fig.savefig('imagenameEN.png')
        #text_representation = tree.export_text(clf)
        #print(text_representation)

        nleaves_CART = (clf.tree_.node_count + 1) / 2
        #nleaves_CART = clf.tree_.node_count
        trainaccu_CART = clf.score(X_train, y_train)
        testaccu_CART = clf.score(X_test, y_test)

        print(">>>>>>>>>>>>>>>>> nleaves_CART:", nleaves_CART)
        print(">>>>>>>>>>>>>>>>> trainaccu_CART:", trainaccu_CART)
        print(">>>>>>>>>>>>>>>>> testaccu_CART:", testaccu_CART)

        # with open(file_CART, 'a+') as f:
            #f.write(";".join([str('NA'), str(lamb), str(nleaves_CART), str(trainaccu_CART), str(testaccu_CART)]) + '\n')


test_accuracy_onefold('../data/preprocessed/monks-1', lambs=[0.025])
#test_accuracy_onefold('../data/preprocessed/monk1-train.csv', lambs=[0.025])
#lambs=[0.1, 0.05, 0.025]
