import pandas as pd
import numpy as np
import heapq
import math
import time
from itertools import product

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

        #file_train = file + '.trainTK' + '.csv'
        #file_test = file + '.test' + str(1) + '.csv'

        file_train = file
        #file_test = file

        #file_train = file + '.train'
        #file_test = file + '.test'

        #data_train = pd.DataFrame(pd.read_table(file_train,sep=" "))
        #data_test = pd.DataFrame(pd.read_table(file_test,sep=" "))
        data_train = pd.DataFrame(pd.read_csv(file_train, sep=";"))
        #data_test = pd.DataFrame(pd.read_csv(file_test, sep=";"))

        #data_train = data_train.iloc[:, 1:-1]
        #data_test = data_test.iloc[:, 1:-1]

        #X_train = data_train.values[:, 1:]
        #y_train = data_train.values[:, 0]

        X_train = data_train.values[:, :-1]
        y_train = data_train.values[:, -1]

        #X_test = data_test.values[:, 1:]
        #y_test = data_test.values[:, 0]

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
        fn = ['Outlook', 'Temp', 'Humidity', 'Wind']
        cn = ['No', 'Yes']
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
        tree.plot_tree(clf,feature_names=fn,class_names=cn,filled=True);

        fig.savefig('imagenamept.png')
        #text_representation = tree.export_text(clf)
        #print(text_representation)

        nleaves_CART = (clf.tree_.node_count + 1) / 2
        #nleaves_CART = clf.tree_.node_count
        trainaccu_CART = clf.score(X_train, y_train)
        #testaccu_CART = clf.score(X_test, y_test)

        print(">>>>>>>>>>>>>>>>> nleaves_CART:", nleaves_CART)
        print(">>>>>>>>>>>>>>>>> trainaccu_CART:", trainaccu_CART)
        #print(">>>>>>>>>>>>>>>>> testaccu_CART:", testaccu_CART)

        # with open(file_CART, 'a+') as f:
            #f.write(";".join([str('NA'), str(lamb), str(nleaves_CART), str(trainaccu_CART), str(testaccu_CART)]) + '\n')


#test_accuracy_onefold('../data/preprocessed/monks-1', lambs=[0.025])
# test_accuracy_onefold('../data/preprocessed/playtennis.csv', lambs=[0.025])
#lambs=[0.1, 0.05, 0.025]


def rule_vand(tt1, tt2):
    vand = tt1 & tt2
    # subtract 1 to remove leading ones
    cnt = gmpy2.popcount(vand) - 1
    return vand, cnt

def make_all_ones(length):
    ones = pow(2, length) - 1
    default_tt = mpz(ones)
    return default_tt

def count_ones(tt):
    return gmpy2.popcount(tt) - 1

def rule_vectompz(vec):
    return mpz('1' + "".join([str(i) for i in vec]), 2)

file_train = '../data/preprocessed/playtennis.csv'

data_train = pd.DataFrame(pd.read_csv(file_train, sep=";"))
# data_test = pd.DataFrame(pd.read_csv(file_test, sep=";"))

x = data_train.values[:, :-1]
y = data_train.values[:, -1]

nrule = x.shape[1]
ndata = x.shape[0]

x_mpz = [rule_vectompz(x[:, i]) for i in range(nrule)]
y_mpz = rule_vectompz(y)
z_mpz = 8193

points_cap = make_all_ones(ndata + 1)
xi = x_mpz[1]


# a = ~xi | mpz(pow(2, ndata))
# d = xi | mpz(pow(2, ndata))
# bin_a = bin(a)
# bin_d = bin(d)
# print("+++++++++", a)
# print("========", d)
# print("+++++++++", bin_a)
# print("+++++++++", bin_d)
#
# l1_cap, ndata1 = rule_vand(points_cap, a)
# print("==========", l1_cap)
# print("==========", ndata1)
# bin_c = bin(l1_cap)
# print("+++++++++", bin_c)


# bin_a = bin(points_cap)
# bin_b = bin(y_mpz)
# print("+++++++++", points_cap)
# print("+++++++++", y_mpz)
# print("+++++++++", bin_a)
# print("+++++++++", bin_b)
# vand =  y_mpz & points_cap
# print("+++++++++", vand)
# # subtract 1 to remove leading ones
# cnt = gmpy2.popcount(vand) - 1
# print("+++++++++", cnt)
# return vand, cnt
# _, num_ones = rule_vand(points_cap, y_mpz)
#new_points_cap, new_num_captured = rule_vand(tag, tag_rule)

##MAKE ALL ONES KAY LIYE
# points_cap = make_all_ones(ndata + 1)\\ ek feature jada karkay power latay hain mp pe covert.
# ones = pow(2, length) - 1
# default_tt = mpz(ones)
# return default_tt
# count_ones(points_cap)

##MPZ CONVERT Kay liye binary se
# for i in range(nrule): // mpz pe convert 1 barha kay kartay hain(binary se number mein)
#     vec = x[:, i]
#     print("=========",x[:, i])
#     a = [str(i) for i in vec]
#     b = '1' + "".join(a)
#     c = mpz(b, 2)
#
#     print("+++++++++", a)
#     print("//////////", b)
#     print("||||||||||", c)
#     # mpz('1' + "".join([str(i) for i in vec]), 2)



