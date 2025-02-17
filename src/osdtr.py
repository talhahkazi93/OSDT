import numpy as np
import pandas as pd
import heapq
import math
import time
import copy
from itertools import product, compress
from gmpy2 import mpz
from utils import ObjFunction
from utils import RulesFunctions as rulef
from utils import OSDT,ExternalPathLength,Internalpathlength,Weightedexternalpathlength
import sklearn.tree
import pickle
import ntpath
import os
import csv

class CacheTree:
    """
    A tree data structure.
    leaves: a 2-d tuple to encode the leaves
    num_captured: a list to record number of data captured by the leaves
    """
    def __init__(self, leaves, obj):
        self.leaves = leaves
        self.rules = [l.rules for l in leaves]
        rulelen = [len(val) for val in self.rules]

        self.ext= sum(rulelen)
        self.risk = obj.calc_risk(leaves=self.leaves)

    def sorted_leaves(self):
        # Used by the cache
        return tuple(sorted(leaf.rules for leaf in self.leaves))

class Tree:
    """
        A tree data structure, based on CacheTree
        cache_tree: a CacheTree
        num_captured: a list to record number of data captured by the leaves
        """

    def __init__(self, cache_tree, ndata, splitleaf=None, prior_metric=None, obj=None):
        self.cache_tree = cache_tree
        # a queue of lists indicating which leaves will be split in next rounds
        # (1 for split, 0 for not split)
        self.splitleaf = splitleaf

        leaves = cache_tree.leaves
        l = len(leaves)
        self.leafsup = sum([l.support_val for l in leaves])
        # print('///////////////LEAFSuPPOrt///////////////'+str(self.leafsup))

        self.lb = obj.calc_loss(cache_tree=cache_tree,splitleaf=splitleaf)

        # which metrics to use for the priority queue
        if leaves[0].num_captured == ndata:
            # this case is when constructing the null tree ((),)
            self.metric = 0
        elif prior_metric == "objective":
            self.metric = cache_tree.risk
        elif prior_metric == "bound":
            self.metric = self.lb
        elif prior_metric == "curiosity":
            removed_leaves = list(compress(leaves, splitleaf))
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
            if num_cap_rm < ndata:
                self.metric = self.lb / ((ndata - num_cap_rm) / ndata)
            else:
                self.metric = self.lb / (0.01 / ndata)
        elif prior_metric == "entropy":
            removed_leaves = list(compress(leaves, splitleaf))
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
            # entropy weighted by number of points captured
            self.entropy = [
                (-leaves[i].p * math.log2(leaves[i].p) - (1 - leaves[i].p) * math.log2(1 - leaves[i].p)) * leaves[
                    i].num_captured if leaves[i].p != 0 and leaves[i].p != 1 else 0 for i in range(l)]
            if num_cap_rm < ndata:
                self.metric = sum(self.entropy[i] for i in range(l) if splitleaf[i] == 0) / (
                        ndata - sum(leaf.num_captured for leaf in removed_leaves))
            else:
                self.metric = sum(self.entropy[i] for i in range(l) if splitleaf[i] == 0) / 0.01
        elif prior_metric == "gini":
            removed_leaves = list(compress(leaves, splitleaf))
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
            # gini index weighted by number of points captured
            self.giniindex = [(2 * leaves[i].p * (1 - leaves[i].p))
                              * leaves[i].num_captured for i in range(l)]
            if num_cap_rm < ndata:
                self.metric = sum(self.giniindex[i] for i in range(l) if splitleaf[i] == 0) / (
                        ndata - sum(leaf.num_captured for leaf in removed_leaves))
            else:
                self.metric = sum(self.giniindex[i] for i in range(l) if splitleaf[i] == 0) / 0.01
        elif prior_metric == "FIFO":
            self.metric = 0

    def __lt__(self, other):
        # define <, which will be used in the priority queue
        return self.metric < other.metric

class CacheLeaf:
    """
    A data structure to cache every single leaf (symmetry aware)
    """
    def __init__(self, ndata, rules, y_mpz, z_mpz, points_cap, num_captured, support, is_feature_dead, obj):
        self.rules = rules
        self.points_cap = points_cap
        self.num_captured = num_captured
        self.is_feature_dead = is_feature_dead
        self.len = len(rules)

        _, num_ones = rulef.rule_vand(points_cap, y_mpz)

        # b0 is defined in (28)

        _, num_errors = rulef.rule_vand(points_cap, z_mpz)
        self.B0 = num_errors / ndata

        if self.num_captured:
            self.prediction = int(num_ones / self.num_captured >= 0.5)
            if self.prediction == 1:
                self.num_captured_incorrect = self.num_captured - num_ones
            else:
                self.num_captured_incorrect = num_ones
            self.p = self.num_captured_incorrect / self.num_captured
        else:
            self.prediction = 0
            self.num_captured_incorrect = 0
            self.p = 0

        self.loss = float(self.num_captured_incorrect) / ndata

        self.support_val = self.num_captured / ndata

        # leaf weighted external path length
        self.leaf_we = self.len * self.support_val

        # Lower bound on leaf support
        if support:
            self.is_dead = obj.calc_leaf_supp(leaf_len=self.len,support=self.loss)
        else:
            self.is_dead = False

def log(tic, lines, COUNT_POP, COUNT, queue, metric, R_c, tree_old, tree_new, sorted_new_tree_rules):
    "log"

    the_time = str(time.time() - tic)

    the_count_pop = str(COUNT_POP)
    the_count = str(COUNT)
    the_queue_size = str(0)  # str(len(queue))
    the_metric = str(metric)
    the_Rc = str(R_c)

    the_old_tree = str(0)  # str(sorted([leaf.rules for leaf in tree_old.cache_tree.leaves]))
    the_old_tree_splitleaf = str(0)  # str(tree_old.splitleaf)
    the_old_tree_objective = str(tree_old.cache_tree.risk)
    the_old_tree_lbound = str(tree_old.lb)
    the_new_tree = str(0)  # str(list(sorted_new_tree_rules))
    the_new_tree_splitleaf = str(0)  # str(tree_new.splitleaf)

    the_new_tree_objective = str(0)  # str(tree_new.cache_tree.risk)
    the_new_tree_lbound = str(tree_new.lb)
    the_new_tree_length = str(0)  # str(len(tree_new.cache_tree.leaves))
    the_new_tree_depth = str(0)  # str(max([len(leaf.rules) for leaf in tree_new.leaves]))

    the_queue = str(0)  # str([[ leaf.rules for leaf in thetree.leaves]  for _,thetree in queue])

    line = ";".join([the_time, the_count_pop, the_count, the_queue_size, the_metric, the_Rc,
                     the_old_tree, the_old_tree_splitleaf, the_old_tree_objective, the_old_tree_lbound,
                     the_new_tree, the_new_tree_splitleaf,
                     the_new_tree_objective, the_new_tree_lbound, the_new_tree_length, the_new_tree_depth,
                     the_queue
                     ])
    lines.append(line)

def generate_new_splitleaf(unchanged_leaves, removed_leaves, new_leaves,
                           R_c, incre_support, obj):
    """
    generate the new splitleaf for the new tree
    """

    n_removed_leaves = len(removed_leaves)
    n_unchanged_leaves = len(unchanged_leaves)
    n_new_leaves = len(new_leaves)

    n_new_tree_leaves = n_unchanged_leaves + n_new_leaves

    splitleaf1 = [0] * n_unchanged_leaves + [1] * n_new_leaves  # all new leaves labeled as to be split

    sl = []
    for i in range(n_removed_leaves):

        splitleaf = [0] * n_new_tree_leaves

        idx1 = 2*i
        idx2 = 2*i+1
        # (Lower bound on incremental classification accuracy)

        a_l = removed_leaves[i].loss - new_leaves[idx1].loss - new_leaves[idx2].loss

        if not incre_support:
            a_l = float('Inf')

        if a_l <= obj.calc_incrm_acc(new_leaves=new_leaves,i=i):
            splitleaf[n_unchanged_leaves + idx1] = 1
            splitleaf[n_unchanged_leaves + idx2] = 1
            sl.append(splitleaf)
        else:
            sl.append(splitleaf1)

    return sl

def gini_reduction(x_mpz, y_mpz, ndata, rule_idx, points_cap=None):
    """
    calculate the gini reduction by each feature
    return the rank of by descending
    """

    if points_cap == None:
        points_cap = rulef.make_all_ones(ndata + 1)

    ndata0 = rulef.count_ones(points_cap)
    _, ndata01 = rulef.rule_vand(y_mpz, points_cap)

    p0 = ndata01 / ndata0
    gini0 = 2 * p0 * (1 - p0)

    gr = []
    for i in rule_idx:
        xi = x_mpz[i]
        l1_cap, ndata1 = rulef.rule_vand(points_cap, ~xi | mpz(pow(2, ndata)))

        _, ndata11 = rulef.rule_vand(l1_cap, y_mpz)

        l2_cap, ndata2 = rulef.rule_vand(points_cap, xi)

        _, ndata21 = rulef.rule_vand(l2_cap, y_mpz)

        p1 = ndata11 / ndata1 if ndata1 != 0 else 0
        p2 = ndata21 / ndata2 if ndata2 != 0 else 0
        gini1 = 2 * p1 * (1 - p1)
        gini2 = 2 * p2 * (1 - p2)
        gini_red = gini0 - ndata1 / ndata0 * gini1 - ndata2 / ndata0 * gini2
        gr.append(gini_red)

    gr = np.array(gr)
    order = list(gr.argsort()[::-1])

    odr = [rule_idx[r] for r in order]

    #print("ndata0:", ndata0)
    #print("ndata1:", ndata1)
    #print("ndata2:", ndata2)
    # print("gr:", gr)
    # print("order:", order)
    # print("odr:", odr)
    #print("the rank of x's columns: ", rank)

    dic = dict(zip(np.array(rule_idx)+1, odr))

    return odr, dic


def bbound(x, y, lamb, prior_metric=None, MAXDEPTH=float('Inf'), MAX_NLEAVES=float('Inf'), niter=float('Inf'), logon=False,
           support=True, incre_support=True, accu_support=True, equiv_points=True,
           lookahead=True, lenbound=True, objfunc=ObjFunction.ExternalPathLength, R_c0 = 1, timelimit=float('Inf'), init_cart = False,
           saveTree = False, readTree = False,file=None):
    """
    An implementation of Algorithm
    ## multiple copies of tree
    ## mark which leaves to be split
    """

    x0 = copy.deepcopy(x)
    y0 = copy.deepcopy(y)

    # Initialize best rule list and objective
    # d_c = None
    # R_c = 1

    tic = time.time()

    nrule = x.shape[1]
    ndata = len(y)
    max_nleaves = 2**nrule
    print("nrule:", nrule)
    print("ndata:", ndata)

    x_mpz = [rulef.rule_vectompz(x[:, i]) for i in range(nrule)]
    y_mpz = rulef.rule_vectompz(y)
    #print("x_mpz000",x_mpz)
    #print("y_mpz000", y_mpz)

    # order the columns by descending gini reduction
    idx, dic = gini_reduction(x_mpz, y_mpz, ndata, range(nrule))
    x = x[:, idx]
    x_mpz = [x_mpz[i] for i in idx]
    print("the order of x's columns: ", idx)
    #print("x_mpz111", x_mpz)
    #print("y_mpz111", y_mpz)

    """
    calculate z, which is for the equivalent points bound
    z is the vector defined in algorithm 5 of the CORELS paper
    z is a binary vector indicating the data with a minority lable in its equivalent set
    """
    z = pd.DataFrame([-1] * ndata).values
    # enumerate through theses samples
    for i in range(ndata):
        # if z[i,0]==-1, this sample i has not been put into its equivalent set
        if z[i, 0] == -1:
            tag1 = np.array([True] * ndata)
            for j in range(nrule):
                rule_label = x[i][j]
                # tag1 indicates which samples have exactly the same features with sample i
                tag1 = (x[:, j] == rule_label) * tag1

            y_l = y[tag1]
            pred = int(y_l.sum() / len(y_l) >= 0.5)
            # tag2 indicates the samples in a equiv set which have the minority label
            tag2 = (y_l != pred)
            z[tag1, 0] = tag2

    z_mpz = rulef.rule_vectompz(z.reshape(1, -1)[0])


    lines = []  # a list for log
    leaf_cache = {}  # cache leaves
    tree_cache = {}  # cache trees

    # # Intinalizing opjective function
    # obj = ObjectiveFunc(objfunc=objfunc,lamb=lamb)

    if objfunc == ObjFunction.OSDT:
        obj = OSDT(lamb=lamb)
    elif objfunc == ObjFunction.ExternalPathLength:
        obj = ExternalPathLength(lamb=lamb)
    elif objfunc == ObjFunction.InternalPathLength:
        obj = Internalpathlength(lamb=lamb)
    elif objfunc == ObjFunction.WeightedExternalPathLength:
        obj = Weightedexternalpathlength(lamb=lamb)

    # initialize the queue to include just empty root
    queue = []
    root_leaf = CacheLeaf(ndata, (), y_mpz, z_mpz, rulef.make_all_ones(ndata + 1), ndata, support, [0] * nrule, obj=obj)

    d_c = CacheTree(leaves=[root_leaf], obj=obj)
    R_c = d_c.risk

    tree0 = Tree(cache_tree=d_c, ndata=ndata, splitleaf=[1], prior_metric=prior_metric, obj=obj)

    heapq.heappush(queue, (tree0.metric, tree0))
    # heapq.heappush(queue, (2*tree0.metric - R_c, tree0))
    # queue.append(tree0)

    clf = None
    best_is_cart = False  # a flag for whether or not the best is the initial CART
    if init_cart: # if warm start
        # CART
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=None if MAXDEPTH == float('Inf') else MAXDEPTH,
                                                  min_samples_split=max(math.ceil(lamb * 2 * len(y)), 2),
                                          min_samples_leaf=math.ceil(lamb * len(y)),
                                          max_leaf_nodes=math.floor(1 / (2 * lamb)),
                                          min_impurity_decrease=lamb
                                          )
        clf = clf.fit(x0, y0)

        nleaves_CART = (clf.tree_.node_count + 1) / 2
        trainaccu_CART = clf.score(x0, y0)

        R_c = 1 - trainaccu_CART + lamb*nleaves_CART
        d_c = clf

        C_c = 0
        time_c = time.time() - tic

        best_is_cart = True

    if R_c0 < R_c:
        R_c = R_c0

    # log(lines, lamb, tic, len(queue), tuple(), tree0, R, d_c, R_c)

    leaf_cache[()] = root_leaf

    COUNT = 0  # count the total number of trees in the queue
    C_c = 0
    time_c = 0
    COUNT_POP = 0

    COUNT_UNIQLEAVES = 0
    COUNT_LEAFLOOKUPS = 0
    # timelimit = float('Inf')
    # last = None

    while queue and COUNT < niter and time.time() - tic < timelimit:
        # for a in queue:
        #     print("QUEUE MEMBERS++++++",a)
        #     print("ALL THE LEAVES=====", [leaf.rules for leaf in a[1].cache_tree.leaves])
        #     print("WHICH NODE TO BE SPLIT",[c for c in a[1].splitleaf])
        #     last = a
        # tree = queue.pop(0)
        # metric, tree = last
        metric, tree = heapq.heappop(queue)



        # ATEMP_LEAVES = [leaf.rules for leaf in d_c.leaves]
        '''
        if prior_metric == "bound":
            if tree.lb + lamb*len(tree.splitleaf) >= R_c:
                break
        '''

        COUNT_POP = COUNT_POP + 1

        # print([leaf.rules for leaf in tree.leaves])
        # print("curio", curio)
        leaves = tree.cache_tree.leaves

        # print("=======COUNT=======",COUNT)
        # print("d",d)
        # print("R",tree.lbound[0]+(tree.num_captured_incorrect[0])/len(y))

        leaf_split = tree.splitleaf

        removed_leaves = list(compress(leaves, leaf_split))

        old_tree_length = len(leaf_split)
        new_tree_length = len(leaf_split) + sum(leaf_split)

        # prefix-specific upper bound on number of leaves
        if lenbound and new_tree_length >= min(old_tree_length + math.floor((R_c - tree.lb) / lamb),
                                               max_nleaves):
            #print("toolong===COUNT:", COUNT)
            continue

        n_removed_leaves = sum(leaf_split)
        n_unchanged_leaves = old_tree_length - n_removed_leaves

        # equivalent points bound combined with the lookahead bound
        supp = sum([l.support_val for l in removed_leaves])
        lb = tree.lb
        b0 = sum([leaf.B0 for leaf in removed_leaves]) if equiv_points else 0
        lambbb = lamb if lookahead else 0

        if objfunc == ObjFunction.WeightedExternalPathLength:
            if obj.calc_lookahead(R_c=R_c,lb=lb,b0=b0,n_removed_leaves=n_removed_leaves,lamb=lambbb,support=supp):
                continue
        else:
            if obj.calc_lookahead(R_c=R_c,lb=lb,b0=b0,n_removed_leaves=n_removed_leaves,lamb=lambbb):
                continue

        #dun
        leaf_no_split = [not split for split in leaf_split]
        unchanged_leaves = list(compress(leaves, leaf_no_split))

        # lb = sum(l.loss for l in unchanged_leaves)
        # b0 = sum(l.b0 for l in removed_leaves)

        # Generate all assignments of rules to the leaves that are due to be split

        rules_for_leaf = [set(range(1, nrule + 1)) - set(map(abs, l.rules)) -
                          set([i+1 for i in range(nrule) if l.is_feature_dead[i] == 1]) for l in removed_leaves]


        for leaf_rules in product(*rules_for_leaf):

            if time.time() - tic >= timelimit:
                break

            new_leaves = []
            flag_increm = False  # a flag for jump out of the loops (incremental support bound)
            for rule, removed_leaf in zip(leaf_rules, removed_leaves):

                rule_index = rule - 1
                tag = removed_leaf.points_cap  # points captured by the leaf's parent leaf

                for new_rule in (-rule, rule):
                    new_rule_label = int(new_rule > 0)
                    new_rules = tuple(
                        sorted(removed_leaf.rules + (new_rule,)))
                    if new_rules not in leaf_cache:

                        COUNT_UNIQLEAVES = COUNT_UNIQLEAVES+1

                        tag_rule = x_mpz[rule_index] if new_rule_label == 1 else ~(x_mpz[rule_index]) | mpz(pow(2, ndata))
                        #print("x_mpz",x_mpz)
                        #print("tag_rule",tag_rule)
                        new_points_cap, new_num_captured = rulef.rule_vand(tag, tag_rule)
                        # print("tag:", tag)
                        # print("tag_rule:", tag_rule)
                        # print("new_points_cap:", new_points_cap)
                        # print("new_num_captured:", new_num_captured)

                        #parent_is_feature_dead =
                        new_leaf = CacheLeaf(ndata, new_rules, y_mpz, z_mpz, new_points_cap, new_num_captured,
                                             support, removed_leaf.is_feature_dead.copy(), obj)
                        leaf_cache[new_rules] = new_leaf
                        new_leaves.append(new_leaf)
                    else:
                        COUNT_LEAFLOOKUPS = COUNT_LEAFLOOKUPS+1

                        new_leaf = leaf_cache[new_rules]
                        new_leaves.append(new_leaf)

                    # print("new_leaf:", new_leaf.rules)
                    # print("leaf loss:", new_leaf.loss)
                    # print("new_leaf.num_captured:",new_leaf.num_captured)
                    # print("new_leaf.num_captured_incorrect",new_leaf.num_captured_incorrect)

                    # print("******* old_rules:", removed_leaf.rules)
                    # print("******* new_rules:", new_rules)


                    # Lower bound on classification accuracy
                    if accu_support == True and obj.calc_acc_supp(new_leaf=new_leaf,ndata=ndata):
                        removed_leaf.is_feature_dead[rule_index] = 1
                        flag_increm = True
                        break

                if flag_increm:
                    break

            if flag_increm:
                continue

            new_tree_leaves = unchanged_leaves + new_leaves

            sorted_new_tree_rules = tuple(sorted(leaf.rules for leaf in new_tree_leaves))

            if sorted_new_tree_rules in tree_cache:
                # print("====== New Tree Duplicated!!! ======")
                # print("sorted_new_tree_rules:", sorted_new_tree_rules)
                continue
            else:
                tree_cache[sorted_new_tree_rules] = True

            child = CacheTree(leaves=new_tree_leaves, obj=obj)

            R = child.risk
            # print("child:", child.sorted_leaves())
            # print("R:",R)
            if R < R_c:
                d_c = child
                R_c = R
                C_c = COUNT + 1
                time_c = time.time() - tic

                best_is_cart = False

            # generate the new splitleaf for the new tree
            sl = generate_new_splitleaf(unchanged_leaves, removed_leaves, new_leaves, R_c, incre_support, obj)
            # print("sl:", sl)

            # A leaf cannot be split if
            # 1. the MAXDEPTH has been reached
            # 2. the leaf is dead (because of antecedent support)
            # 3. all the features that have not been used are dead
            cannot_split = [len(l.rules) >= MAXDEPTH or l.is_dead or
                            all([l.is_feature_dead[r - 1] for r in range(1, nrule + 1)
                                 if r not in map(abs, l.rules)])
                            for l in new_tree_leaves]

            # if len(new_tree_leaves)!=new_tree_length:
            #    print("len(new_tree_leaves):",len(new_tree_leaves))
            #    print("new_tree_length:", new_tree_length)

            # For each copy, we don't split leaves which are not split in its parent tree.
            # In this way, we can avoid duplications.
            can_split_leaf = [(0,)] * n_unchanged_leaves + \
                             [(0,) if cannot_split[i]
                              else (0, 1) for i in range(n_unchanged_leaves, new_tree_length)]
            # Discard the first element of leaf_splits, since we must split at least one leaf
            new_leaf_splits0 = np.array(list(product(*can_split_leaf))[1:])#sorted(product(*can_split_leaf))[1:]
            len_sl = len(sl)
            if len_sl == 1:
                # Filter out those which split at least one leaf in dp (d0)
                new_leaf_splits = [ls for ls in new_leaf_splits0
                                   if np.dot(ls, sl[0]) > 0]
                # print("n_unchanged_leaves:",n_unchanged_leaves)
                # print("cannot_split:", cannot_split)
                # print("can_split_leaf:",can_split_leaf)
                # print("new_leaf_splits:",new_leaf_splits)
            else:
                # Filter out those which split at least one leaf in dp and split at least one leaf in d0
                new_leaf_splits = [ls for ls in new_leaf_splits0
                                   if all([np.dot(ls, sl[i]) > 0 for i in range(len_sl)])]

            for new_leaf_split in new_leaf_splits:
                # construct the new tree
                tree_new = Tree(cache_tree=child, ndata=ndata, splitleaf=new_leaf_split, prior_metric=prior_metric, obj=obj)

                # MAX Number of leaves
                if len(new_leaf_split)+sum(new_leaf_split) > MAX_NLEAVES:
                    continue

                COUNT = COUNT + 1
                # heapq.heappush(queue, (2*tree_new.metric - R_c, tree_new))
                heapq.heappush(queue, (tree_new.metric, tree_new))

                if logon:
                    log(tic, lines, COUNT_POP, COUNT, queue, metric, R_c, tree, tree_new, sorted_new_tree_rules)

                if COUNT % 1000000 == 0:
                    print("COUNT:", COUNT)

    totaltime = time.time() - tic

    if not best_is_cart:
        accu = 0
        leaves_c = [leaf.rules for leaf in d_c.leaves]
        prediction_c = [leaf.prediction for leaf in d_c.leaves]

        rulelen = [len(val) for val in leaves_c]
        ext = sum(rulelen)
        intpl = ext - 2 * (len(d_c.leaves) - 1)
        extpl = sum([leaf.leaf_we for leaf in d_c.leaves])

        if objfunc == ObjFunction.OSDT:
            accu = 1 - (R_c-lamb*len(d_c.leaves))
        elif objfunc == ObjFunction.ExternalPathLength:
            accu = 1 - (R_c - lamb * ext)
        elif objfunc == ObjFunction.InternalPathLength:
            accu = 1 - (R_c - lamb * intpl)
        elif objfunc == ObjFunction.WeightedExternalPathLength:
            accu = 1 - (R_c - lamb * extpl)

        num_captured = [leaf.num_captured for leaf in d_c.leaves]

        num_captured_incorrect = [leaf.num_captured_incorrect for leaf in d_c.leaves]

        nleaves = len(leaves_c)
    else:
        accu = trainaccu_CART
        leaves_c = 'NA'
        prediction_c = 'NA'
        # get_code(d_c, ['x'+str(i) for i in range(1, nrule+1)], [0, 1])
        num_captured = 'NA'
        num_captured_incorrect = 'NA'
        nleaves = nleaves_CART

    if saveTree:
        with open('tree.pkl', 'wb') as f:
            pickle.dump(d_c, f)
        with open('leaf_cache.pkl', 'wb') as f:
            pickle.dump(leaf_cache, f)

    if logon:
        header = ['time', '#pop', '#push', 'queue_size', 'metric', 'R_c',
                  'the_old_tree', 'the_old_tree_splitleaf', 'the_old_tree_objective', 'the_old_tree_lbound',
                  'the_new_tree', 'the_new_tree_splitleaf',
                  'the_new_tree_objective', 'the_new_tree_lbound', 'the_new_tree_length', 'the_new_tree_depth', 'queue']

        fname = "_".join([str(nrule), str(ndata), prior_metric,
                          str(lamb), str(MAXDEPTH), str(init_cart), ".txt"])
        with open(fname, 'w') as f:
            f.write('%s\n' % ";".join(header))
            f.write('\n'.join(lines))

    # Write results to file
    print(">>> dataset", ntpath.basename(file))
    headers = []
    curpath = os.path.abspath(os.curdir)
    # print(ntpath.basename(file))
    path = curpath+"\\results\\"+str(objfunc)
    fname = "_".join([path,str(lamb),".csv"])

    header = ['dataset','objective_function','lambda','total_time','Accuracy','leaves','num_captured','num_captured_incorrect',
              'prediction','Objective','TOTAL_COUNT','COUNT_UNIQLEAVES','COUNT_LEAFLOOKUPS',
              'COUNT_of_the_best_tree','time_to_achieved_best tree']
    try:
        line = [ntpath.basename(file), str(objfunc), str(lamb), str(totaltime), str(accu), str(leaves_c),
                str(num_captured), str(num_captured_incorrect),
                str(prediction_c), str(R_c), str(COUNT), str(COUNT_UNIQLEAVES), str(COUNT_LEAFLOOKUPS), str(C_c),
                str(time_c)]

    except Exception as err:
        print("Uh oh, please send me this message: '" + err + "'")
        line = [ntpath.basename(file), str(objfunc), str(lamb), str(totaltime), str(accu), str(leaves_c),
                str(num_captured), str(num_captured_incorrect),
                str(prediction_c), str(R_c), str(COUNT), str(COUNT_UNIQLEAVES), str(COUNT_LEAFLOOKUPS), str(0),
                str(0)]

    try:
        if os.path.exists(fname):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        with open(fname, append_write, newline='') as f:
            writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            if append_write == 'w':
                writer.writerow(header)
            writer.writerow(line)
    except Exception as err:
        print("Uh oh, please send me this message: '" + str(err) + "'")

    # print(">>> support bound:", support)
    # print(">>> accu_support:", accu_support)
    # print(">>> accurate support bound:", incre_support)
    # print(">>> equiv points bound:", equiv_points)
    # print(">>> lookahead bound:", lookahead)
    # print("prior_metric=", prior_metric)
    #
    # print("COUNT_UNIQLEAVES:", COUNT_UNIQLEAVES)
    # print("COUNT_LEAFLOOKUPS:", COUNT_LEAFLOOKUPS)
    #
    # print("Objective Function: ", objfunc)

    print("total time: ", totaltime)
    print("lambda: ", lamb)
    print("leaves: ", leaves_c)
    print("num_captured: ", num_captured)
    print("num_captured_incorrect: ", num_captured_incorrect)
    # # print("lbound: ", d_c.cache_tree.lbound)
    # # print("d_c.num_captured: ", [leaf.num_captured for leaf in d_c.cache_tree.leaves])
    print("prediction: ", prediction_c)
    print("Objective: ", R_c)
    print("Accuracy: ", accu)
    print("COUNT of the best tree: ", C_c)
    print("time when the best tree is achieved: ", time_c)
    print("TOTAL COUNT: ", COUNT)

    return leaves_c, prediction_c, dic, nleaves, nrule, ndata, totaltime, time_c, COUNT, C_c, accu, best_is_cart, clf


def predict(leaves_c, prediction_c, dic, x, y, best_is_cart, clf):
    """

    :param leaves_c:
    :param dic:
    :return:
    """
    ndata = x.shape[0]

    caps = []

    for leaf in leaves_c:
        cap = np.array([1] * ndata)
        for feature in leaf:
            idx = dic[abs(feature)]
            feature_label = int(feature > 0)
            cap = (x[:, idx] == feature_label) * cap
        caps.append(cap)

    yhat = np.array([1] * ndata)

    for j in range(len(caps)):
        idx_cap = [i for i in range(ndata) if caps[j][i] == 1]
        yhat[idx_cap] = prediction_c[j]

    right = yhat==y
    accu = right.mean()

    print("Testing Accuracy:", accu)

    return yhat, accu
