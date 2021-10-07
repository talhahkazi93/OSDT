import pandas as pd

from osdtr import bbound, predict
from utils import ObjFunction
import argparse
import sys

class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('check arguments \n')
        sys.exit(255)

def test_accuracy_onefold(file, lambs, timelimit,objfunc):
    """
    Run OSDT, External path length and internal path length as objective function
    use all data, only training accuracy
    :param X:
    :param y:
    :param lambs:
    :param file_CART:
    :param file_OSDT:
    :return:
    """
    for lamb in lambs:

        file_train = file
        #file_train = file + '.train' + str(1) + '.csv'
        # file_test = file + '.test' + str(1) + '.csv'

        data_train = pd.DataFrame(pd.read_csv(file_train, sep=";"))

        X_train = data_train.values[:, :-1]
        y_train = data_train.values[:, -1]


         # X_test = data_test.values[:, :-1]
        # y_test = data_test.values[:, -1]

        # CART
        # clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=max(math.ceil(lamb * 2 * len(y_train)), 2),
        #                                   min_samples_leaf=math.ceil(lamb * len(y_train)),
        #                                   max_leaf_nodes=math.floor(1 / (2 * lamb)),
        #                                   min_impurity_decrease=lamb
        #                                   )
        # clf = clf.fit(X_train, y_train)
        #
        # nleaves_CART = (clf.tree_.node_count + 1) / 2
        # trainaccu_CART = clf.score(X_train, y_train)
        # testaccu_CART = clf.score(X_test, y_test)
        # print(">>>>>>>>>>>>>>>>> nleaves_CART:", nleaves_CART)
        # print(">>>>>>>>>>>>>>>>> trainaccu_CART:", trainaccu_CART)
        # print(">>>>>>>>>>>>>>>>> testaccu_CART:", testaccu_CART)

        # with open(file_CART, 'a+') as f:
        #     f.write(";".join([str('NA'), str(lamb), str(nleaves_CART), str(trainaccu_CART), str('NA')]) + '\n')

        # OSDT
        leaves_c, prediction_c, dic, nleaves_OSDT, nrule, ndata, totaltime, time_c, COUNT, C_c, trainaccu_OSDT, best_is_cart, clf =\
            bbound(X_train, y_train, lamb=lamb,support=True, incre_support=True, accu_support=True, equiv_points=True,
           lookahead=True, lenbound=False, objfunc=objfunc, prior_metric="curiosity", timelimit=timelimit, init_cart=False,file = file)

        if nleaves_OSDT >= 16:
            break

# lambs1 = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025]

# Read in the dataset
compas = '../data/datasets/Osdt_enc/compas-binary.csv'
monk1 = '../data/datasets/Osdt_enc/monk1-train.csv'
monk2 = '../data/datasets/Osdt_enc/monk2-train.csv'
monk3 = '../data/datasets/Osdt_enc/monk3-train.csv'
balance = '../data/datasets/Osdt_enc/balance-scale.csv'
tictactoe = '../data/datasets/Osdt_enc/tic-tac-toe.csv'
car = '../data/datasets/Osdt_enc/car-evaluation.csv'

test1 = '../data/preprocessed/monk1-train.csv.test1.csv'
test2 = '../data/preprocessed/monk2-train.csv.test1.csv'

monk1 = '../data/datasets/Numberbased_enc/monks-1.train.csv'


def main():
    # parser = ArgumentParser(description='Inputs')
    # # parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag', required=True)
    # # Required Parameter
    # parser.add_argument('-obj', action='append', dest='objfunc', nargs='?', help='Objective Fucntion', required=True,const='O')
    # parser.add_argument('-l', action='append', dest='lamb', nargs='+', help='lambdha values', type=float)
    # # Optional Parameter
    # # parser.add_argument('-s', action='append', dest='authfile', help='auth-file', nargs='?', const='bank.auth')
    #
    #
    # args = parser.parse_args()
    # # print(args.lamb[0])
    # if args.objfunc[0] in ['o','O']:
    #     ob = ObjFunction.OSDT
    # elif args.objfunc[0] in ['e','E']:
    #     ob = ObjFunction.ExternalPathLength
    # elif args.objfunc[0] in ['i','I']:
    #     ob = ObjFunction.InternalPathLength
    # else:
    #     sys.stderr.write('Incorrect objective function \n')
    #     sys.exit(255)


    # ObjFunction.OSDT
    # ObjFunction.ExternalPathLength
    # ObjFunction.InternalPathLength
    # [compas,monk1, balance, tictactoe, car]

    obj = [ObjFunction.OSDT]
    timelimi1 = 1800
    datasets = [compas,monk1]
    # datasets = [test1,test2]
    for ob in obj:
        for file in datasets:
        # file = '../data/preprocessed/monk1-train.csv'
            test_accuracy_onefold(file, lambs=[0.005],timelimit=timelimi1,objfunc=ob)
    # test_accuracy_onefold(compas, lambs=args.lamb[0], timelimit=timelimi1,objfunc=ob)
    # test_accuracy_onefold(compas, lambs=[0.005], timelimit=timelimi1,objfunc=ob)


if __name__ == "__main__":
    main()