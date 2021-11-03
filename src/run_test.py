import pandas as pd

from osdtr import bbound, predict
from utils import ObjFunction , Encodings
import sys

def test_accuracy_onefold(file, lambs, timelimit,objfunc):
    """
    Run OSDT, External path length, internal path length and Weighted external path length as objective function
    use all data, only training accuracy
    :param file
    :param lambs
    :param Time limit:
    :param Objective function:
    :return:
    """
    for lamb in lambs:

        data_train = pd.DataFrame(pd.read_csv(file, sep=";"))

        X_train = data_train.values[:, :-1]
        y_train = data_train.values[:, -1]

        leaves_c, prediction_c, dic, nleaves_OSDT, nrule, ndata, totaltime, time_c, COUNT, C_c, trainaccu_OSDT, best_is_cart, clf =\
            bbound(X_train, y_train, lamb=lamb,support=True, incre_support=True, accu_support=True, equiv_points=True,
           lookahead=True, lenbound=False, objfunc=objfunc, prior_metric="curiosity", timelimit=timelimit,file = file)


def main():
    """
    Select objective function from(multiple allowed):
    ObjFunction.OSDT
    ObjFunction.ExternalPathLength
    ObjFunction.InternalPathLength
    ObjFunction.WeightedExternalPathLength
    """
    objective_function = [ObjFunction.WeightedExternalPathLength]

    """
    Select Encodings from:
    Encodings.AsymEnc
    Encodings.GenericEnc
    Encodings.NumBasedEnc
    Encodings.OsdtEnc
    """
    encoding = Encodings.OsdtEnc

    if encoding == Encodings.AsymEnc:
        monk1 = '../data/datasets/Asymmetric_enc/monk1-train.csv'
        monk2 = '../data/datasets/Asymmetric_enc/monk2-train.csv'
        monk3 = '../data/datasets/Asymmetric_enc/monk3-train.csv'
        tictactoe = '../data/datasets/Asymmetric_enc/tic-tac-toe.csv'
        car = '../data/datasets/Asymmetric_enc/car-evaluation.csv'
    elif encoding == Encodings.GenericEnc:
        monk1 = '../data/datasets/Generic_enc/monk1-train.csv'
        monk2 = '../data/datasets/Generic_enc/monk2-train.csv'
        monk3 = '../data/datasets/Generic_enc/monk3-train.csv'
        tictactoe = '../data/datasets/Generic_enc/tic-tac-toe.csv'
        car = '../data/datasets/Generic_enc/car-evaluation.csv'
    elif encoding == Encodings.NumBasedEnc:
        monk1 = '../data/datasets/Numberbased_enc/monk1-train.csv'
        monk2 = '../data/datasets/Numberbased_enc/monk2-train.csv'
        monk3 = '../data/datasets/Numberbased_enc/monk3-train.csv'
        tictactoe = '../data/datasets/Numberbased_enc/tic-tac-toe.csv'
        car = '../data/datasets/Numberbased_enc/car-evaluation.csv'
    elif encoding == Encodings.OsdtEnc:
        compas = '../data/datasets/Osdt_enc/compas-binary.csv'
        monk1 = '../data/datasets/Osdt_enc/monk1-train.csv'
        monk2 = '../data/datasets/Osdt_enc/monk2-train.csv'
        monk3 = '../data/datasets/Osdt_enc/monk3-train.csv'
        fico = '../data/datasets/Osdt_enc/fico_binary.csv'
        tictactoe = '../data/datasets/Osdt_enc/tic-tac-toe.csv'
        car = '../data/datasets/Osdt_enc/car-evaluation.csv'
    else:
        sys.stderr.write('Incorrect encoding selected\n')
        sys.exit(255)

    """
    Select datasets from(multiple allowed):
    compas, monk1, monk2, monk3, fico, tictactoe, car
    (compas and fico only available for osdt encoding)
    """
    datasets = [compas]

    """
    Select timelimit in seconds :
    1800s:30min, 3600s:60min, 7200s:120min
    """
    timelimit = 1800

    """
    Select lambda values according to objective function(multiple allowed) :
    0.1, 0.05, 0.025, 0.01, 0.005, 0.0025
    """
    lambdha = [0.005]

    for ob in objective_function:
        for file in datasets:
            test_accuracy_onefold(file, lambs=lambdha,timelimit=timelimit,objfunc=ob)

if __name__ == "__main__":
    main()

