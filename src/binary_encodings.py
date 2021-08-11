import pandas as pd
import math
import category_encoders as ce
from enum import Enum,auto

class Encodings(Enum):
    # Options of encoding defined
    NumberBased = auto()
    Asymmetric = auto()
    ModifiedAsymmetric = auto()
    GenericEncoding = auto()

def convert_dataset(features):
    dict = {}
    for column in features:

        ncol_lst = []
        cols = features[column].astype('category')
        cols = cols.cat.codes

        # # UNIQUE FEATURES IN CLOUMN
        unq_val = cols.nunique()
        # # expected columns for feature
        expt_columns = math.ceil(math.log(unq_val, 2))

        l = 1

        if expt_columns > 1:
            if unq_val > 2:
                maintin_list = []
                for i in cols:
                    bits = bin(i)[2:].zfill(expt_columns)
                    bit_lst = [int(d) for d in str(bits)]
                    maintin_list += [bit_lst]
                # print(bin(i)[2:].zfill(expt_columns))

            # dict = dict((el,0) for el in ncol_lst)
            l = len(maintin_list[0])

        for i in range(0, expt_columns):
            ncol_lst.append(str(column) + '_' + str(i + 1))

        for i in range(0, l):
            # varnam = "d" + str(i)
            if expt_columns > 1:
                varnam = [item[i] for item in maintin_list]
            else:
                varnam = [item for item in cols]
            dict[ncol_lst[i]] = varnam

    cnvt_data = pd.DataFrame(dict)
    return cnvt_data

def num_encoding(ds):
    features = ds.iloc[:, :-1]
    labels = ds.iloc[:, -1:]

    # print(features)
    # print(labels)
    binfeat = convert_dataset(features)
    binclass = convert_dataset(labels)

    # bintable = binclass.append(binfeat)
    bintable = pd.concat([binfeat, binclass], axis=1)

    return bintable

def osdt_enc(ds):

    # modified Asymetric encoding
    de = pd.get_dummies(data=ds, drop_first=True)

    return de

    # print(de.head())

def Asym_enc(ds):
    # modified Asymetric encoding
    de = pd.get_dummies(data=ds, drop_first=False)

    return de

def gneric_encoding(ds):
    # # Default binary encoder
    encoder= ce.BinaryEncoder(cols=ds,return_df=True)
    data_encoded=encoder.fit_transform(ds)

    return data_encoded


def converter(dataset,selection,idCol):

    de = None
    # First Column Remove
    if idCol == 'last':
        ds = dataset.iloc[:, :-1]
    # Second Column Remove
    else:
        ds = dataset.iloc[:, 1:]

    if selection == Encodings.ModifiedAsymmetric:
        de = osdt_enc(ds)
    elif selection == Encodings.Asymmetric:
        de = Asym_enc(ds)
    elif selection == Encodings.NumberBased:
        de = num_encoding(ds)
    elif selection == Encodings.GenericEncoding:
        de = gneric_encoding(ds)

    # Export to csv
    # de.to_csv('Result.csv')
    # for col in de:
    #     print(col)
    # print(de)


# ds = pd.DataFrame(pd.read_csv('../data/Playtennis.csv', sep=";"))

def encode(filename='../data/monks-1.test',idCol='first',encodingtype=Encodings.NumberBased):
    file = filename

    # idCol = 'last'

    if file.endswith('.csv'):
        ds = pd.DataFrame(pd.read_csv(file, sep=";"))
    else:
        ds = pd.DataFrame(pd.read_csv(file, sep=" "))

    converter(ds,encodingtype,idCol)

encode(filename='../data/Playtennis.csv',encodingtype=Encodings.NumberBased)
