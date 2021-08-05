import pandas as pd
import math
import category_encoders as ce


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

        print(unq_val)
        print(expt_columns)
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
            print(l)

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
    features = ds.iloc[:, 1:-1]
    labels = ds.iloc[:, -1:]

    binfeat = convert_dataset(features)
    binclass = convert_dataset(labels)

    # bintable = binclass.append(binfeat)
    bintable = pd.concat([binfeat, binclass], axis=1)

    return bintable

def osdt_enc(ds):
    ds = ds.iloc[:, 1:]

    de = pd.get_dummies(data=ds, drop_first=True)

    for col in de:
        print(col)
    # print(de.head())



ds = pd.DataFrame(pd.read_csv('../data/Playtennis.csv',sep=";"))

osdt_enc(ds)

# features = ds.iloc[:, 1:-1]
#
# temp = features.iloc[:,1:-2]
# # Generic binary encoder
# encoder= ce.BinaryEncoder(cols=temp,return_df=True)
# data_encoded=encoder.fit_transform(temp)
#
# # modified Asymetric encoding
# de=pd.get_dummies(data=ds,drop_first=True)
# # Asymmetric Encoding
# dc=pd.get_dummies(data=temp,drop_first=False)
# # print(dc)


# final_table = num_encoding(ds)

# print(final_table.head())


