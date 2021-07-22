import pandas as pd
import math
import category_encoders as ce

ds = pd.DataFrame(pd.read_csv('../data/Playtennis.csv',sep=";"))


features = ds.iloc[:, 1:-1]
labels = ds.iloc[:, -1]

# for name, values in features.iteritems():
#     print(len(values))

ncol_lst = []

# print(bin(3))
# print("{0:03b}".format(3))
#isolate cloumns\

dict = {}
for column in features:

    ncol_lst = []
    print(column)
    cols = features[column].astype('category')
    # print(cols.cat.codes)
    cols = cols.cat.codes


    # # UNIQUE FEATURES IN CLOUMN
    unq_val = cols.nunique()
    # # print(features[column].nunique())
    # # expected columns for feature
    expt_columns = math.ceil(math.log(unq_val,2))

    # print(unq_val)
    # print(expt_columns)

    if unq_val > 2:
        maintin_list = []
        for i in cols:
            bits = bin(i)[2:].zfill(expt_columns)
            bit_lst = [int(d) for d in str(bits)]
            maintin_list += [bit_lst]
        print(maintin_list)
            # print(bin(i)[2:].zfill(expt_columns))

    for i in range(0,expt_columns):
        ncol_lst.append(str(column)+'_'+str(i+1))

    # dict = dict((el,0) for el in ncol_lst)
    l = len(maintin_list[0])
    print(l)
    for i in range(0,l):
        varnam = "d"+str(i)
        varnam =  [item[i] for item in maintin_list]
        dict[ncol_lst[i]] = varnam

    # print(dict)
    print('---------------------------------')

cnvt_data =  pd.DataFrame(dict)
print(cnvt_data.head())

# print(features.columns)
# print(ncol_lst)


# temp = features.iloc[:,1:-2]

# encoder= ce.BinaryEncoder(cols=temp,return_df=True)
#
# data_encoded=encoder.fit_transform(temp)
# data_encoded
# # de=pd.get_dummies(data=temp,drop_first=False)
#
# print(data_encoded)


