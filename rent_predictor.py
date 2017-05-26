import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
import math
from scipy.stats import norm

# Dictionaries and Lists #
myDictNA = {
'uf1_1':[8],
'uf1_2':[8],
'uf1_3':[8],
'uf1_4':[8],
'uf1_5':[8],
'uf1_6':[8],
'uf1_7':[8],
'uf1_8':[8],
'uf1_9':[8],
'uf1_10':[8],
'uf1_11':[8],
'uf1_12':[8],
'uf1_13':[8],
'uf1_14':[8],
'uf1_15':[8],
'uf1_16':[8],
'uf1_35':[8],
'uf1_17':[8],
'uf1_18':[8],
'uf1_19':[8],
'uf1_20':[8],
'uf1_21':[8],
'uf1_22':[8],
'sc23':[8],
'sc24':[8],
'sc36':[3,8],
'sc37':[3,8],
'sc38':[3,8],
'sc147':[3,8],
'sc173':[3,8,9],
'sc171':[3,8],
'sc198':[8],
'sc187':[8],
'sc190':[8],
'sc191':[8],
'sc192':[8],
'rec21':[8]
}
vacantList = [
#'recid',
'boro',
'uf1_1',
'uf1_2',
'uf1_3',
'uf1_4',
'uf1_5',
'uf1_6',
'uf1_7',
'uf1_8',
'uf1_9',
'uf1_10',
'uf1_11',
'uf1_12',
'uf1_13',
'uf1_14',
'uf1_15',
'uf1_16',
'uf1_35',
'uf1_17',
'uf1_18',
'uf1_19',
'uf1_20',
'uf1_21',
'uf1_22',
'sc23',
'sc24',
'sc36',
'sc37',
'sc38',
'sc30',
'sc518',
'uf49',
'sc520',
'uf33',
'uf51',
'sc522',
'sc553',
'sc555',
'sc523',
'sc524',
'sc525',
'sc526',
'sc527',
'sc528',
'sc529',
'sc530',
'sc531',
'sc532',
'sc533',
'sc534',
'sc535',
'uf31',
'uf19',
'new_csr',
'nusc',
'sc26',
'uf23',
'rec62',
'rec64',
'uf32',
'rec21',
'cd',
'seqno',
'fw',
'hflag6',
'hflag3',
'hflag15',
'hflag17',
'hflag8',
'hflag5'
]
additionalList=[
'uf48',
'sc147',
'uf11',
'sc149',
'sc173',
'sc171',
'sc150',
'sc151',
'sc152',
'sc153',
'sc155',
'sc156',
'sc158',
'sc198',
'sc187',
'sc190',
'sc191',
'sc192',
'sc193',
]
# delete if missing data is more
myListDel=['uf1_2','uf1_6','uf1_15','hflag3','seqno'
#'fw'
]
myListDel2=['sc187_1.0','sc187_2.0'
#'cd_9',#'cd_10',
]
continous=['fw','uf17'
#'seqno',
]

def load_data(file,split=0.33):

    #file is string/URL
    df = pd.read_csv(file)

    # Removing data points based on y-variable:
    df = df[df.uf17!=7999]
    df = df[df.uf17!=99999]
    df = data_clean(df)

    # Passing everything as a categorical variable
    cont=[col for col in df.columns if col not in continous]
    df[cont]=df[cont].astype('object')

    # Load only columns that match both Vacant and Occupied layouts
    features_to_use=[col for col in df.columns if col in vacantList or col in additionalList]
    dfx=df[features_to_use]
    dfy=pd.Series.to_frame(df['uf17'])
    # Cleaning some data
    #dfx = data_clean(dfx)
    scaler=StandardScaler().fit_transform
    dfx['fw']=scaler(dfx['fw'])


    # Imputing some data
    dfx=DataFrameImputer().fit_transform(dfx)
    cont=[col for col in dfx.columns if col not in continous]
    dfx[cont]=dfx[cont].astype('object')
    X_train, X_test, y_train, y_test= train_test_split(dfx, dfy,test_size=split, random_state=0)
    print("## Data is fully loaded")

    return X_train, X_test, y_train, y_test

def data_clean(df):
    for item in myDictNA.items():
        for i in item[1]:
            df[item[0]].ix[df[item[0]]==i]=np.nan

    return df

def removeFromList(df,deletelist):
    for i in deletelist:
        del df[i]
    return df

def getDummy(df):
    df=pd.get_dummies(df)
    return df


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def score_rent():

    # Data Import
    url        = 'https://ndownloader.figshare.com/files/7586326'

    ## DATA MODELING

    # Modify Data
    '''Split the data into 70% training and 30% test data.'''
    X_train, X_test, y_train, y_test = load_data(url,split=0.30)


    # Remove Insignificant variable
    X_train=removeFromList(X_train,myListDel)
    X_test=removeFromList(X_test,myListDel)

    # One Hot Encoding
    X_train=getDummy(X_train)
    X_test=getDummy(X_test)

    # Iterative Parameter Selection
    X_train=removeFromList(X_train,myListDel2)
    X_test=removeFromList(X_test,myListDel2)

    # Perform Lasso to remove less performing variables:
    lasso = Lasso().fit(X_train,y_train)

    X_train=X_train.iloc[:,lasso.coef_!=0]
    X_test=X_test.iloc[:,lasso.coef_!=0]

    lasso = Lasso().fit(X_train,y_train)
    print("Training set score: {:.2f}".format(lasso.score(X_train,y_train)))
    print("Test set score: {:.2f}".format(lasso.score(X_test,y_test)))
    print("Number of features used: {}".format(np.sum(lasso.coef_!=0)))

    return lasso.score(X_train,y_train)

def predict_rent():
    # Data Import
    url        = 'https://ndownloader.figshare.com/files/7586326'

    # Modify Data
    '''Split the data into 70% training and 30% test data.'''
    X_train, X_test, y_train, y_test = load_data(url,split=0.30)

    # Remove Insignificant variable
    X_train=removeFromList(X_train,myListDel)
    X_test=removeFromList(X_test,myListDel)

    # One Hot Encoding
    X_train=getDummy(X_train)
    X_test=getDummy(X_test)

    # Iterative Parameter Selection
    X_train=removeFromList(X_train,myListDel2)
    X_test=removeFromList(X_test,myListDel2)


    # Perform Lasso to remove less performing variables:
    lasso = Lasso().fit(X_train,y_train)

    X_train=X_train.iloc[:,lasso.coef_!=0]
    X_test=X_test.iloc[:,lasso.coef_!=0]

    lasso = Lasso().fit(X_train,y_train)

    test_data = X_test
    predicted_labels = lasso.predict(test_data)
    true_labels = y_test


    return np.asarray(test_data), np.asarray(true_labels), np.asarray(predicted_labels)

score_rent()

predict_rent()
