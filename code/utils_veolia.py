import pandas as pd
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
import numpy as np


from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit

def score_function(Y_true, Y_pred):

    nb_years = Y_true.shape[1]

    weights = np.array([0.6, 0.4])
    AUC_col = np.zeros(nb_years)
    for j in range(Y_true.shape[1]):
        AUC_col[j] = roc_auc_score(np.squeeze(Y_true[:, j]),
                                   np.squeeze(Y_pred[:, j]))
    AUC = np.dot(weights, AUC_col)
    return AUC

def print_repartition(output):
    try:
        output[output['2015'] == 1].shape[0]
        print "Repartition train: "
        print "2015: ", output[output['2015'] == 1].shape[0]
        print "2014: ", output[output['2014'] == 1].shape[0]
        print "Not Broken: ", output[(output['2015']!=1) & (output['2014']!=1)].shape[0]
    except:
        print "Repartition train: "
        print "Broken", sum(output)
        print "Not Broken: ", len(output)-sum(output)

def load_data():
   '''
   Loader for data, returns:
   input_train, output_train, input_submission
   '''

   DATA_PATH = '../data/'
   INPUT_TRAIN = DATA_PATH+'input_train.csv'
   OUTPUT_TRAIN = DATA_PATH+'output_train.csv'
   INPUT_SUBMISSION = DATA_PATH+'input_test.csv'

   input_train = pd.read_csv(INPUT_TRAIN,index_col='Id')
   output_train = pd.read_csv(OUTPUT_TRAIN,sep=';',index_col='Id')
   input_submission = pd.read_csv(INPUT_SUBMISSION ,index_col='Id')

   return input_train, output_train, input_submission

def basic_preprocessing(dataframe):
   '''
   Does basic preprocessing: changes categorical data into dummies + replace na by -1 in the last failure col.
   Returns dataframe
   '''
   X = dataframe

   # Categorical data
   X = pd.concat([X,pd.get_dummies(X['Feature1'])],axis=1)
   X = pd.concat([X,pd.get_dummies(X['Feature2'])],axis=1)
   X = pd.concat([X,pd.get_dummies(X['Feature4'])],axis=1)

   X = X.drop(["Feature1","Feature2","Feature4"],axis=1)

   X = X.fillna(-1)
   return X

def preprocess(dataframe,year=2014, more_features = False):
    X = dataframe

    # The relevant value is the age of the pipes
    X['Age'] = year - X['YearConstruction']
    X = X.fillna(10000)

    # How long has it been since last failure
    X['YearsOldLastFailure'] = year - X['YearLastFailureObserved']

    # Categorical data
    X = pd.concat([X,pd.get_dummies(X['Feature1'])],axis=1)
    X = pd.concat([X,pd.get_dummies(X['Feature2'])],axis=1)
    X = pd.concat([X,pd.get_dummies(X['Feature4'])],axis=1)

    X = X.drop(["YearConstruction","YearLastFailureObserved","Feature1","Feature2","Feature4"],axis=1)

    X['Feature3'] = normalize(X['Feature3']).tolist()[0]
    X['Length'] = normalize(X['Length']).tolist()[0]
    X['Age'] = normalize(X['Age']).tolist()[0]
    X['YearsOldLastFailure'] = normalize(X['YearsOldLastFailure']).tolist()[0]

    if more_features:
        col = X.columns[4:]
        for c in col:
            for u in col:
                X[c+u] = X[c]*X[u]
    return X

def preprocess_output(dataframe,year=2014):
    '''
    Selects the right colum for the year studied
    '''
    return dataframe[str(year)]


def split_train_test_Kfold(output_raw, input_preprocessed, year):

    output_bool = (output_raw[str(year)])>0
    skf = StratifiedKFold(output_bool,n_folds=2, shuffle=True)
    for k, [train_index, test_index] in enumerate(skf):
        #print("TRAIN:", train_index+1, "TEST:", test_index+1)
        test_index = test_index+1
        train_index = train_index+1
        input_train, input_test = input_preprocessed.loc[train_index], input_preprocessed.loc[test_index]
        output_train, output_test = output_raw.loc[train_index], output_raw.loc[test_index]

    return input_train, output_train, input_test, output_test

def split_train_test_stratified_shuffle(output_raw, input_preprocessed, test_size, year):
    output_bool = (output_raw[str(year)])>0
    sss = StratifiedShuffleSplit(output_bool, n_iter=1, test_size=test_size)
    for train_index, test_index in sss:
        test_index = test_index+1
        train_index = train_index+1
        input_train, input_test = input_preprocessed.loc[train_index], input_preprocessed.loc[test_index]
        output_train, output_test = output_raw.loc[train_index], output_raw.loc[test_index]
        
    return input_train, output_train, input_test, output_test

def data_augmentation_basic(input_train, output_train, year='both', repetitions = 6):

    if type(year) == int:
        ID= output_train[output_train[str(year)]==1].index.tolist()
        # Augment data with breaks to counter unbalanced dataset only for training
        input_train_duplicate = input_train
        output_train_duplicate = output_train
        REPETITIONS = repetitions
        for k in range(0,REPETITIONS):
            input_train_duplicate = pd.concat([input_train_duplicate.loc[ID],input_train_duplicate])
            output_train_duplicate = pd.concat([output_train_duplicate.loc[ID],output_train_duplicate])

        return input_train_duplicate, output_train_duplicate[str(year)]
    else:
        # Select the rows with a canalisation breaks
        ID_2014 = output_train[output_train['2014']==1].index.tolist()
        ID_2015 = output_train[output_train['2015']==1].index.tolist()
        # Augment data with breaks to counter unbalanced dataset only for training
        input_train_duplicate = input_train
        output_train_duplicate = output_train
        REPETITIONS = repetitions
        for k in range(0,REPETITIONS):
            input_train_duplicate = pd.concat([input_train_duplicate.loc[ID_2014],input_train_duplicate.loc[ID_2015],input_train_duplicate])
            output_train_duplicate = pd.concat([output_train_duplicate.loc[ID_2014],output_train_duplicate.loc[ID_2015],output_train_duplicate])

        return input_train_duplicate, output_train_duplicate
