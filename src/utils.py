import pandas as pd
import numpy as np
import os
from joblib import load,dump
from sklearn.metrics import matthews_corrcoef,roc_auc_score

def normalize_col(col):
    return (col-col.mean())/col.std()
    # return (col-col.min())/(col.max()-col.min())

def normalize_df(df):
    norm = pd.DataFrame([])
    for col in df:
        norm[col] = normalize_col(df[col])
    return norm

def save_model(model,name):
    path = "./model/"+name
    create_dir_if_not_exist('./model')
    s = dump(model,path)

def load_model(name):
    path = "./model/"+name
    print("Loading Model: ", path)
    return load(path)

def create_dir_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def remove_outliers(arr):
    mean = np.mean(arr)
    standard_deviation = np.std(arr)
    distance_from_mean = abs(arr - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    trimmed = arr[not_outlier]
    return trimmed

def calc_if_outlier(df,column):
    arr = df[column]
    mean = np.mean(arr)
    standard_deviation = np.std(arr)
    distance_from_mean = abs(arr - mean)
    max_deviations = 2
    col = np.where(distance_from_mean < (max_deviations*standard_deviation),0,1)
    return col

def calc_mcc(pred,actual):
    mcc = matthews_corrcoef(actual,pred)
    return mcc

def calc_auc(pred,actual):
    auc = roc_auc_score(actual,pred)
    return auc

def split_train_test(x,y):
    df = pd.DataFrame(x)
    df = df.join(y)
    train = df.sample(frac=.9,replace=False)
    test = df.drop(train.index)

    x_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1:].to_numpy()
    y_train = y_train.reshape(len(y_train),)
    
    x_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1:].to_numpy()
    y_test = y_test.reshape(len(y_test),)


    return x_train,x_test,y_train,y_test