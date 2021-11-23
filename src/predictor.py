import argparse
import os
import sys
import pickle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
import pandas as pd
import numpy as np

try:
    from sklearn.externals import joblib
except:
    import joblib


def main(argv):
    train_raw = loadData('train')
    xTrain,yTrain = preprocessData(train_raw)
    val_raw = loadData('validation')
    xTest,yTest = preprocessData(val_raw)
    model = buildModel(xTrain,yTrain,xTest,yTest)
    
def loadData(file):
    print("Loading Datasets")
    if (file == 'train'):
        raw = pd.read_csv('reviews_train.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
    if (file == 'validation'):
        raw = pd.read_csv('reviews_validation.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
    if (file == 'test'):
        raw = pd.read_csv('reviews_test_attributes.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
    return raw
 
def preprocessData(raw):
    print("Preprocessing Data")
    try:
        vectorizer = pickle.load(open('vectorizer.pk', 'rb'))
    except:
        vectorizer = TfidfVectorizer(max_features=5000)

    corpora = raw['text_'].astype(str).values.tolist()
    vectorizer.fit(corpora)
    X = vectorizer.transform(corpora)
    X = X.toarray()
    x_cat = pd.DataFrame(raw['category'])['category'].astype('category').cat.codes
    X = np.insert(X,0,raw['rating'],1)
    X = np.insert(X,0,x_cat.to_numpy(),1)
    with open('vectorizer.pk', 'wb') as fout:
        pickle.dump(vectorizer, fout)

    print(X.shape,len(raw['real review?']))
    return X,raw['real review?']

def buildModel(x,y,valX,valY):
    print("Building Model")
    joblib_file = "LR_model.pkl"
    best_accuracy = 0
    for c in [100,75,50,25,12,10,6,3,1,.1,.01,.001,.0001]:
        print("C=",c)
        model = LogisticRegression(penalty="l1",tol=0.001, C=c, fit_intercept=True, solver="saga", intercept_scaling=1, random_state=42)
        model.fit(x, y)
        t_acc,t_mcc = scoreModel(model,x,y)
        accuracy,mcc = scoreModel(model,valX,valY)
        print("Training Scores: ", t_acc,t_mcc)
        print("Validation Scores: ", accuracy,mcc)

        print(f'Fraction of non-zero model parameters {np.sum(model.coef_!=0)+1/(len(model.coef_)+1)}')

        if mcc > best_accuracy:
            save_model(model, joblib_file)
            best_accuracy = accuracy
            print("Best acc c= ",c)

def scoreModel(model,x,y):
    pred = predict(model,x)
    acc = (pred==y).sum()/len(y)
    mcc = calc_mcc(pred,y)
    return acc,mcc

def finalPredictions():
    print("Making Final Predictions")
    raw = loadData('test')
    x,_ = preprocessData(raw)
    model = load_model("LR_model.pkl")
    pred = predict(model,x)
    pred_df = pd.DataFrame(x['ID'])
    pred_df['real review?'] = pred.values
    pred_df.to_csv('predictions.csv')
    
def predict(model,x):
    yhat = model.predict(x)
    pred = (yhat >= .5)+0
    return pred

if __name__ == '__main__':
    main(sys.argv[1:])