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

RATING_SPLIT = False

def main(argv):
    if argv[0]=='--train':
        train_raw = loadData('train')
        test_raw = loadData('test')
        generate_vectorizer(test_raw)
        if (RATING_SPLIT == False):
            xTrain,xTest,yTrain,yTest  = preprocessData(train_raw)
            # xTrain,yTrain = preprocessData(train_raw,split=False)
            # xTest,yTest = preprocessData(loadData('validation'),split=False)
            buildModel(xTrain,yTrain,xTest,yTest)
            return
        
        else:   
            models = {}
            for rating in [5,4,3,2,1]:
                print("\nRating=", rating)
                try:
                    model = load_model("model_"+str(rating)+".pkl")
                    models[rating]=model
                except:
                    raw_rating = train_raw.loc[train_raw['rating']==rating]
                    raw_rating = raw_rating.reset_index()
                    xTrain,xTest,yTrain,yTest  = preprocessData(raw_rating)
                    buildModel(xTrain,yTrain,xTest,yTest,"model_"+str(rating)+".pkl")
                    model = load_model("model_"+str(rating)+".pkl")
                    models[rating] = model
            finalPredictions(models)
            return
    elif argv[0]=='--test':
        finalPredictions()
    else:
        print("ERROR, unknown argument: ", argv)
    
def loadData(file):
    print("Loading Datasets")
    if (file == 'train'):
        raw = pd.read_csv('./data/reviews_train.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
    if (file == 'validation'):
        raw = pd.read_csv('./data/reviews_validation.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
    if (file == 'test'):
        raw = pd.read_csv('./data/reviews_test_attributes.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
    return raw
 
def preprocessData(raw,split=True):
    print("Preprocessing Data")
    load = False
    try:
        vectorizer = pickle.load(open('./model/vectorizer.pk', 'rb'))
        load = True
        print("\tLoaded Vectorizer")
    except:
        print("\tERROR: Could not find Vectorizer, generating new vectorizer")
        vectorizer = TfidfVectorizer(max_features=6000)

    corpora = raw['text_'].astype(str).values.tolist()
    if (load is False):
        print("\tFitting Vectorizer")
        vectorizer.fit(corpora)
    X = vectorizer.transform(corpora)
    X = X.toarray()
    X = np.insert(X,0,raw['rating'],1)
    if load is False:
        with open('./model/vectorizer.pk', 'wb') as fout:
            pickle.dump(vectorizer, fout)

    if split:
        x_train,x_test,y_train,y_test = split_train_test(X,raw['real review?'])
        return x_train,x_test,y_train,y_test
    else:
        if ('real review?' not in raw.columns):
            return X
        return X,raw['real review?']
        

def buildModel(x,y,valX,valY,name="LR_model.pkl"):
    print("Building Model")
    joblib_file = name
    best_accuracy = 0
    for c in [100,25,6,3,1]:
        print("\n\tC = ", c)
        model = LogisticRegression(penalty="l2",tol=0.001,class_weight='auto' ,C=c, fit_intercept=True, solver="saga", intercept_scaling=1, random_state=42)
        model.fit(x, y)
        t_acc,t_mcc,t_auc = scoreModel(model,x,y)
        accuracy,mcc,auc = scoreModel(model,valX,valY)

        print("\tTraining Scores: ", t_acc,t_mcc,t_auc)
        print("\tValidation Scores: ", accuracy,mcc,auc)

        if auc > best_accuracy:
            save_model(model, joblib_file)
            best_accuracy = accuracy
            print("\tBest acc c= ",c)
            print(f'\tNumber of non-zero model parameters {np.sum(model.coef_!=0)}')
            
    

def scoreModel(model,x,y):
    pred = predict(model,x)
    acc = (pred==y).sum()/len(y)
    mcc = calc_mcc(pred,y)
    auc = calc_auc(pred,y)
    return acc,mcc,auc

def finalPredictions(models=None):
    print("Making Final Predictions")
    raw = loadData('test')
    x,_ = preprocessData(raw,split=False)
    if models == None:
        model = load_model("LR_model.pkl")
        pred = predict(model,x)
    else:
        x = np.c_[x,raw['rating']] # add rating column to row
        def predict_row(row):
            model = models[row[len(row)-1]]
            pred = predict(model,[row.iloc[:-1]])
            return pred[0]
        df = pd.DataFrame(x)
        pred = df.apply(predict_row,axis=1)

    pred_df = pd.DataFrame()
    pred_df['real review?'] = pred
    print(pred_df)    
    pred_df.to_csv('predictions.csv')
    
def predict(model,x):
    yhat = model.predict_proba(x)[:,1]
    pred = (yhat >= .5)+0
    return pred

def generate_vectorizer(raw):
    print("Generating Vectorizer")
    vectorizer = TfidfVectorizer(max_features=6000)
    corpora = raw['text_'].astype(str).values.tolist()
    vectorizer.fit(corpora)
    with open('./model/vectorizer.pk', 'wb') as fout:
        pickle.dump(vectorizer, fout)

if __name__ == '__main__':
    main(sys.argv[1:])