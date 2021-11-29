import argparse
import os
import sys
import pickle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


print(f'pandas version {pd.__version__}')

# Sklearn version should be >= 0.23
print(f'Sklearn version {sklearn.__version__}')

try:
    from sklearn.externals import joblib
except:
    import joblib


def run(arguments):
    test_file = None
    train_file = None
    validation_file = None
    joblib_file = "LR_model.pkl"


    parser = argparse.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('-e', '--test', help='Test attributes (to predict)')
    group1.add_argument('-n', '--train', help='Train data')
    parser.add_argument('-v', '--validation', help='Validation data')

    args = parser.parse_args(arguments)

    Train = False
    Test = False
    Validation = False

    if args.test != None:
        print(f"Test file with attributes to predict: {args.test}")
        Test = True
            
    else:
        if args.train != None:
            print(f"Training data file: {args.train}")
            Train = True

        if args.validation != None:
            print(f"Validation data file: {args.validation}")
            Validation = True

    if Train and Validation:
        file_train = pd.read_csv(args.train,quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
        # real review? 1=real review, 0=fake review
        # Category: Type of product
        # Product rating: Rating given by user
        # Review text: What reviewer wrote
        feat = 5000
        # Create TfIdf vector of review using 5000 words as features
        vectorizer = TfidfVectorizer()
        # Transform text data to list of strings
        corpora = file_train['text_'].astype(str).values.tolist()
        # Obtain featurizer from data
        vectorizer.fit(corpora)
        # Create feature vector
        X = vectorizer.transform(corpora)
        X = X.toarray()
        x_cat = pd.DataFrame(file_train['category'])['category'].astype('category').cat.codes
        # X = X.join(file_train['rating'])
        X = np.insert(X,0,file_train['rating']/5,1)
        X = np.insert(X,0,x_cat.to_numpy(),1)
        print("Words used as features:")
        try:
            print(vectorizer.get_feature_names_out())
        except:
            print(vectorizer.get_feature_names())

        # Saves the words used in training
        with open('vectorizer.pk', 'wb') as fout:
            pickle.dump(vectorizer, fout)

        file_validation = pd.read_csv(args.validation,quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
        
        val_str = file_validation['text_'].astype(str).values.tolist()
        vectorizer.fit(val_str)
        val = vectorizer.transform(val_str)
        # val = val.join(file_validation['rating'])
        val = val.toarray()
        x_cat = pd.DataFrame(file_validation['category'])['category'].astype('category').cat.codes
        
        val = np.insert(val,0,file_validation['rating']/5,1)
        val = np.insert(val,0,x_cat.to_numpy(),1)
        
        best_accuracy = 0

        # TODO: The following code is performing regularization incorrectly.
        # Your goal is to fix the code.
        for C in [50,25,12,6,1,.01,.001,.0001]:
            print("C=",C)
            lr = LogisticRegression(penalty="elasticnet",tol=0.0001, C=C, fit_intercept=True, solver="saga", intercept_scaling=1, random_state=42,l1_ratio=.8)
            # You can safely ignore any "ConvergenceWarning" warnings
            lr.fit(X, file_train['real review?'])
            # Get logistic regression predictions
            # y_hat = lr.predict(X.toarray())
            y_hat = lr.predict(val)

            y_pred = (y_hat > 0.5) + 0 # + 0 makes it an integer

            # Accuracy of predictions with the true labels and take the percentage
            # Because our dataset is balanced, measuring just the accuracy is OK
            # accuracy = (y_pred == file_train['real review?']).sum() / file_train['real review?'].size
            accuracy = (y_pred == file_validation['real review?']).sum() / file_validation['real review?'].size
            print(f'Accuracy {accuracy}')
            print(f'Fraction of non-zero model parameters {np.sum(lr.coef_==0)+1}')
        
            if accuracy > best_accuracy:
                # Save logistic regression model
                joblib.dump(lr, joblib_file)
                best_accuracy = accuracy
                print("Best acc c= ",C)


    elif Test:
        # This part will be used to apply your model to the test data
        try:
            # DO NOT CHANGE. Needed for grading
            vectorizer = pickle.load(open('/autograder/submission/vectorizer.pk', 'rb'))
        except:
            vectorizer = pickle.load(open('vectorizer.pk', 'rb'))
            
        # Read test file
        file_test = pd.read_csv(args.test,quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
        # Transform text into list of strigs
        corpora = file_test['text_'].astype(str).values.tolist()
        # Use the words obtained in training to encode in testing
        X = vectorizer.fit_transform(corpora)
        X = X.toarray()
        x_cat = pd.DataFrame(file_test['category'])['category'].astype('category').cat.codes
        
        X = np.insert(X,0,file_test['rating']/5,1)
        X = np.insert(X,0,x_cat.to_numpy(),1)
        
        # Load trained logistic regression model
        try: 
            # DO NOT CHANGE. Needed for grading
            lr = joblib.load("/autograder/submission/"+joblib_file)
        except:
            lr = joblib.load(joblib_file)

        y_hat = lr.predict(X)

        y_pred = (y_hat > 0.5)+0 # + 0 makes it an integer

        # DO NOT CHANGE. Needed for grading
        for i,y in enumerate(y_pred):
            print(f"A\t{i}\t{y}")


    else:
        print("Training requires both training and validation data files. Test just requires test attributes.")

        
if __name__ == "__main__":
    run(sys.argv[1:])
    