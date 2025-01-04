import os
import sys
import dill

import numpy as np
import pandas as pd

import pickle

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

## save object is from data transformation
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
# evaluate models is from model trainer
def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    try:
        report = {}

        for i in range(len(list(models))):              # go through each and every model
            model=list(models.values())[i]

            ## go through each parameter and do grid search Validation
            para=param[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)                         ## fit X train and y train with grid search for hyper parameter tunning
            model.set_params(**gs.best_params_)


            model.fit(X_train, y_train)                  # train the model of X_train and y_train

            y_train_pred = model.predict(X_train)        # Predict X train
            y_test_pred = model.predict(X_test)          # Predict X test

            train_model_score=r2_score(y_train, y_train_pred)
            test_model_score=r2_score(y_test, y_test_pred)                      # r2 score of test model is appended inside report

            report[list(models.keys())[i]] = test_model_score
        
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
## load_object is from predict pipeline
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

    
    