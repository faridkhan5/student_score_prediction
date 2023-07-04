import os
import sys

import numpy as np
import pandas as pd
import dill 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            #dill - to create pickle file
    
    except Exception as e:
        raise CustomException(e, sys)
        

def evaluate_all_models(X_train, y_train, X_test, y_test, models, params):
#models -> dict = {"model name": model()}
#params -> nested dict = {"model name": {"param1":[], "param2":[]}}
    try:
        report = {}

        models_cnt = len(list(models))
        for i in range(models_cnt):
            model = list(models.values())[i]
            #new model is picked at each iter
            param = params[list(models.keys())[i]]

            gscv = GridSearchCV(model, param)
            gscv.fit(X_train, y_train)

            model.set_params(**gscv.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            #stores the R^2 score on test data, eg: report -> {LinearRegression: 0.84, ..., XGBoostRegressor: 0.91}
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
