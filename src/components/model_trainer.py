import sys
from dataclasses import dataclass
import os

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_all_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()
        #contains path mentioned in the above class

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, X_test, y_train, y_test = train_array[:,:-1], test_array[:,:-1], train_array[:,-1], test_array[:,-1]
            logging.info("train and test data split performed")
            
            n = len(train_array[0][:-1])
            #no of features

            models_dict = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'splitter': ['best', 'random'],
                    #best = selects based on highest information gain
                    #random - faster
                    'max_depth': [n, int(n/2), int(n/3)],
                    'min_samples_split': [4,6],
                    'min_samples_leaf':[2,4],
                    # 'max_features': [“auto”, “sqrt”, “log2”],
                    # 'min_impurity_decrease': []
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,64,256]
                },
                "Linear Regression": {
                    
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 10],
                    'weights': ['uniform', 'distance'],
                    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                    'p': [1,2]
                },
                "XGBRegressor": {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,64,256]
                },
                "CatBoosting Regressor": {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,64,256]
                }
            
            }

            model_report:dict=evaluate_all_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models_dict, params=params)
            #evaluate_all_models() -> func extracted from utils.py

            best_model_score = max(sorted(model_report.values()))
            #selects the highest R^2 score

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            #all models that have the highest R^2 score are stored

            best_model = models_dict[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found based on R^2 score: {0}.".format(best_model))

            save_object(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)