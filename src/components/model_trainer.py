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
            
            models_dict = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            model_report:dict=evaluate_all_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models_dict)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models_dict[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found based on R^2 score.")

            save_object(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)