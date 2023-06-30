import sys
from dataclasses import dataclass
import os

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
    #this func is responsible for feature transformation
        try:
            num_features = ['writing_score', 'reading_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
            ])
            
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info('Numerical features standard scaling completed.')
            logging.info('Categorical features encoding completed.')

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_features),
                ('cat_pipeline', cat_pipeline, cat_features)
            ])

            return preprocessor
        except Exception as e:
            return CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logging.info('Read train and test data completed.')

            preprocessing_obj = self.get_data_transformer_obj()
            logging.info('Obtained preprocessing object.')

            target_feature = 'math_score'
            num_features = ['writing_score', 'reading_score']
            
            df_train_features = df_train.drop(columns=[target_feature], axis=1)
            df_train_target = df_train[target_feature]

            df_test_features = df_test.drop(columns=[target_feature], axis=1)
            df_test_target = df_test[target_feature]

            df_train_features_tr = preprocessing_obj.fit_transform(df_train_features) 
            df_test_features_tr = preprocessing_obj.transform(df_test_features)
            #fit_transform -> returns a np array
            logging.info('Preprocessing on training dataframe and test dataframe done.')
            
            train_arr = np.c_[df_train_features_tr, np.array(df_train_target)]
            test_arr = np.c_[df_test_features_tr, np.array(df_test_target)]
            #since we have transformed features as np array, we convert target as well to np array
            #for a list of np arrays, np.c_ -> concatenates along rows
            #for a 2d list of np arrays, np.c_ -> concatenates along cols

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            #used to save pickle file
            logging.info('saved preprocessing object.')

            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            raise CustomException(e, sys)