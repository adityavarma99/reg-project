import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:                      
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")      ## to save the model pickel file into some path

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''

        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"numerical columns: {numerical_columns}")

            #logging.info("Numerical columns standard scaling completed")
            #logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
            
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException
            

    def initiate_data_transformation(self,train_path,test_path):     ## we are getting train_path and test_path from data ingestion

        try:
             train_df=pd.read_csv(train_path)
             test_df=pd.read_csv(test_path)

             logging.info("Read train and test data completely")

             logging.info("Obtaining preprocessing object")

             preprocessing_obj=self.get_data_transformer_object()

             target_column_name="math_score"                                      # y
             numerical_columns = ['writing_score', 'reading_score']

             input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1) # X train
             target_feature_train_df=train_df[target_column_name]                      # y train

             input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)   # X test
             target_feature_test_df=test_df[target_column_name]                        # y test

             logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

             input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)  # fit and transform train data
             input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)        # transform test data

             train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]   # np.c_ = column stacking of input train array and target value
             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

             logging.info(f"Saved preprocessing object.")

             save_object(                                                                      # ave object function is from utils.py
                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
                 obj=preprocessing_obj
             )

             return (
                 train_arr,
                 test_arr,
                 self.data_transformation_config.preprocessor_obj_file_path,
             )



        except Exception as e:
           raise CustomException(e,sys)

    