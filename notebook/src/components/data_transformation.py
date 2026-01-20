import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np  
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler 

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PROJECT_ROOT)

from notebook.src.exception import CustomException
from notebook.src.logger import logging
from notebook.src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            logging.error("Error in creating data transformer")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Loading train & test data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column_name = "math_score"

            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            logging.info("Applying preprocessing")

            preprocessor_obj = self.get_data_transformer_object()

            X_train_arr = preprocessor_obj.fit_transform(X_train)
            X_test_arr = preprocessor_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessor saved successfully")

            return train_arr, test_arr

        except Exception as e:
            logging.error("Error in Data Transformation")
            raise CustomException(e, sys)
