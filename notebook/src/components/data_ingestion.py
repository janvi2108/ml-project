import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PROJECT_ROOT)

from notebook.src.exception import CustomException
from notebook.src.logger import logging
from notebook.src.components.data_transformation import DataTransformation
from notebook.src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        print("Running Data Ingestion...")
        logging.info("Data Ingestion started")

        try:
            csv_path = os.path.join(PROJECT_ROOT, "notebook", "data", "stud.csv")
            df = pd.read_csv(csv_path)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            print("Data Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            print("ERROR IN DATA INGESTION:", e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    print("Running Data Transformation...")

    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    print("Running Model Training...")

    model_trainer = ModelTrainer()
    best_model, metrics = model_trainer.initiate_model_trainer(train_arr, test_arr)

    print("\n========== REGRESSION RESULTS ==========")
print("Best Model :", best_model)
print("R2 Score   :", metrics[0])
print("MAE        :", metrics[1])
print("RMSE       :", metrics[2])
print("=======================================\n")


print("Pipeline Finished Successfully!")
