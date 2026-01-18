import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PROJECT_ROOT)

from notebook.src.exception import CustomException
from notebook.src.logger import logging
from notebook.src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(objective="reg:squarederror"),
                "CatBoost": CatBoostRegressor(verbose=False),
            }

            model_report = {}

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                mse = mean_squared_error(y_test, y_pred)
                rmse = mse ** 0.5   # ✅ manual RMSE

                model_report[model_name] = (r2, mae, rmse)

            best_model_name = max(model_report, key=lambda name: model_report[name][0])
            best_model = models[best_model_name]
            best_model_metrics = model_report[best_model_name]

            logging.info(f"Best model: {best_model_name} | R2: {best_model_metrics[0]}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_metrics

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)
