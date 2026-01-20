import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
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
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
                "CatBoost": CatBoostRegressor(verbose=False, random_state=42),
            }

            params = {
                "Random Forest": {"n_estimators": [50, 100, 200]},
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200]
                },
                "Linear Regression": {},
                "XGBoost": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200]
                }
            }

            model_report = {}

            # 🔹 GridSearch for sklearn-compatible models
            for model_name, model in models.items():
                if model_name == "CatBoost":
                    continue  # Skip GridSearch for CatBoost

                logging.info(f"Tuning {model_name}")

                grid = GridSearchCV(
                    model,
                    params[model_name],
                    cv=3,
                    scoring="r2",
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                mse = mean_squared_error(y_test, y_pred)
                rmse = mse ** 0.5

                model_report[model_name] = {
                    "r2": r2,
                    "mae": mae,
                    "rmse": rmse,
                    "best_params": grid.best_params_,
                    "model": best_model
                }

            # 🔹 Train CatBoost separately (no GridSearch)
            logging.info("Training CatBoost separately")

            cat_model = models["CatBoost"]
            cat_model.fit(X_train, y_train)
            y_pred = cat_model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5

            model_report["CatBoost"] = {
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "best_params": "Default",
                "model": cat_model
            }

            # 🔹 Select Best Model
            best_model_name = max(model_report, key=lambda x: model_report[x]["r2"])
            best_model = model_report[best_model_name]["model"]

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Metrics: {model_report[best_model_name]}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, model_report[best_model_name]

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)
