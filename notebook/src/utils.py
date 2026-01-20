import os
import pickle
import sys
from sklearn.model_selection import GridSearchCV
from notebook.src.exception import CustomException
from notebook.src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
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

            from sklearn.metrics import r2_score

            r2 = r2_score(y_test, y_pred)

            report[model_name] = {
                "r2": r2,
                "best_params": grid.best_params_,
                "model": best_model
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
