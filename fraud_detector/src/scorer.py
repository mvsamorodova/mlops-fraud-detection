import json
import logging
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

model = CatBoostClassifier(task_type="CPU", thread_count=4)
model.load_model(str(MODELS_DIR / "my_catboost.cbm"))

with (MODELS_DIR / "model_info.json").open(encoding="utf-8") as file:
    model_th = json.load(file)["threshold"]


def make_pred(dt, source_info="kafka"):
    dt = dt.copy()

    categorical_cols = [
        "hour", "year", "month", "day_of_month", "day_of_week",
        "gender_cat", "merch_cat", "cat_id_cat",
        "one_city_cat", "us_state_cat", "jobs_cat",
    ]

    for col in categorical_cols:
        dt[col] = dt[col].astype(str)

    scores = model.predict_proba(dt)[:, 1]

    submission = pd.DataFrame({
        "score": scores,
        "fraud_flag": (scores >= model_th).astype(int),
    })

    logger.info("Prediction complete for data from %s", source_info)
    return submission