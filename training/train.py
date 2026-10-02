import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fraud_detector' / 'src'))

from preprocessing import (
    load_train_data,
    fit_preprocessor,
    run_preproc,
)


def main():
    data = pd.read_csv(ROOT / 'data' / 'train.csv')
    data['transaction_time'] = pd.to_datetime(data['transaction_time'])
    data = data.sort_values('transaction_time').reset_index(drop=True)

    # Последние 20% транзакций оставляем для проверки.
    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    validation = data.iloc[split:].copy()

    # Все статистики вычисляем только по обучающей части.
    reference = load_train_data(train)
    preprocessor = fit_preprocessor(reference)

    x_train = run_preproc(
        preprocessor,
        train.drop(columns='target')
    )
    preprocessor['feature_columns'] = x_train.columns.tolist()

    x_validation = run_preproc(
        preprocessor,
        validation.drop(columns='target')
    )

    categorical_cols = [
        'gender_cat', 'merch_cat', 'cat_id_cat',
        'one_city_cat', 'us_state_cat', 'jobs_cat',
        'hour', 'year', 'month', 'day_of_month', 'day_of_week'
    ]

    for col in categorical_cols:
        x_train[col] = x_train[col].astype(str)
        x_validation[col] = x_validation[col].astype(str)

    y_train = train['target'].astype(int)
    y_validation = validation['target'].astype(int)

    model = CatBoostClassifier(
        iterations=100,
        depth=4,
        learning_rate=0.1,
        loss_function='Logloss',
        auto_class_weights='Balanced',
        cat_features=categorical_cols,
        task_type='CPU',
        thread_count=4,
        random_seed=42,
        allow_writing_files=False,
        verbose=20,
    )
    model.fit(x_train, y_train)

    scores = model.predict_proba(x_validation)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_validation, scores
    )
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )

    best_index = int(np.argmax(f1))
    threshold = float(thresholds[best_index])

    models_dir = ROOT / 'fraud_detector' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(models_dir / 'my_catboost.cbm'))
    joblib.dump(
        preprocessor,
        models_dir / 'preprocessor.joblib'
    )

    metadata = {
        'threshold': threshold,
        'validation_roc_auc': float(roc_auc_score(y_validation, scores)),
        'validation_f1': float(f1[best_index]),
    }

    (models_dir / 'model_info.json').write_text(
        json.dumps(metadata, indent=2),
        encoding='utf-8',
    )

    print(json.dumps(metadata, indent=2))
    print('Модель и параметры препроцессинга сохранены.')


if __name__ == '__main__':
    main()