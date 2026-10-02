# Import standard libraries
import pandas as pd
import numpy as np
import logging

# Import extra modules
from geopy.distance import great_circle
from sklearn.impute import SimpleImputer 

logger = logging.getLogger(__name__)
RANDOM_STATE = 42

def add_time_features(df):
    logger.debug('Adding time features...')
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    dt = df['transaction_time'].dt
    df['hour'] = dt.hour
    df['year'] = dt.year
    df['month'] = dt.month
    df['day_of_month'] = dt.day
    df['day_of_week'] = dt.dayofweek
    df.drop(columns='transaction_time', inplace=True)
    return df


def cat_encode(train, input_df, col):
    
    logger.debug('Encoding category: %s', col)
    new_col = col + '_cat'
    mapping = train[[col, new_col]].drop_duplicates()
    
    # Merge to initial dataset
    input_df = input_df.merge(mapping, how='left', on=col).drop(columns=col)
    
    return input_df


def add_distance_features(df):
    
    logger.debug('Calculating distances...')
    df['distance'] = df.apply(
        lambda x: great_circle(
            (x['lat'], x['lon']), 
            (x['merchant_lat'], x['merchant_lon'])
        ).km,
        axis=1
    )
    return df.drop(columns=['lat', 'lon', 'merchant_lat', 'merchant_lon'])


# Calculate means for encoding at docker container start
def load_train_data(input_df):
    target_col = 'target'
    categorical_cols = [
        'gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs'
    ]
    n_cats = 50

    train = input_df.copy().drop(
        columns=['name_1', 'name_2', 'street', 'post_code']
    )
    train = add_time_features(train)

    for col in categorical_cols:
        new_col = col + '_cat'

        temp_df = (
            train.groupby(col, dropna=False)[[target_col]]
            .count()
            .sort_values(target_col, ascending=False)
            .reset_index()
            .set_axis([col, 'count'], axis=1)
            .reset_index()
        )

        temp_df['index'] = temp_df.apply(
            lambda x: np.nan if pd.isna(x[col]) else x['index'],
            axis=1
        )

        temp_df[new_col] = [
            'cat_NAN' if pd.isna(x)
            else 'cat_' + str(x) if x < n_cats
            else f'cat_{n_cats}+'
            for x in temp_df['index']
        ]

        train = train.merge(
            temp_df[[col, new_col]],
            how='left',
            on=col
        )

    return add_distance_features(train)



def fit_preprocessor(train):
    raw_categories = [
        'gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs'
    ]
    categorical_cols = [col + '_cat' for col in raw_categories]
    categorical_cols.extend([
        'hour', 'year', 'month', 'day_of_month', 'day_of_week'
    ])
    continuous_cols = ['amount', 'population_city', 'distance']

    mappings = {
        col: train[[col, col + '_cat']].drop_duplicates()
        for col in raw_categories
    }

    means = {
        col: (
            train.groupby(col)[['target']]
            .mean()
            .reset_index()
            .rename(columns={'target': f'{col}_mean_enc'})
        )
        for col in categorical_cols
    }

    imputer = SimpleImputer(
        missing_values=np.nan,
        strategy='mean'
    )
    imputer.fit(train[continuous_cols])

    return {
        'mappings': mappings,
        'means': means,
        'imputer': imputer,
    }


def run_preproc(preprocessor, input_df):
    categorical_cols = [
        'gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs'
    ]
    continuous_cols = ['amount', 'population_city']
    drop_col = ['name_1', 'name_2', 'street', 'post_code']

    input_df = input_df.copy().reset_index(drop=True)
    input_df = input_df.drop(columns=drop_col)

    for col in categorical_cols:
        input_df = cat_encode(
            preprocessor['mappings'][col],
            input_df,
            col
        )

    input_df = add_time_features(input_df)

    categorical_cols = [col + '_cat' for col in categorical_cols]
    categorical_cols.extend([
        'hour', 'year', 'month', 'day_of_month', 'day_of_week'
    ])

    for col in categorical_cols:
        input_df[col] = input_df[col].fillna('cat_NAN')
        means_tb = preprocessor['means'][col]
        input_df = input_df.merge(means_tb, how='left', on=col)

    input_df = add_distance_features(input_df)
    continuous_cols.extend(['distance'])

    imputer = preprocessor['imputer']

    output_df = pd.concat([
        input_df.drop(columns=continuous_cols),
        pd.DataFrame(
            imputer.transform(input_df[continuous_cols]),
            columns=continuous_cols,
            index=input_df.index
        )
    ], axis=1)

    for col in continuous_cols:
        output_df[col + '_log'] = np.log(output_df[col] + 1)
        output_df.drop(columns=col, inplace=True)

    if 'feature_columns' in preprocessor:
        output_df = output_df[preprocessor['feature_columns']]

    return output_df