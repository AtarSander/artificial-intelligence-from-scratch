import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def remove_outliers(df, columns, n_std):
    for column in columns:
        mean = df[column].mean()
        sd = df[column].std()
        df = df[(df[column] <= mean + (n_std * sd))]
        df = df[(df[column] >= mean - (n_std * sd))]
    return df


def set_discrete_columns(df, columns):
    intervals = {}
    for column in columns:
        numbered_intervals = {}
        if column == "age":
            num_bins = int(np.sqrt(np.sqrt(len(df[column].unique()))))
        else:
            num_bins = int(np.sqrt(len(df[column].unique())))
        temp_intervals = pd.cut(df[column], bins=num_bins).unique().tolist()
        sorted_intervals = sorted(
            [(interval.mid, interval) for interval in temp_intervals]
        )
        for i, (_, interval) in enumerate(sorted_intervals):
            numbered_intervals[i] = str(interval)
        intervals[column] = numbered_intervals
        df[column] = pd.cut(df[column], bins=num_bins, labels=False)
    return df, intervals


def split_data(dataset, target, test_size=0.2, validation_size=0.2, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        dataset.drop(columns=[target]),
        dataset[target],
        test_size=test_size + validation_size,
        random_state=random_state,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_size / (test_size + validation_size),
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
