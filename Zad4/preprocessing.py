from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


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


def remove_outliers(df, columns, n_std):
    for column in columns:
        mean = df[column].mean()
        sd = df[column].std()
        df = df[(df[column] <= mean + (n_std * sd))]
        df = df[(df[column] >= mean - (n_std * sd))]
    return df


def split_dataset(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=42
    )

    return X_train, y_train, X_dev, y_dev, X_test, y_test
