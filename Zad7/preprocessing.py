import pandas as pd
import numpy as np


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
