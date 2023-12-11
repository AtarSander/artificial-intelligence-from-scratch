import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def set_discrete_columns(df, columns):
    intervals = {}
    for column in columns:
        num_bins = int(np.sqrt(len(df[column].unique())))
        temp_intervals = pd.cut(df[column], bins=num_bins).unique().tolist()
        sorted_intervals = sorted([(interval.mid, interval) for interval in temp_intervals])
        intervals[column] = [str(interval) for (_, interval) in sorted_intervals]
        df[column+"_group"] = pd.cut(df[column], bins=num_bins,
                                     labels=False)
        df.drop(columns=column, inplace=True)
    return df, intervals


def remove_outliers(df, columns, n_std):
    for column in columns:
        mean = df[column].mean()
        sd = df[column].std()
        df = df[(df[column] <= mean+(n_std*sd))]
        df = df[(df[column] >= mean-(n_std*sd))]
    return df
