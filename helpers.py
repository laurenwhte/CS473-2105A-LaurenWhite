import pandas as pd


# CS473-2105A Data Mining
# Prof. Chintan Thakkar
# December 12, 2021
# Lauren White

# This program is being created to ingest the Titanic dataset,
# display details about each of the features, and apply a k-means
# clustering algorithm to predict survival.

# HELPER METHODS
# I have written a few functions to check for nulls, unique values, and ranges in the data.

# Method to check for NaN values.
def null_values(df):
    total = df.isnull().sum().sort_values(ascending=False)  # Add up how many missing data points there are.
    percent_1 = df.isnull().sum() / df.isnull().count() * 100  # Divide that by the total records to get % missing.
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)  # Format % and sort descending.
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])  # Zip/concat the data into a new table
    print("\nCount and percentage of null values in each column: \n")
    print(missing_data.head(14))  # View it


# Method to return unique values.
def unique_values(df):
    print("\nCount of unique values in each column: \n")
    print(df.nunique())


# Method to return the range of values
def value_range(df):
    min_val = df.min()
    max_val = df.max()
    range_data = pd.concat([min_val, max_val], axis=1, keys=['Min', 'Max'])  # Zip/concat the data into a new table
    print("\nMinimum and maximum values in each column: \n")
    print(range_data.head(14))  # View it
