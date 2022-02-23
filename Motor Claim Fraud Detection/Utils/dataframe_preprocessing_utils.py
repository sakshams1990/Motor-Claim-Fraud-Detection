import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler


# Drop duplicate rows from dataframe
def drop_duplicate_rows_from_dataframe(df):
    duplicate_rows = df.duplicated().sum()
    print(f'\nThere are {duplicate_rows} duplicate records in the dataframe..')
    if duplicate_rows > 0:
        df = df.drop_duplicates(keep='first')
    return df


# Drop unwanted columns from dataframe
def drop_columns_from_dataframe(df, list_of_cols):
    df = df.drop(list_of_cols, axis=1)
    return df


# Change column datatype to datetime
def change_column_type_to_datetime(df, list_of_cols):
    df[list_of_cols] = df[list_of_cols].apply(pd.to_datetime, format='%Y-%m-%d', errors='ignore')
    return df


# Change column datatype to float
def change_column_type_to_float(df, list_of_cols):
    df[list_of_cols] = df[list_of_cols].astype('float')
    return df


# Change column datatype to integer
def change_column_type_to_int(df, list_of_cols):
    df[list_of_cols] = df[list_of_cols].astype('int64')
    return df


# Change column datatype to object
def change_column_type_to_category(df, list_of_cols):
    df[list_of_cols] = df[list_of_cols].astype('category')
    return df


# Label Encoding for target variable
def target_column_label_encoding(df, target_col_name):
    le = LabelEncoder()
    df[target_col_name] = le.fit_transform(df[target_col_name])
    return df


# One hot encoding for Nominal column
def category_column_one_hot_encoding(df, list_of_cols):
    df = pd.get_dummies(df, columns=list_of_cols, drop_first=True)
    return df


# Ordinal Encoding for Ordinal column
def category_column_ordinal_encoding(df, list_of_cols, list_of_categories):
    oe = OrdinalEncoder(categories=list_of_categories, dtype='int64')
    df[list_of_cols] = oe.fit_transform(df[list_of_cols])
    return df


# Count frequency Encoding for high cardinality column
def category_column_count_frequency_encoding(df, list_of_cols):
    for col in list_of_cols:
        fe = df.groupby(col).size()
        df.loc[:, col + '_count'] = df[col].map(fe)
    df = drop_columns_from_dataframe(df, list_of_cols)
    return df


# Removal of class imbalance using SMOTE
def class_imbalance_SMOTE(X, y, random_state):
    sm = SMOTE(random_state=random_state)
    X_smote, y_smote = sm.fit_resample(X, y)
    return X_smote, y_smote


# Removal of class imbalance using undersampling
def class_imbalance_undersampling(X, y, random_state):
    rus = RandomUnderSampler(random_state=random_state, replacement=True)
    X_rus, y_rus = rus.fit_resample(X, y)
    return X_rus, y_rus


def standard_scaler_normalization(df):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    return scaled_df


def min_max_scaler_normalization(df):
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df)
    return scaled_df


if __name__ == '__main__':
    pass
