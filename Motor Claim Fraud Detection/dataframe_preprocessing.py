import os

from Utils.dataframe_preprocessing_utils import change_column_type_to_int, change_column_type_to_category, \
    category_column_ordinal_encoding, category_column_one_hot_encoding
from Utils.dataframe_preprocessing_utils import drop_duplicate_rows_from_dataframe, change_column_type_to_datetime, \
    change_column_type_to_float, target_column_label_encoding, \
    drop_columns_from_dataframe
from get_values_from_config_file import drop_duplicate_records, date_columns, float_columns, int_columns, cat_columns, \
    one_hot_encoding_columns, target_column, root_path, dataset_folder, \
    final_dataset_folder, drop_columns
from get_values_from_config_file import ordinal_categories_list, ordinal_encoding_columns


def preprocess_dataframe(input_df):
    # Strip spaces from column names
    input_df.columns = input_df.columns.str.strip()

    # Drop duplicate rows from dataframe
    if drop_duplicate_records:
        input_df = drop_duplicate_rows_from_dataframe(input_df)
    else:
        input_df = input_df

    # Change the column data type to date
    input_df = change_column_type_to_datetime(input_df, date_columns)
    # Change the column data type to float
    input_df = change_column_type_to_float(input_df, float_columns)
    # Change the column data type to int64
    input_df = change_column_type_to_int(input_df, int_columns)
    # Change the column data type to category
    input_df = change_column_type_to_category(input_df, cat_columns)

    # Ordinal Encoding for Ordinal Columns
    input_df = category_column_ordinal_encoding(input_df, ordinal_encoding_columns, ordinal_categories_list)

    # One Hot Encoding for Nominal Columns
    input_df = category_column_one_hot_encoding(input_df, one_hot_encoding_columns)

    input_df = target_column_label_encoding(input_df, target_column)
    input_df = drop_columns_from_dataframe(input_df, drop_columns)

    dataset_path = os.path.join(root_path, dataset_folder)
    final_dataset_path = os.path.join(dataset_path, final_dataset_folder)
    if not os.path.exists(final_dataset_path):
        os.makedirs(final_dataset_path)

    input_df.to_csv(f'{final_dataset_path}/final.csv', index=False)
    return input_df


if __name__ == '__main__':
    pass
