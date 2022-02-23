import os

from get_values_from_config_file import root_path, dataset_folder, preprocessed_dataset_folder


def manual_feature_engineering_of_data(input_df):
    """
    months_as_customer_bins = [0, 100, 200, 300, 400, 500]
    months_as_customer_labels = ['very new customer', 'new customer', 'regular customer', 'old customer', 'very old customer']
    input_df['months_as_customer_labels'] = pd.cut(input_df['months_as_customer'], bins=months_as_customer_bins, labels=months_as_customer_labels,
                                                   include_lowest=True)

    age_bins = [1, 19, 30, 50, 100]
    age_labels = ['minor', 'young', 'old', 'very old']
    input_df['age_labels'] = pd.cut(input_df['age'], bins=age_bins, labels=age_labels)
    """

    input_df = input_df.replace(to_replace='?', value='UNKNOWN')
    dataset_path = os.path.join(root_path, dataset_folder)
    preprocessed_dataset_path = os.path.join(dataset_path, preprocessed_dataset_folder)
    if not os.path.exists(preprocessed_dataset_path):
        os.makedirs(preprocessed_dataset_path)

    input_df.to_csv(f'{preprocessed_dataset_path}/preprocessed.csv', index=False)
    return input_df


if __name__ == '__main__':
    pass
