import json
import os.path

import pandas as pd

from get_values_from_config_file import cat_columns, target_column, model_resources, root_path

resources_path = os.path.join(root_path, model_resources)
if not os.path.exists(resources_path):
    os.makedirs(resources_path)


def generate_textual_explanation_for_categorical_columns(input_df):
    cat_text_dict_fraud = {}
    cat_text_dict_non_fraud = {}
    extended_cat_columns = ["incident_hour_of_the_day", "number_of_vehicles_involved", "bodily_injuries", "witnesses"]
    cat_columns.extend(extended_cat_columns)
    for col in cat_columns:
        for i in input_df[col].unique():
            fraud_percentage = len(input_df[(input_df[col] == i) & (input_df[target_column] == 'Y')]) / len(
                (input_df[input_df[col] == i])) * 100
            fraud_percentage = round(fraud_percentage, 2)
            non_fraud_percentage = 100.00 - fraud_percentage
            dict_key = f'{col} ({i})'
            cat_text_dict_fraud[
                dict_key] = f'{fraud_percentage} % fraudulent cases depicted for involvement of {col} ({i})'
            cat_text_dict_non_fraud[
                dict_key] = f'{non_fraud_percentage} % non-fraudulent cases depicted for involvement of {col} ({i})'

    fraud_cat_text_file = open(f"{resources_path}/fraud_cat_text.json", "w", encoding='utf8')
    json.dump(cat_text_dict_fraud, fraud_cat_text_file, indent=4)

    non_fraud_cat_text_file = open(f"{resources_path}/non_fraud_cat_text.json", "w", encoding='utf8')
    json.dump(cat_text_dict_non_fraud, non_fraud_cat_text_file, indent=4)


def generate_textual_explanation_from_int_columns(input_df):
    int_text_dict_fraud = {}
    int_text_dict_non_fraud = {}
    claim_cols = ["injury_claim", "property_claim", "vehicle_claim"]
    for auto in input_df['auto_make'].unique():
        for key in claim_cols:
            dict_key = f"{key} ({auto})"
            fraud_injury_claim_mean = round(
                input_df[(input_df[target_column] == 'Y') & (input_df['auto_make'] == auto)][key].mean(), 2)
            non_fraud_injury_claim_mean = round(
                input_df[(input_df[target_column] == 'N') & (input_df['auto_make'] == auto)][key].mean(), 2)
            int_text_dict_fraud[dict_key] = fraud_injury_claim_mean
            int_text_dict_non_fraud[dict_key] = non_fraud_injury_claim_mean

    json_fraud_object = json.dumps(int_text_dict_fraud, indent=4)
    with open(f"{resources_path}/fraud_int_text.json", "w") as f:
        f.write(json_fraud_object)

    json_non_fraud_object = json.dumps(int_text_dict_non_fraud, indent=4)
    with open(f"{resources_path}/non_fraud_int_text.json", "w") as f:
        f.write(json_non_fraud_object)


if __name__ == '__main__':
    df = pd.read_csv('Dataset/PREPROCESSED/preprocessed.csv')
    generate_textual_explanation_for_categorical_columns(df)
    generate_textual_explanation_from_int_columns(input_df=df)
