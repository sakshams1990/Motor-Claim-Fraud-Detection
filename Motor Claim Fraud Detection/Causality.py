import itertools
import json
import operator as op

import numpy as np

from Shap import sort_shap_value
from get_values_from_config_file import prediction_threshold, number_of_causalities, fraudScore_Number_of_Decimals, \
    model_resources

FRAUD_EXPLANATION = "fraud_explanation"
NON_FRAUD_EXPLANATION = "non_fraud_explanation"

# Initializing labels and causalities category
labels = ["fraud_reported_n", "fraud_reported_y"]


# Normalizing positive values in the dictionary
def normalize_shap_pos_values(dic):
    return {k: v / max(dic.values()) * 100 for k, v in dic.items()}


# Normalizing negative values in the dictionary
def normalize_shap_neg_values(dic):
    return {k: v / min(dic.values()) * (-100) for k, v in dic.items()}


# Converting the shap dictionary into 2 separate dictionaries to contain positive and negative shap values
# Normalizing both the dictionaries and sorting the shap values in descending value
def get_causality_data(shap_dic):
    causality = {}
    shap_dict_pos = {}  # Positive Shap Values
    shap_dict_neg = {}  # Negative Shap Values

    for (key, value) in shap_dic.items():  # For loop to save values in pos & neg dict
        if op.lt(value, 0):  # Using operator package for less than. Imported operator as op package.lt stands for
            # less than
            shap_dict_neg[key] = value
        else:
            shap_dict_pos[key] = value

    # shap_norm_pos = normalize_shap_pos_values(shap_dict_pos)  # Normalizing the values
    # shap_norm_neg = normalize_shap_neg_values(shap_dict_neg)  # Normalizing the values
    causality["fraud_reported_y"] = dict(itertools.islice(sort_shap_value(shap_dict_pos, True).items(),
                                                          number_of_causalities))
    causality["fraud_reported_n"] = dict(itertools.islice(sort_shap_value(shap_dict_neg, False).items(),
                                                          number_of_causalities))

    # Multiplying the causality values with -1
    for key in causality["fraud_reported_y"]:
        causality["fraud_reported_y"][key] = causality["fraud_reported_y"][key] * (-1)
    for key in causality["fraud_reported_n"]:
        causality["fraud_reported_n"][key] = causality["fraud_reported_n"][key] * (-1)
    return causality


def get_text_explanation(causality_dict, json_record):
    fraud_response = {FRAUD_EXPLANATION: [], NON_FRAUD_EXPLANATION: []}

    with open(f"{model_resources}/fraud_cat_text.json", "rb") as f:
        fraud_cat_text = json.load(f)

    with open(f"{model_resources}/non_fraud_cat_text.json", "rb") as nf:
        non_fraud_cat_text = json.load(nf)

    with open(f"{model_resources}/fraud_int_text.json", "rb") as f:
        fraud_int_text = json.load(f)

    with open(f"{model_resources}/non_fraud_int_text.json", "rb") as nf:
        non_fraud_int_text = json.load(nf)

    for key in causality_dict['fraud_reported_y']:
        if key in fraud_cat_text.keys():
            fraud_response[FRAUD_EXPLANATION].append(fraud_cat_text[key])

    for key in causality_dict['fraud_reported_n']:
        if key in non_fraud_cat_text.keys():
            fraud_response[NON_FRAUD_EXPLANATION].append(non_fraud_cat_text[key])

    claim_cols = ["injury_claim", "vehicle_claim", "property_claim"]

    for col in claim_cols:
        for key in causality_dict['fraud_reported_y'].keys():
            if key.startswith(col):
                auto_make = json_record["auto_make"]
                key = key.split("(")[0]
                key = key.strip()
                new_key = key + " (" + auto_make + ")"

                if new_key in fraud_int_text:
                    if json_record[key] > fraud_int_text[new_key]:
                        statement = f"The {key} amount (${json_record[key]}) for auto make [{auto_make}] for this claim is unusual, higher than expected. Average cost for such claim is ${fraud_int_text[new_key]}"
                        fraud_response[FRAUD_EXPLANATION].append(statement)

                    else:
                        statement = f"The {key} amount (${json_record[key]}) for auto make [{auto_make}] for this claim is below the average cost of ${fraud_int_text[new_key]}"
                        fraud_response[FRAUD_EXPLANATION].append(statement)

        for key in causality_dict['fraud_reported_n'].keys():
            if key.startswith(col):
                auto_make = json_record["auto_make"]
                key = key.split("(")[0]
                key = key.strip()
                new_key = key + " (" + auto_make + ")"

                if new_key in non_fraud_int_text:
                    if json_record[key] > non_fraud_int_text[new_key]:
                        statement = f"The {key} amount (${json_record[key]}) for auto make [{auto_make}] for this claim is unusual, higher than expected. Average cost for such claim is ${non_fraud_int_text[new_key]}. Red alert for claim amount."
                        fraud_response[NON_FRAUD_EXPLANATION].append(statement)
                    else:
                        statement = f"The {key} amount (${json_record[key]}) for auto make [{auto_make}] for this claim is below the average cost of ${non_fraud_int_text[new_key]}"
                        fraud_response[NON_FRAUD_EXPLANATION].append(statement)

    return fraud_response


def write_response(model, single_data, shap_dict, json_record):
    single_data_array = np.array(single_data)
    fraudScore = model.predict_proba([single_data_array])[:, 1][0].astype('float64')
    fraudScore = round(fraudScore, fraudScore_Number_of_Decimals)
    causality = get_causality_data(shap_dict)
    fraudStatus = "The claim is FRAUD!!" if fraudScore > prediction_threshold else "The claim is NON FRAUD!!"
    text_explanation = get_text_explanation(causality_dict=causality, json_record=json_record)
    response = {"causality": causality, "fraudScore": fraudScore, "fraudStatus": fraudStatus,
                "explanation": text_explanation}
    return response


if __name__ == '__main__':
    pass
