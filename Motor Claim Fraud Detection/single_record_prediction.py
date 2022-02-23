import json
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd

import Causality
from Shap import shap_value
from get_values_from_config_file import ordinal_insured_education_level_hierarchy, ordinal_incident_severity_hierarchy, \
    best_model_folder, root_path, model_resources, cat_columns

warnings.filterwarnings(action="ignore")

resources_path = os.path.join(root_path, model_resources)
if not os.path.exists(resources_path):
    os.makedirs(resources_path)


def create_binary_list_for_column(value, input_list):
    binary_list = []
    for ele in input_list:
        if value == ele:
            binary_list.append(1)
        else:
            binary_list.append(0)
    return binary_list


def create_dict_from_key_list_binary_list(key, feature_list, feature_binary_list):
    feature_list = [key + '_' + sub for sub in feature_list]
    feature_dict = dict(zip(feature_list, feature_binary_list))
    return feature_dict


def preprocess_input_json(json_data):
    single_data_record_list = []
    single_dataframe_dict = {}
    for key, val in json_data.items():
        if key == 'policy_annual_premium':
            single_data_record_list.append(json_data[key])
            single_dataframe_dict[key] = val
        if key == 'umbrella_limit':
            single_data_record_list.append(json_data[key])
            single_dataframe_dict[key] = val
        if key == 'insured_education_level':
            if val is None:
                break
            elif val in ordinal_insured_education_level_hierarchy:
                pos_ind = ordinal_insured_education_level_hierarchy.index(val)
                single_data_record_list.append(pos_ind)
                single_dataframe_dict[key] = pos_ind
            else:
                continue

        if key == 'capital-gains':
            single_data_record_list.append(json_data[key])
            single_dataframe_dict[key] = val
        if key == 'capital-loss':
            single_data_record_list.append(json_data[key])
            single_dataframe_dict[key] = val
        if key == 'incident_severity':
            if val in ordinal_incident_severity_hierarchy:
                pos_ind = ordinal_incident_severity_hierarchy.index(val)
                single_data_record_list.append(pos_ind)
                single_dataframe_dict[key] = pos_ind
        if key == "incident_hour_of_the_day":
            single_data_record_list.append(json_data[key])
            single_dataframe_dict[key] = val
        if key == 'number_of_vehicles_involved':
            single_data_record_list.append(json_data[key])
            single_dataframe_dict[key] = val
        if key == 'bodily_injuries':
            single_data_record_list.append(json_data[key])
            single_dataframe_dict[key] = val
        if key == 'witnesses':
            single_data_record_list.append(json_data[key])
            single_dataframe_dict[key] = val
        if key == 'injury_claim':
            single_data_record_list.append(json_data[key])
            single_dataframe_dict[key] = val
        if key == 'property_claim':
            single_data_record_list.append(json_data[key])
            single_dataframe_dict[key] = val
        if key == 'vehicle_claim':
            single_data_record_list.append(json_data[key])
            single_dataframe_dict[key] = val
        auto_make_list = ['Audi', 'BMW', 'Chevrolet', 'Dodge', 'Ford', 'Honda', 'Jeep', 'Mercedes', 'Nissan', 'Saab',
                          'Suburu', 'Toyota', 'Volkswagen']
        auto_make_binary_list = []
        if key == 'auto_make':
            auto_make_binary_list = create_binary_list_for_column(val, auto_make_list)
            auto_make_dict = create_dict_from_key_list_binary_list(key, auto_make_list, auto_make_binary_list)
            single_dataframe_dict.update(auto_make_dict)
        single_data_record_list.extend(auto_make_binary_list)

        policy_state_list = ['IN', 'OH']
        policy_state_binary_list = []
        if key == 'policy_state':
            policy_state_binary_list = create_binary_list_for_column(val, policy_state_list)
            policy_state_dict = create_dict_from_key_list_binary_list(key, policy_state_list, policy_state_binary_list)
            single_dataframe_dict.update(policy_state_dict)
        single_data_record_list.extend(policy_state_binary_list)

        policy_csl_list = ['250/500', '500/1000']
        policy_csl_binary_list = []
        if key == 'policy_csl':
            policy_csl_binary_list = create_binary_list_for_column(val, policy_csl_list)
            policy_csl_dict = create_dict_from_key_list_binary_list(key, policy_csl_list, policy_csl_binary_list)
            single_dataframe_dict.update(policy_csl_dict)
        single_data_record_list.extend(policy_csl_binary_list)

        insured_occupation_list = ["armed-forces", "craft-repair", "exec-managerial", "farming-fishing",
                                   "handlers-cleaners", "machine-op-inspct", "other-service", "priv-house-serv",
                                   "prof-specialty",
                                   "protective-serv", "sales", "tech-support", "transport-moving"]
        insured_occupation_binary_list = []
        if key == 'insured_occupation':
            insured_occupation_binary_list = create_binary_list_for_column(val, insured_occupation_list)
            insured_occupation_dict = create_dict_from_key_list_binary_list(key, insured_occupation_list,
                                                                            insured_occupation_binary_list)
            single_dataframe_dict.update(insured_occupation_dict)
        single_data_record_list.extend(insured_occupation_binary_list)

        insured_hobbies_list = ["basketball", "board-games", "bungie-jumping", "camping", "chess", "cross-fit",
                                "dancing", "exercise", "golf", "hiking", "kayaking", "movies",
                                "paintball", "polo", "reading", "skydiving", "sleeping", "video-games", "yachting"]
        insured_hobbies_binary_list = []
        if key == 'insured_hobbies':
            insured_hobbies_binary_list = create_binary_list_for_column(val, insured_hobbies_list)
            insured_hobbies_dict = create_dict_from_key_list_binary_list(key, insured_hobbies_list,
                                                                         insured_hobbies_binary_list)
            single_dataframe_dict.update(insured_hobbies_dict)
        single_data_record_list.extend(insured_hobbies_binary_list)

        insured_relationship_list = ["not-in-family", "other-relative", "own-child", "unmarried", "wife"]
        insured_relationship_binary_list = []
        if key == "insured_relationship":
            insured_relationship_binary_list = create_binary_list_for_column(val, insured_relationship_list)
            insured_relationship_dict = create_dict_from_key_list_binary_list(key, insured_relationship_list,
                                                                              insured_relationship_binary_list)
            single_dataframe_dict.update(insured_relationship_dict)
        single_data_record_list.extend(insured_relationship_binary_list)

        incident_type_list = ["Parked Car", "Single Vehicle Collision", "Vehicle Theft"]
        incident_type_binary_list = []
        if key == "incident_type":
            incident_type_binary_list = create_binary_list_for_column(val, incident_type_list)
            incident_type_dict = create_dict_from_key_list_binary_list(key, incident_type_list,
                                                                       incident_type_binary_list)
            single_dataframe_dict.update(incident_type_dict)
        single_data_record_list.extend(incident_type_binary_list)

        collision_type_list = ["Rear Collision", "Side Collision", "UNKNOWN"]
        collision_type_binary_list = []
        if key == "collision_type":
            collision_type_binary_list = create_binary_list_for_column(val, collision_type_list)
            collision_type_dict = create_dict_from_key_list_binary_list(key, collision_type_list,
                                                                        collision_type_binary_list)
            single_dataframe_dict.update(collision_type_dict)
        single_data_record_list.extend(collision_type_binary_list)

        authorities_contacted_list = ["Fire", "None", "Other", "Police"]
        authorities_contacted_binary_list = []
        if key == "authorities_contacted":
            authorities_contacted_binary_list = create_binary_list_for_column(val, authorities_contacted_list)
            authorities_contacted_dict = create_dict_from_key_list_binary_list(key, authorities_contacted_list,
                                                                               authorities_contacted_binary_list)
            single_dataframe_dict.update(authorities_contacted_dict)
        single_data_record_list.extend(authorities_contacted_binary_list)

        incident_state_list = ["NY", "OH", "PA", "SC", "VA", "WV"]
        incident_state_binary_list = []
        if key == "incident_state":
            incident_state_binary_list = create_binary_list_for_column(val, incident_state_list)
            incident_state_dict = create_dict_from_key_list_binary_list(key, incident_state_list,
                                                                        incident_state_binary_list)
            single_dataframe_dict.update(incident_state_dict)
        single_data_record_list.extend(incident_state_binary_list)

        incident_city_list = ["Columbus", "Hillsdale", "Northbend", "Northbrook", "Riverwood", "Springfield"]
        incident_city_binary_list = []
        if key == "incident_city":
            incident_city_binary_list = create_binary_list_for_column(val, incident_city_list)
            incident_city_dict = create_dict_from_key_list_binary_list(key, incident_city_list,
                                                                       incident_city_binary_list)
            single_dataframe_dict.update(incident_city_dict)
        single_data_record_list.extend(incident_city_binary_list)

        property_damage_list = ["UNKNOWN", "YES"]
        property_damage_binary_list = []
        if key == "property_damage":
            property_damage_binary_list = create_binary_list_for_column(val, property_damage_list)
            property_damage_dict = create_dict_from_key_list_binary_list(key, property_damage_list,
                                                                         property_damage_binary_list)
            single_dataframe_dict.update(property_damage_dict)
        single_data_record_list.extend(property_damage_binary_list)

        police_report_list = ["UNKNOWN", "YES"]
        police_report_binary_list = []
        if key == "police_report_available":
            police_report_binary_list = create_binary_list_for_column(val, police_report_list)
            police_report_dict = create_dict_from_key_list_binary_list(key, police_report_list,
                                                                       police_report_binary_list)
            single_dataframe_dict.update(police_report_dict)
        single_data_record_list.extend(police_report_binary_list)

        auto_year_list = ["1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006",
                          "2007", "2008", "2009",
                          "2010", "2011", "2012", "2013", "2014", "2015"]
        auto_year_binary_list = []
        if key == "auto_year":
            auto_year_binary_list = create_binary_list_for_column(val, auto_year_list)
            auto_year_dict = create_dict_from_key_list_binary_list(key, auto_year_list, auto_year_binary_list)
            single_dataframe_dict.update(auto_year_dict)
        single_data_record_list.extend(auto_year_binary_list)

        months_as_customer_labels_list = ["new customer", "regular customer", "old customer", "very old customer"]
        months_as_customer_labels_binary_list = []
        if key == "months_as_customer_labels":
            months_as_customer_labels_binary_list = create_binary_list_for_column(val, months_as_customer_labels_list)
            months_as_customer_dict = create_dict_from_key_list_binary_list(key, months_as_customer_labels_list,
                                                                            months_as_customer_labels_binary_list)
            single_dataframe_dict.update(months_as_customer_dict)
        single_data_record_list.extend(months_as_customer_labels_binary_list)

        age_labels_list = ["young", "old", "very old"]
        age_labels_binary_list = []
        if key == "age_labels":
            age_labels_binary_list = create_binary_list_for_column(val, age_labels_list)
            age_labels_dict = create_dict_from_key_list_binary_list(key, age_labels_list, age_labels_binary_list)
            single_dataframe_dict.update(age_labels_dict)
        single_data_record_list.extend(age_labels_binary_list)

        policy_deductable_list = ["1000", "2000"]
        policy_deductable_binary_list = []
        if key == "policy_deductable":
            policy_deductable_binary_list = create_binary_list_for_column(val, policy_deductable_list)
            policy_deductable_dict = create_dict_from_key_list_binary_list(key, policy_deductable_list,
                                                                           policy_deductable_binary_list)
            single_dataframe_dict.update(policy_deductable_dict)
        single_data_record_list.extend(policy_deductable_binary_list)
    return single_data_record_list, single_dataframe_dict


# Combine the keys and values of the input data and consider the combined as keys
def combine_input_keys_values(key, value):
    dic = {}
    combined_keys = [str(i) + ' (' + str(j) + ')' for i, j in zip(key, value)]
    for k in combined_keys:
        for val in value:
            dic[k] = val
            value.remove(val)
            break
    return dic


def reformat_json_dictionary(input_json):
    new_dict = combine_input_keys_values(list(input_json.keys()), list(input_json.values()))
    return new_dict


def get_new_key_list_for_claim_amount(json_file):
    new_key_list = []
    for key in json_file.keys():
        if key in ["injury_claim", "property_claim", "vehicle_claim"]:
            new_key = key + " (" + json_file["auto_make"] + ")"
            new_key_list.append(new_key)
    return new_key_list


# Converting the shap values into a list format
def shap_value_to_list(arr):
    arr = np.array(arr).tolist()[0]
    return arr


# Combine claim features with the auto make value
def preprocess_claim_features_shap_dict(shap_dict):
    auto_make = None
    for key in shap_dict.keys():
        if key.startswith("auto_make"):
            auto_make = re.split(r"[( )]", key)
            auto_make = auto_make[2]
    claim_cols = ["injury_claim", "property_claim", "vehicle_claim"]
    for cols in claim_cols:
        for key, val in list(shap_dict.items()):
            if key.startswith(cols):
                new_key = key.split('(')[0]
                new_key = new_key.strip()
                new_key = new_key + ' (' + auto_make + ')'
                shap_dict[new_key] = shap_dict.pop(key)

    return shap_dict


# Preprocessing shap_dict to remove one hot encoded features that corresponds to 0
def preprocess_shap_dict(dic):
    for key, val in list(dic.items()):
        for col in cat_columns:
            if key.startswith(col):
                if key.__contains__('(0)'):
                    del dic[key]
                elif key.__contains__('(1)'):
                    new_key = key.split('(')[0]
                    new_key = new_key.strip()
                    dic[new_key] = dic.pop(key)
                    if new_key.count("_") > 1:
                        last_underscore = new_key.rfind("_")
                        new_string = new_key[:last_underscore] + " (" + new_key[last_underscore + 1:] + ")"
                        dic[new_string] = dic.pop(new_key)
    return dic


def predict_single_json_record(json_record):
    single_record_list, single_dataframe_record_dict = preprocess_input_json(json_record)
    single_record_dataframe = pd.DataFrame([single_dataframe_record_dict])
    ordinal_insured_education_level_dict = {k: v for k, v in enumerate(ordinal_insured_education_level_hierarchy)}
    ordinal_incident_severity_dict = {k: v for k, v in enumerate(ordinal_incident_severity_hierarchy)}

    for key in ordinal_insured_education_level_dict.keys():
        if key == single_dataframe_record_dict['insured_education_level']:
            single_dataframe_record_dict['insured_education_level'] = ordinal_insured_education_level_dict[key]

    for key in ordinal_incident_severity_dict.keys():
        if key == single_dataframe_record_dict['incident_severity']:
            single_dataframe_record_dict['incident_severity'] = ordinal_incident_severity_hierarchy[key]

    best_model_path = f"{best_model_folder}/best_model.pkl"
    with open(best_model_path, "rb") as pf:
        model_file = pickle.load(pf)
    shap_values = shap_value_to_list(shap_value(model_file, single_record_dataframe))
    single_record_dict = combine_input_keys_values(list(single_dataframe_record_dict.keys()),
                                                   list(single_dataframe_record_dict.values()))
    shap_dict = dict(zip(single_record_dict.keys(), shap_values))
    final_shap_dict = preprocess_shap_dict(shap_dict)
    final_response = Causality.write_response(model_file, single_record_list, final_shap_dict, json_record=json_record)
    return final_response


if __name__ == '__main__':
    with open('single_file_prediction_fraud.json', "rb") as f:
        json_file = json.load(f)
    response = predict_single_json_record(json_file)
    print(response)
