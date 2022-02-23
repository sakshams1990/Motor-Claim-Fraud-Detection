from dataframe_creation import create_dataframe_from_source_file, create_target_variable_mapping
from dataframe_preprocessing import preprocess_dataframe
from manual_feature_engineering import manual_feature_engineering_of_data
from model_selection_training import creating_final_dataset_for_training, run_model_training_experiments

if __name__ == '__main__':
    input_df = create_dataframe_from_source_file()
    labels, label_map = create_target_variable_mapping(input_df)
    preprocessed_df = manual_feature_engineering_of_data(input_df)
    final_df = preprocess_dataframe(preprocessed_df)
    X_train, X_test, y_train, y_test = creating_final_dataset_for_training(final_df)
    best_model = run_model_training_experiments(X_train, X_test, y_train, y_test, labels)
