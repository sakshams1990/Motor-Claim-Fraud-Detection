import os.path
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from Utils.dataframe_preprocessing_utils import class_imbalance_SMOTE, class_imbalance_undersampling
from Utils.model_classification_performance_metrics_utils import \
    generate_model_visualisations_and_model_classification_report
from Utils.model_classification_performance_metrics_utils import generate_performance_metrics, plot_feature_importance
from Utils.model_classifier_hyperparameter_tuning_utils import get_best_params_from_randomized_search_cv, \
    get_stratified_value, get_best_params_from_grid_search_cv
from get_values_from_config_file import target_column, class_imbalance_flag, random_state_number, \
    class_imbalance_method, train_test_ratio_split, algorithm_list, \
    random_forest_hyperparameter, n_splits, shuffle_flag, n_iters, hyperparameter_tuning_flag, \
    hyperparameter_tuning_method, xgboost_hyperparameter, decision_tree_hyperparameter, root_path, \
    best_model_folder, model_visualization, model_results, model_resources


def creating_final_dataset_for_training(input_df):
    X_bal = None
    y_bal = None
    X = input_df.drop(columns=[target_column], axis=1)
    y = input_df[[target_column]]
    # if class imbalance is true , handle class imbalance using the configured method
    if class_imbalance_flag:
        if class_imbalance_method == 'smote':
            X_bal, y_bal = class_imbalance_SMOTE(X, y, random_state=random_state_number)
        elif class_imbalance_flag == 'undersampling':
            X_bal, y_bal = class_imbalance_undersampling(X, y, random_state=random_state_number)
        else:
            print("Incorrect imbalance method. Either choose SMOTE or undersampling")
    else:
        X_bal = X
        y_bal = y

    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=train_test_ratio_split, stratify=y_bal,
                                                        random_state=random_state_number)

    return X_train, X_test, y_train, y_test


def get_best_model_accuracy_after_training(algo, classifier_f1_score, classifier_best_model, classifier_accuracy,
                                           best_acc=0.0, best_algo=None, best_model=None):
    if class_imbalance_flag:
        if classifier_f1_score > best_acc:
            best_algo = algo
            best_model = classifier_best_model
            best_acc = classifier_f1_score
        else:
            best_acc = best_acc
    else:
        if classifier_accuracy > best_acc:
            best_algo = algo
            best_model = classifier_best_model
            best_acc = classifier_accuracy
        else:
            best_acc = best_acc
    return best_acc, best_algo, best_model


def run_model_training_experiments(X_train, X_test, y_train, y_test, target_names):
    best_acc = 0.0
    best_algo = None
    best_model = None
    strkf = get_stratified_value(n_split=n_splits, shuffle=shuffle_flag, random_state=random_state_number)

    for algo in algorithm_list:
        if algo == "Random Forest Classifier":
            print(f"\nStarting {algo} training")
            param_grid = random_forest_hyperparameter
            random_forest_classifier = RandomForestClassifier()
            if hyperparameter_tuning_flag and hyperparameter_tuning_method == 'randomizedsearchcv':
                random_forest_search_model = get_best_params_from_randomized_search_cv(
                    classifier=random_forest_classifier,
                    params=param_grid,
                    n_iter=n_iters,
                    strkf=strkf,
                    random_state=random_state_number)

            elif hyperparameter_tuning_flag and hyperparameter_tuning_method == 'gridsearchcv':
                random_forest_search_model = get_best_params_from_grid_search_cv(classifier=random_forest_classifier,
                                                                                 params=param_grid,
                                                                                 strkf=strkf)

            else:
                random_forest_search_model = random_forest_classifier

            random_forest_search_model.fit(X_train, y_train.values.ravel())
            random_forest_best_estimator = random_forest_search_model.best_estimator_
            random_forest_model_data = {'Model Name': algo,
                                        'Random Forest Best Estimator': random_forest_best_estimator,
                                        'Random Forest Best Params': random_forest_search_model.best_params_,
                                        'Random Forest Best Training Score': random_forest_search_model.best_score_}
            for key, val in random_forest_model_data.items():
                print('\n', key, ':', val)
            random_forest_best_model = random_forest_best_estimator.fit(X_train, y_train.values.ravel())
            y_pred = random_forest_best_model.predict(X_test)
            y_pred_proba = random_forest_best_model.predict_proba(X_test)[:, 1]
            random_forest_accuracy, random_forest_f1_score = generate_performance_metrics(classifier=algo,
                                                                                          y_test=y_test, y_pred=y_pred)

            generate_model_visualisations_and_model_classification_report(algo=algo, y_test=y_test, y_pred=y_pred,
                                                                          y_pred_proba=y_pred_proba,
                                                                          target_names=target_names,
                                                                          model_results=model_results,
                                                                          model_visualization=model_visualization,
                                                                          root_path=root_path)

            if class_imbalance_flag:
                if random_forest_f1_score > best_acc:
                    best_algo = algo
                    best_model = random_forest_best_model
                    best_acc = random_forest_f1_score
                else:
                    best_acc = best_acc
            else:
                if random_forest_accuracy > best_acc:
                    best_algo = algo
                    best_model = random_forest_best_model
                    best_acc = random_forest_accuracy
                else:
                    best_acc = best_acc

        elif algo == "XGBoost Classifier":
            print(f"\nStarting {algo} training")
            param_grid = xgboost_hyperparameter
            if len(target_names) == 2:
                xgboost_classifier = XGBClassifier(eval_metric='logloss', objective="binary:logistic",
                                                   use_label_encoder=False)
            elif len(target_names) > 2:
                xgboost_classifier = XGBClassifier(eval_metric='mlogloss', objective="multi:softmax",
                                                   use_label_encoder=False)
            else:
                print("The target feature should have at least 2 classes!!")
                break

            if hyperparameter_tuning_flag and hyperparameter_tuning_method == 'randomizedsearchcv':
                xgboost_classifier_search_model = get_best_params_from_randomized_search_cv(
                    classifier=xgboost_classifier,
                    params=param_grid,
                    n_iter=n_iters,
                    strkf=strkf,
                    random_state=random_state_number)
            elif hyperparameter_tuning_flag and hyperparameter_tuning_method == 'gridsearchcv':
                xgboost_classifier_search_model = get_best_params_from_grid_search_cv(classifier=xgboost_classifier,
                                                                                      params=param_grid,
                                                                                      strkf=strkf)
            else:
                xgboost_classifier_search_model = xgboost_classifier

            xgboost_classifier_search_model.fit(X_train.values, y_train.values.ravel())
            xgboost_best_estimator = xgboost_classifier_search_model.best_estimator_

            xgboost_model_data = {'Model Name': algo,
                                  'XGBoost Best Estimator': xgboost_best_estimator,
                                  'XGBoost Best Params': xgboost_classifier_search_model.best_params_,
                                  'XGBoost Best Training Score': xgboost_classifier_search_model.best_score_}
            for key, val in xgboost_model_data.items():
                print(key, ':', val)
            xgboost_best_model = xgboost_best_estimator.fit(X_train.values, y_train.values.ravel())
            y_pred = xgboost_best_model.predict(X_test.values)
            y_pred_proba = xgboost_best_model.predict_proba(X_test)[:, 1]
            xgboost_accuracy, xgboost_f1_score = generate_performance_metrics(classifier=algo, y_test=y_test,
                                                                              y_pred=y_pred)

            generate_model_visualisations_and_model_classification_report(algo=algo, y_test=y_test, y_pred=y_pred,
                                                                          y_pred_proba=y_pred_proba,
                                                                          target_names=target_names,
                                                                          model_results=model_results,
                                                                          model_visualization=model_visualization,
                                                                          root_path=root_path)

            if class_imbalance_flag:
                if xgboost_f1_score > best_acc:
                    best_algo = algo
                    best_model = xgboost_best_model
                    best_acc = xgboost_f1_score
                else:
                    best_acc = best_acc
            else:
                if xgboost_accuracy > best_acc:
                    best_algo = algo
                    best_model = xgboost_best_model
                    best_acc = xgboost_accuracy
                else:
                    best_acc = best_acc

        elif algo == "Decision Tree Classifier":
            print(f"\nStarting {algo} training")
            param_grid = decision_tree_hyperparameter
            decision_tree_classifier = DecisionTreeClassifier()
            if hyperparameter_tuning_flag and hyperparameter_tuning_method == 'randomizedsearchcv':
                decision_tree_classifier_search_model = get_best_params_from_randomized_search_cv(
                    classifier=decision_tree_classifier,
                    params=param_grid,
                    n_iter=n_iters,
                    strkf=strkf,
                    random_state=random_state_number)
            elif hyperparameter_tuning_flag and hyperparameter_tuning_method == 'gridsearchcv':
                decision_tree_classifier_search_model = get_best_params_from_grid_search_cv(
                    classifier=decision_tree_classifier,
                    params=param_grid,
                    strkf=strkf)
            else:
                decision_tree_classifier_search_model = decision_tree_classifier

            decision_tree_classifier_search_model.fit(X_train, y_train.values.ravel())
            decision_tree_best_estimator = decision_tree_classifier_search_model.best_estimator_
            decision_tree_model_data = {'Model Name': algo,
                                        'Decision Tree Best Estimator': decision_tree_best_estimator,
                                        'Decision Tree Best Params': decision_tree_classifier_search_model.best_params_,
                                        'Decision Tree Best Training Score': decision_tree_classifier_search_model.best_score_}
            for key, val in decision_tree_model_data.items():
                print(key, ':', val)
            decision_tree_best_model = decision_tree_best_estimator.fit(X_train, y_train.values.ravel())
            y_pred = decision_tree_best_model.predict(X_test)
            y_pred_proba = decision_tree_best_model.predict_proba(X_test)[:, 1]
            decision_tree_accuracy, decision_tree_f1_score = generate_performance_metrics(classifier=algo,
                                                                                          y_test=y_test, y_pred=y_pred)

            generate_model_visualisations_and_model_classification_report(algo=algo, y_test=y_test, y_pred=y_pred,
                                                                          y_pred_proba=y_pred_proba,
                                                                          target_names=target_names,
                                                                          model_results=model_results,
                                                                          model_visualization=model_visualization,
                                                                          root_path=root_path)

            if class_imbalance_flag:
                if decision_tree_f1_score > best_acc:
                    best_algo = algo
                    best_model = decision_tree_best_model
                    best_acc = decision_tree_f1_score
                else:
                    best_acc = best_acc
            else:
                if decision_tree_accuracy > best_acc:
                    best_algo = algo
                    best_model = decision_tree_best_model
                    best_acc = decision_tree_accuracy
                else:
                    best_acc = best_acc

        else:
            print("Unknown algorithm identified. Cannot train the model!!")
            break

    print(f"\nBest classifier model is: {best_algo}")
    print(f"\nSaving model of {best_algo} with test accuracy of {best_acc}")
    save_best_classifier_model(best_model)
    plot_feature_importance(best_model.feature_importances_, X_train.columns, classifier=best_algo,
                            model_visualization=model_visualization, root_path=root_path,
                            model_resources=model_resources,
                            model_results=model_results)
    return best_model


def save_best_classifier_model(best_model):
    model_save_folder = os.path.join(root_path, best_model_folder)
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    with open(f'{model_save_folder}/best_model.pkl', 'wb') as pkl:
        pickle.dump(best_model, pkl)


if __name__ == '__main__':
    pass
