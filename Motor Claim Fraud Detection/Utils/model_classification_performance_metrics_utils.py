import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, classification_report
from sklearn.metrics import roc_curve, f1_score, confusion_matrix, recall_score


def create_model_classifier_visualisation_folder(classifier, root_path, model_visualization):
    model_visualisation_folder = os.path.join(root_path, model_visualization)
    if not os.path.exists(model_visualisation_folder):
        os.makedirs(model_visualisation_folder)

    classifier_model_visualisation_folder = os.path.join(model_visualisation_folder, classifier)
    if not os.path.exists(classifier_model_visualisation_folder):
        os.makedirs(classifier_model_visualisation_folder)
    return classifier_model_visualisation_folder


def create_model_results_folder(classifier, root_path, model_results):
    model_results_folder = os.path.join(root_path, model_results)
    if not os.path.exists(model_results_folder):
        os.makedirs(model_results_folder)

    classifier_model_results_folder = os.path.join(model_results_folder, classifier)
    if not os.path.exists(classifier_model_results_folder):
        os.makedirs(classifier_model_results_folder)
    return classifier_model_results_folder


def generate_roc_curve(classifier, y_test, y_pred, y_pred_proba, root_path, model_visualization):
    model_roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{classifier} (area = %0.2f)' % model_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    classifier_visualization_model_folder = create_model_classifier_visualisation_folder(classifier, root_path,
                                                                                         model_visualization)
    plt.savefig(f'{classifier_visualization_model_folder}/roc_auc_curve.png', bbox_inches="tight")


def generate_confusion_matrix(classifier, y_test, y_pred, class_names, root_path, model_visualization):
    model_confusion_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(model_confusion_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    classifier_visualization_model_folder = create_model_classifier_visualisation_folder(classifier, root_path,
                                                                                         model_visualization)
    plt.savefig(f'{classifier_visualization_model_folder}/confusion_matrix.png', bbox_inches="tight")


def generate_performance_metrics(classifier, y_test, y_pred):
    print(f'AUC Score for {classifier} is:', roc_auc_score(y_test, y_pred))
    print(f'F1 Score for {classifier} is:', f1_score(y_test, y_pred, average='weighted'))
    print(f'Recall score for {classifier} is:', recall_score(y_test, y_pred, average='weighted'))
    print(f'Precision score for {classifier} is:', precision_score(y_test, y_pred, average='weighted'))
    print(f'Accuracy score for {classifier} is:', accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')


def generate_classification_report(classifier, y_true, y_pred, class_names, root_path, model_results):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    classifier_results_model_folder = create_model_results_folder(classifier, root_path, model_results)
    df.to_csv(f"{classifier_results_model_folder}/classification_report.csv", index=class_names)


def plot_feature_importance(importance, names, classifier, root_path, model_visualization, model_results,
                            model_resources):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    top_25_df = fi_df.iloc[:25]
    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Seaborn bar chart
    sns.barplot(x=top_25_df['feature_importance'], y=top_25_df['feature_names'])
    # Add chart labels
    plt.title(classifier + ' Feature Importance')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    classifier_visualization_model_folder = create_model_classifier_visualisation_folder(classifier, root_path,
                                                                                         model_visualization)
    model_resources_path = os.path.join(root_path, model_resources)
    if not os.path.exists(model_resources_path):
        os.makedirs(model_resources_path)
    plt.savefig(f"{classifier_visualization_model_folder}/feature_importance.png", bbox_inches="tight")
    fi_df.to_csv(f"{model_resources_path}/feature_importance.csv", index=False)


def generate_model_visualisations_and_model_classification_report(algo, y_test, y_pred_proba, y_pred, target_names,
                                                                  model_results, root_path, model_visualization):
    generate_roc_curve(classifier=algo, y_test=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba, root_path=root_path,
                       model_visualization=model_visualization)
    generate_confusion_matrix(classifier=algo, y_test=y_test, y_pred=y_pred, class_names=target_names,
                              root_path=root_path, model_visualization=model_visualization)
    generate_classification_report(classifier=algo, y_true=y_test, y_pred=y_pred, class_names=target_names,
                                   model_results=model_results, root_path=root_path)


if __name__ == '__main__':
    pass
