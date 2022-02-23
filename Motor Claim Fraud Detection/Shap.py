import shap


def shap_value(model, single_value_dataframe):
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    val = explainer.shap_values(single_value_dataframe)
    return val


def sort_shap_value(dic, reverse):
    return dict(sorted(dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=reverse))


if __name__ == '__main__':
    pass
