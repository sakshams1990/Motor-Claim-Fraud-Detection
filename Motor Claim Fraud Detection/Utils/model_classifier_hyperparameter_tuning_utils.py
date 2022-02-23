from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold


def get_stratified_value(n_split, shuffle, random_state):
    strkf = StratifiedKFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)
    return strkf


def get_best_params_from_grid_search_cv(classifier, params, strkf):
    search_model = GridSearchCV(classifier, param_grid=params, cv=strkf,
                                refit=True, return_train_score=True, verbose=1)
    return search_model


def get_best_params_from_randomized_search_cv(classifier, params, strkf, n_iter, random_state):
    search_model = RandomizedSearchCV(classifier, param_distributions=params, cv=strkf, n_iter=n_iter,
                                      random_state=random_state,
                                      refit=True, return_train_score=True, verbose=1)
    return search_model


if __name__ == '__main__':
    pass
