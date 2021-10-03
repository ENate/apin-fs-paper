# check
import os
import sys
import eli5
import time
from eli5 import permutation_importance
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.datasets import load_breast_cancer
from eli5.sklearn import PermutationImportance
from IPython.display import display
from sklearn.inspection import permutation_importance

# for random forest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# cat and xgnoost
# for xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

# catboost
from catboost import Pool, CatBoostRegressor, cv
from sklearn.preprocessing import OneHotEncoder

sys.path.append("../data/")
sys.path.append(".")
__path__ = [os.path.dirname(os.path.abspath(__file__))]
# from make_dataset import WCDSPreprocessing  # noqa
from data.make_dataset import WCDSPreprocessing


def wcds_preprocess(wsdata):
    """
    Prepare the WCDS data set for training
    """
    data = pd.read_csv(wsdata, sep=",")
    data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
    y_vals = data.diagnosis.values
    x_data = data.drop(["diagnosis", "id", "Unnamed: 32"], axis=1).values
    normalized_x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
    new_cat_features = y_vals.reshape(-1, 1)
    ohe = OneHotEncoder(sparse=False)  # Easier to read
    ynew = ohe.fit_transform(new_cat_features)
    return x_data


def calling_wcds():
    """All the details"""
    DATA_FILE = "/home/nath/forLenovoUbuntu/datfile/otherdata/tfExample/datafiles/breast-cancer-wisconsin-data/data.csv"
    y_data = wcds_preprocess(DATA_FILE)
    return y_data


def lasso_fs_example(x_input, y_output, cancer):
    """Lasso training with sklearn

    Args:
        x_input (Array): input data set.
        y_output (Array): Output data set.
        cancer (Array): All data set.
    """
    skf = StratifiedKFold(n_splits=10)
    lasso = LassoCV(cv=skf, random_state=42).fit(x_input, y_output)
    print(
        "Selected Features:", list(cancer.feature_names[np.where(lasso.coef_ != 0)[0]])
    )
    lr = LogisticRegression(
        C=10, class_weight="balanced", max_iter=10000, random_state=42
    )
    preds = cross_val_predict(
        lr, x_input[:, np.where(lasso.coef_ != 0)[0]], y_output, cv=skf
    )
    print(classification_report(y_output, preds))


def random_forest_fs_example(data_train, data_test, y_test, cancer):
    """RF classification and feature selection

    Args:
        data_train (Array): Training input data.
        data_test (Array): Testing data
        y_test (Array): Testing output
        cancer (Array): All data sets
    """
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    rf.fit(data_train.get("x_train"), data_train.get("y_train"))
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    f1 = plt.figure()
    plt.title("Feature importances")
    plt.bar(
        range(data_train.get("x_train").shape[1]),
        importances[indices],
        color="lightsalmon",
        align="center",
    )
    plt.xticks(
        range(data_train.get("x_train").shape[1]),
        cancer.feature_names[indices],
        rotation=90,
    )
    plt.xlim([-1, data_train.get("x_train").shape[1]])
    f1.savefig(
        "/home/nath/tf2-feature-selection-codes/reports/figures/rf_feature_importances.pdf"
    )
    plt.show()
    # selecting some features from the original space
    sfm = SelectFromModel(rf, threshold=0.1)
    sfm.fit(data_train.get("x_train"), data_train.get("y_train"))
    x_important_train = sfm.transform(data_train.get("x_train"))
    x_important_test = sfm.transform(data_test.get("x_test"))
    rf.fit(x_important_train, data_train.get("y_train"))
    y_pred = rf.predict(x_important_test)
    print(classification_report(y_test, y_pred))


def permutation_importance_fs_example(x_data, train_data, test_data, y_data, cancer):
    """Permutation feature importance using RF classifier

    Args:
        x_data (Array): A matrix of float containing all input data sets.
        train_data (Array): A matrix of float containing training data sets.
        test_data (Array): A matrix of float containing test data sets.
        y_data (Array): A matrix of float containing all labels.
        cancer (Array): A mair of input and output matrices containing all input and output data.
    """
    skf = StratifiedKFold(n_splits=10)
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    rf.fit(train_data.get("x_train"), train_data.get("y_train"))
    perm = PermutationImportance(rf, random_state=42).fit(
        test_data.get("x_test"), test_data.get("y_test")
    )
    perm_importance = permutation_importance(
        rf, test_data.get("x_test"), test_data.get("y_test")
    )
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(
        cancer.feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx]
    )
    plt.xlabel("Permutation Importance")
    f2 = plt.figure(1)
    f2.savefig(
        "/home/nath/tf2-feature-selection-codes/reports/figures/rf_permutation.pdf"
    )
    plt.show()
    # eli5.show_weights(perm)
    # display(eli5.show_weights(perm))
    # selected features
    preds = cross_val_predict(
        rf, x_data[:, np.where(perm.feature_importances_ >= 0.008)[0]], y_data, cv=skf
    )
    print(classification_report(y_data, preds))


def perform_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    model = CatBoostRegressor(random_seed=400, loss_function="RMSE", iterations=400)
    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features_indices,
        eval_set=(X_valid, y_valid),
        verbose=False,
    )

    print("RMSE on training data: " + model.score(X_train, y_train).astype(str))
    print("RMSE on test data: " + model.score(X_test, y_test).astype(str))

    return model


class MyXGBClassifier(xgb.XGBClassifier):
    @property
    def coef_(self):
        return None


def boost_classifier(x_train, y_train, x_test, y_test):
    model = MyXGBClassifier()
    model.fit(x_train, y_train)
    # make predictions for test data and evaluate
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # Fit model using each importance as a threshold
    thresholds = np.sort(model.feature_importances_)
    print(thresholds)

    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(x_train)
        # train model
        # selection_model = xgb.XGBClassifier()
        # selection_model.fit(select_X_train, y_train)
        # eval model
        # select_X_test = selection.transform(x_test)
        # predictions = selection_model.predict(select_X_test)
        # accuracy = accuracy_score(y_test, predictions)
        # print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


if __name__ == "__main__":
    cancer = load_breast_cancer()
    X_DATA = cancer.data
    Y_DATA = cancer.target
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        X_DATA, Y_DATA, test_size=0.3, random_state=42, stratify=Y_DATA
    )
    train_dataset = {"x_train": X_TRAIN, "y_train": Y_TRAIN}
    test_dataset = {"x_test": X_TEST, "y_test": Y_TEST}
    t2_start_lasso = time.perf_counter()
    lasso_fs_example(X_DATA, Y_DATA, cancer)
    t2_stop_lasso = time.perf_counter()
    print("Lasso elapsed time: ", t2_stop_lasso - t2_start_lasso)
    # random forest**
    t2_start_rf = time.perf_counter()
    random_forest_fs_example(train_dataset, test_dataset, Y_TEST, cancer)
    t2_stop_rf = time.perf_counter()
    print("Elapsed time: ", t2_stop_rf - t2_start_rf)
    #
    t2_start_perm = time.perf_counter()
    data_file = {"x_data": X_DATA, "y_data": Y_DATA}
    permutation_importance_fs_example(
        X_DATA, train_dataset, test_dataset, Y_DATA, cancer
    )
    t2_stop_perm = time.perf_counter()
    print("Elapsed time: ", t2_stop_perm - t2_start_perm)
    # boost
    # boost_classifier(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)
