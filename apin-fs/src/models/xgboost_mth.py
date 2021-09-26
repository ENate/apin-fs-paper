# use feature importance for feature selection, with fix for xgboost 1.0.2
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# define custom class to fix bug in xgboost 1.0.2


class MyXGBClassifier(XGBClassifier):
    @property
    def coef_(self):
        return None


# load data
cancer = load_breast_cancer()
X_DATA = cancer.data
Y_DATA = cancer.target
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    X_DATA, Y_DATA, test_size=0.3, random_state=42, stratify=Y_DATA
)
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_DATA, Y_DATA, test_size=0.33, random_state=7
)
# fit model on all training data
model = MyXGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    predictions = selection_model.predict(select_X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(
        "Thresh=%.3f, n=%d, Accuracy: %.2f%%"
        % (thresh, select_X_train.shape[1], accuracy * 100.0)
    )
