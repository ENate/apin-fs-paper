from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import io
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request as url
import re

with open("/home/nath/Desktop/dec2019/LM_code_final/src/hdatacolumns.txt") as file:
    content = file.read().splitlines()
feature_names = [x.strip() for x in content]

name_cols = []
for line in feature_names:  # .split('\n')
    name_cols.append(line.strip().split()[1].strip(": "))
file1 = io.TextIOWrapper(
    url.urlopen(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/cleveland.data"
    ),
    encoding="ISO-8859-1",
)
file2 = io.TextIOWrapper(
    url.urlopen(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/hungarian.data"
    ),
    encoding="ISO-8859-1",
)
file3 = io.TextIOWrapper(
    url.urlopen(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/long-beach-va.data"
    ),
    encoding="ISO-8859-1",
)
outer_list = []
inner_list = []
for file_ in (file1, file2, file3):
    for line in file_:
        quantities = re.split("[^0-9.name-]+", line.strip())
        for x in quantities:
            if x != "name":
                inner_list.append(x)
            else:
                outer_list.append(inner_list)
                inner_list = []

removeable_rows = sorted(
    [i for i, row in enumerate(outer_list) if len(row) != 75], reverse=True
)
for i in removeable_rows:
    del outer_list[i]
df = pd.DataFrame(outer_list, columns=name_cols)

for column in df.columns:
    try:
        df[column] = df[column].astype(int)
    except:
        df[column] = df[column].astype(float)
df = df.applymap(lambda x: np.NaN if x == -9 else x)

df = df[df.columns[:58]]
for feat in [
    "id",
    "ccf",
    "ekgday",
    "ekgmo",
    "ekgyr",
    "proto",
    "dummy",
    "restckm",
    "exerckm",
    "thalsev",
    "thalpul",
    "earlobe",
    "cmo",
    "cday",
    "cyr",
]:
    del df[feat]

for column in df.columns:
    if len(np.unique(df[column].dropna())) < 2:
        del df[column]

dfcolumns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

for feat in df.columns:
    # if the column is mostly empty na values, drop it
    if df[feat].dropna().size < df[feat].size / 2:
        del df[feat]

df = df[(df["prop"] != 22.0)]
df = pd.get_dummies(df, columns=["cp", "restecg"])

scaler = StandardScaler()
# df.iloc[:, 0:-1] = scaler.fit_transform(df.iloc[:, 0:-1].to_numpy())
# df.iloc[:, 0:-1] = df.iloc[:, 0:-
#                           1].apply(lambda x: (x-x.mean()) / x.std(), axis=0)

X = df[df.columns.difference(["num"])]
X = (X - X.min()) / (X.max() - X.min())

y = df["num"]

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X)
X = imputer.transform(X)

nb_classes = 6
data = [[2, 3, 4, 0]]


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


# print(max(y.to_numpy().shape))
# print(y[0:10])
# print(X[0:10, 0:5])


def send_data():
    ynew = indices_to_one_hot(y, 5)
    x_train, x_test, y_train, y_test = train_test_split(X, ynew, test_size=0.3)
    return x_train, x_test, y_train, y_test
