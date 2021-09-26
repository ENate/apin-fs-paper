""" Run, run, the way"""
# Show the function flexing!
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def func_rna_seq(file_dir_name1, label_file):
    """Prepare the rna-sequence data from UCI repository for training
    :label_file: labels file
    :file_dir_name: features file
    :returns: formatted input and output classes

    """
    data = pd.read_csv(file_dir_name1)
    all_labels = pd.read_csv(label_file)
    data = data.iloc[:, 1:]
    data = data.values
    labels = all_labels.iloc[:, 1:]
    labels = labels.values.flatten()
    # pre-processing data by encoding string labels to integers
    label2id = {"BRCA": 0, "COAD": 1, "KIRC": 2, "LUAD": 3, "PRAD": 4}
    y_labels_raw = np.array([label2id[label] for label in labels])

    x_std_scale = MinMaxScaler().fit_transform(data)
    return x_std_scale, y_labels_raw


def indices_to_one_hot(nb_classes, y_output):
    """A function assign classes to LSTV

    :nb_classes: number of classes
    :y_output: initial indices representing output classes
    :returns: encoded indices as output classes

    """
    encoded_y = np.eye(nb_classes)[y_output]
    return encoded_y


def func_inputs_and_label(file_data_csv, label_csv, classes):
    """A function to return the input and labels for the rna-seq data

    :file_data_csv: A csv file containing features
    :label_csv: file containing labels as csv
    :classes: number of classes to encode
    :returns: x_features and y_label variables for training

    """
    x_formatted, raw_output = func_rna_seq(file_data_csv, label_csv)
    encoded_labels = indices_to_one_hot(classes, raw_output)
    x_training, x_testing, y_training, y_testing = train_test_split(
        x_formatted, encoded_labels, test_size=0.20, random_state=52
    )
    return x_training, x_testing, y_training, y_testing


if __name__ == "__main__":
    N_CLASSES = 5
    X_TRAINED, X_TESTED, Y_TRAINED, Y_TESTED = func_inputs_and_label(
        "data.csv", "labels.csv", N_CLASSES
    )
    print("The shape of the input data is: ")
    print(X_TRAINED.shape, Y_TRAINED.shape)
    print(X_TESTED.shape, Y_TESTED.shape)
    print(Y_TRAINED[0:5, :])
