# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
from pathlib import Path
from sklearn import preprocessing
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.backend import variable
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder


sys.path.append(os.path.dirname(os.path.realpath(__file__)))


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


def data_one_preprocessing(file_dat):
    """Pre-process one data set prior to training

    Args:
        file_dat (file): A csv file containing the data set.
    """
    return file_dat


class WCDSPreprocessing:
    def __init__(self, wsdata):
        self.wsdata = wsdata

    def wcds_preprocess(self):
        """
        Prepare the WCDS data set for training
        """
        data = pd.read_csv(self.wsdata, sep=",")
        data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
        y_vals = data.diagnosis.values
        x_data = data.drop(["diagnosis", "id", "Unnamed: 32"], axis=1).values
        normalized_x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
        new_cat_features = y_vals.reshape(-1, 1)
        ohe = OneHotEncoder(sparse=False)  # Easier to read
        ynew = ohe.fit_transform(new_cat_features)
        x_train, x_test, y_train, y_test = train_test_split(
            normalized_x, ynew, test_size=0.30, random_state=42
        )
        return x_train, x_test, y_train, y_test

    # Pre-process the data sets and send to the main function
    def process_dataset_func(self, enter_filename):
        """The goal of this function is to import data, clean and pre-process
        :param enter_filename: file name containing data sets
        :return: training and testing sets.
        """
        df_dataset = pd.read_csv(
            enter_filename, sep=",", dtype={"diagnosis": "category"}
        )
        # #######################################################################
        dummies = pd.get_dummies(
            df_dataset["diagnosis"], prefix="diagnosis", drop_first=False
        )
        dataxynew = pd.concat([df_dataset, dummies], axis=1)
        dataxynew1 = dataxynew.drop(["Unnamed: 32", "id", "diagnosis"], axis=1)
        output_labels = dataxynew1[["diagnosis_B", "diagnosis_M"]]
        input_features = dataxynew1.drop(["diagnosis_B", "diagnosis_M"], axis=1)
        self.x_shape_0 = input_features.shape[0]
        return x_shape_0, input_features, output_labels


class LSTVData:
    def __init__(self):
        self.data_file = None

    def prepare_lstv_data(self, lstv_dat):
        # Generate the random samples.
        self.lstv_dat = lstv_dat
        lstv_read = pd.read_excel(self.lstv_dat)
        scaled = MinMaxScaler()
        normalized_values = scaled.fit_transform(lstv_read)
        lstv_read.loc[:, :] = normalized_values
        sample_points = lstv_read.shape[0]
        sample_pts = sample_points // 3 + 1
        seq_value = [3 * num_index - 3 for num_index in range(1, sample_pts)]
        index_outputs = [2 for k in range(0, sample_points)]
        for k in seq_value:
            index_outputs[k] = 1
        n_classes = 3
        all_features = lstv_read.to_numpy()
        y_labels = LSTVData().indices_one_to_hot(n_classes, index_outputs)
        print(all_features.shape)
        print(y_labels.shape)
        x_train_d, y_train_d = all_features[0:100, :], y_labels[0:100, :]
        x_test_d, y_test_d = all_features[100:, :], y_labels[100:, :]
        return x_train_d, x_test_d, y_train_d, y_test_d

    def indices_one_to_hot(self, nb_classes, y_output):
        """A function assigns classes to LSTV values during preprocessing.

        :param nb_classes: number of classes
        :param y_output: initial indices representing output classes
        :returns: encoded indices as output classes.
        """
        encoded_y = np.eye(nb_classes)[y_output]
        return encoded_y


class ArtificialDataset:
    def solar_dataset(self, file_solar, data_msg):
        """Solar data from the matlab file

        Args:
            :param file_solar: A MATLAB file containing Solar dataset
            :param data_msg: A description of the file containing testData and Solar data choices.

        Returns:
            [array]: list of input, output training and testing data sets.
        """

    def load_solar_data(self, mdatfile, datmsg):
        """This is the mat lab data function"""
        mat_contents = sio.loadmat(mdatfile, struct_as_record=False)
        oct_struct = mat_contents[datmsg]
        if datmsg == "testData":
            valdata = oct_struct[0, 0].xyvalues
            # valdata = (valdata - valdata.min(0)) / valdata.ptp(0)
            x_data = valdata[:, 0:-1]
            y_data = valdata[:, -1]

        else:
            # datmsg == 'solardatanorm':
            valdata = oct_struct[0, 0].values
            # valdata = (valdata - valdata.min(0)) / valdata.ptp(0)
            x_data = valdata[:, 0:-1]
            y_data = valdata[:, -1]
        y_data = np.expand_dims(y_data, axis=1)
        # normalized_x = (x_data - np.min(x_data)) / \
        (np.max(x_data) - np.min(x_data))
        # normalized_y = (y_data - np.min(y_data)) / \
        (np.max(y_data) - np.min(y_data))
        x_train, x_test, y_train_set, y_test_set = train_test_split(
            x_data, y_data, test_size=0.20, shuffle=False
        )
        return x_train, x_test, y_train_set, y_test_set


class ParkisonsData(object):
    def __init__(self, park_filename, *args):
        super(ParkisonsData, self).__init__(*args)
        self.park_filename = park_filename

    def function_park_data(self):
        df_park_data = pd.read_csv(self.arg_filename)
        y_label = df_park_data["status"]
        x_features = df_park_data.drop(["status", "name"])
        for each_x in x_features:
            x_features[each_x] = (x_features[each_x] - x_features[each_x].min()) / (
                x_features[each_x].max() - x_features[each_x].min()
            )
        y_data = y_label.to_numpy()
        y_data = y_data[:, None]
        x_data = x_features.to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.20, shuffle=False
        )
        return x_train, x_test, y_train, y_test


class HeartData(object):
    def __init__(self, heart_data):
        super(HeartData, self).__init__()
        self.heart_data = heart_data

    # function
    def main_cleaning_74(self):
        self.h_data = pd.read_csv(self.heart_data, sep=",", header=None)
        y = self.h_data[57]
        datafile = self.h_data.drop(
            self.h_data.columns[[0, 6, 7, 43, 44, 52, 57, 75]], axis=1
        )
        print(datafile.shape)
        datafile = datafile.drop(datafile.columns[[0, 39, 44]], axis=1).values
        min_max_scaling = preprocessing.MinMaxScaler()
        training_x = min_max_scaling.fit_transform(datafile)
        xfeats = pd.DataFrame(training_x)
        y = y.to_numpy()
        onehotencoder = OneHotEncoder(categories="auto")
        y2 = onehotencoder.fit_transform(y.reshape(-1, 1))
        encoder = LabelEncoder()
        ohe = OneHotEncoder(sparse=False)  # Easier to read
        ynew = ohe.fit_transform(y.reshape(-1, 1))
        y[y < 0.5] = 0
        y[y >= 0.5] = 1
        print(training_x.shape)
        x_train, x_test, y_train, y_test = train_test_split(
            training_x, ynew, test_size=0.3
        )
        return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
