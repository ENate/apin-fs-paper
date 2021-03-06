""" The main implementation !!! """
import os
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from hdata_cleaner import send_data
from tensorflow.python.keras.metrics import Reduce
from tensorflow.python.keras.backend import random_normal

sys.path.append("../data/")
sys.path.append(".")
sys.path.append("..../")
sys.path.append("../visualization")
__path__ = [os.path.dirname(os.path.abspath(__file__))]
from sequence_rna import func_inputs_and_label  # noqa
from make_dataset import (
    WCDSPreprocessing,
    ArtificialDataset,
    LSTVData,
    HeartData,
)  # noqa
import levenberg_marquardt as lm_2  # noqa
from custom_loss_regularization import (
    CustomMSE,
    CustomCategoricalLoss,
    ReducedOutputMSE,
    CategoricalMeanSquaredErr,
    CategoricalCELoss,
)  # noqa
from visualize import analyze_labels # noqa
from all_processed_data import processed_data
# METHOD = "c_classifier"
# METHOD = "regressor"
# METHOD = "classify"

METHOD = "classification"


class TrainerAlgorithm(object):
    """Prepare training data and initialize tf training methods."""

    def __init__(self, set_points, batch_size_in, input_size):
        super(TrainerAlgorithm, self).__init__()
        self.set_points = set_points
        self.batch_size_in = batch_size_in
        self.input_size = input_size

    # train model
    def main_training(self, all_datasets):
        """Build tensorflow keras model and train

        Args:
            set_points (dict): Data set points batch sizes for interval and number of input parameters to train.
        """
        x_train = all_datasets.get("x_train")
        y_train = all_datasets.get("y_train")
        x_test = all_datasets.get("x_test")
        y_test = all_datasets.get("y_test")

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(self.set_points)
        train_dataset = train_dataset.batch(self.batch_size_in).cache()
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Now we get a test dataset.
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.shuffle(x_test.shape[0])
        test_dataset = test_dataset.batch(x_test.shape[0]).cache()
        # call both batches
        # train_dataset = train_dataset.shuffle(set_points).batch(64)
        output_dim = y_train.shape[1]

        if METHOD == "classification":
            model0 = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        20,
                        activation="sigmoid",
                        use_bias=True,
                        input_shape=(self.input_size,),
                    ),
                    tf.keras.layers.Dense(10, activation="sigmoid", use_bias=True),
                    tf.keras.layers.Dense(output_dim, use_bias=True),
                ]  # output_dim, activation="softmax", use_bias=True
            )
        elif METHOD == "classify":
            model0 = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        10,
                        activation="sigmoid",
                        use_bias=True,
                        input_shape=(self.input_size,),
                    ),
                    tf.keras.layers.Dense(8, use_bias=True, activation="sigmoid"),
                    tf.keras.layers.Dense(
                        output_dim, use_bias=True, activation="softmax"
                    ),
                ]
            )
        else:
            model0 = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        9,
                        activation="sigmoid",
                        use_bias=True,
                        input_shape=(self.input_size,),
                    ),
                    tf.keras.layers.Dense(5, use_bias=True, activation="sigmoid"),
                    tf.keras.layers.Dense(output_dim, use_bias=True),
                ]
            )
        #
        return train_dataset, model0

    # train model
    def training_models(self, my_datasets):
        """Train models for training

        Args:
            my_datasets (Array): List of training dataset
        """

        # regularized
        train_dataset, model0 = TrainerAlgorithm(
            self.set_points, self.batch_size_in, self.input_size
        ).main_training(my_datasets)
        # Choose loss function
        if METHOD == "classification":

            trainer = lm_2.Trainer(
                model=model0,
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                loss=CustomCategoricalLoss(model0.trainable_variables),
            )

            """
            model0.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss=CustomCategoricalLoss(model0.trainable_variables),
                metrics=["accuracy"])

            
            
            model_wrapper = lm_2.ModelWrapper(
                tf.keras.models.clone_model(model0))

            model_wrapper.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss=CustomCategoricalLoss(model0.trainable_variables),
                solve_method='qr',
                metrics=['accuracy'])
            """
        elif METHOD == "regressor":
            trainer = lm_2.Trainer(
                model=model0,
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                loss=ReducedOutputMSE(model0.trainable_variables),
            )

        elif METHOD == "c_classifier":
            trainer = lm_2.Trainer(
                model=model0,
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss=CategoricalCELoss(model0.trainable_variables),
            )

        else:
            trainer = lm_2.Trainer(
                model=model0,
                optimizer=tf.keras.optimizers.SGD(learning_rate=1),
                loss=CustomMSE(model0.trainable_variables),
            )

        # Call fit function
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        print("\n_________________________________________________________________")
        print("Train using Levenberg-Marquardt 2 option")
        t2_start = time.perf_counter()

        # model0.fit(train_dataset, epochs=200)
        # model_wrapper.fit(train_dataset, epochs=100)
        loss_bin = tf.keras.metrics.CategoricalAccuracy()
        trainer.fit(dataset=train_dataset, epochs=100, metrics=[loss_bin])

        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`
        optimal_model = model0.trainable_variables[0].numpy()
        model_with_zeros = np.where(np.abs(optimal_model) < 0.1, 0.0, optimal_model)
        # print(np.round(model_with_zeros, 4))
        # print(optimal_model.shape)
        mat_input_hidden_1 = np.round(model_with_zeros, 2)
        # mat_input_hidden_1 = np.round(optimal_model, 2)
        print(np.transpose(mat_input_hidden_1))
        # print(np.round(optimal_model, 2))
        kate_sum = 0
        t2_stop = time.perf_counter()
        print("Elapsed time: ", t2_stop - t2_start)
        predicted_output = model0(my_datasets.get("x_test"))
        # tensorboard call backs
        # plotting
        if METHOD == "regression" or METHOD == "regressor":
            f1 = plt.figure()
            nm_train = my_datasets.get("x_test").shape[0]
            colors = np.random.rand(nm_train)
            area = (10 * np.random.rand(nm_train)) ** 2
            plt.scatter(
                predicted_output, my_datasets.get("y_test"), s=area, c=colors, alpha=0.5
            )
            f1.suptitle(
                "Predicting Artificial Data from Model", fontsize=14, fontweight="bold"
            )
            plt.xlabel("Model", fontsize=14, fontweight="bold")
            plt.ylabel("Data", fontsize=14, fontweight="bold")
            plt.show()
        else:
            for num in range(mat_input_hidden_1.shape[0]):
                if np.sum(mat_input_hidden_1[num, :]) != 0.0:
                    kate_sum = kate_sum + 1
            print("======NON-INFORMATIVE INPUTS============")
            print("Number of SELECTED INPUTS for this run: ")
            print(kate_sum)
            print("======NON-INFORMATIVE INPUTS WITHOUT PARAMETERS < 0.001============")
            print("The other method:")
            print(np.sum(~mat_input_hidden_1.any(0)))
                
            print("The model architecture ")
            print(model0.summary())
        return model0

    def dropout_example(self, train_data):
        """To include testing data results"""
        input_train_x = train_data.get("x_train")
        label_train_y = train_data.get("y_train")
        # Build tf Data
        input_output_data = tf.data.Dataset.from_tensor_slices(
            (input_train_x, label_train_y)
        )
        input_output_data = input_output_data.shuffle(input_train_x.shape[0])
        input_output_data = input_output_data.batch(input_train_x.shape[0]).cache()
        # Build model
        model_dropout = tf.keras.Sequential(
            [
                tf.keras.layers.Dropout(0.2, input_shape=(self.input_size,)),
                tf.keras.layers.Dense(10, activation="tanh"),
                tf.keras.layers.Dense(10, activation="sigmoid"),
                tf.keras.layers.Dense(2, activation="softmax"),
            ]
        )
        model_dropout.summary()

        model_dropout.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        model_wrapper = lm_2.ModelWrapper(tf.keras.models.clone_model(model_dropout))

        model_wrapper.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=CustomMSE(model_dropout.trainable_variables),
        )
        # initialize

        # start timing
        t2_start = time.perf_counter()
        model_wrapper.fit(input_output_data, epochs=100)
        t2_stop = time.perf_counter()
        print("Elapsed time: ", t2_stop - t2_start)
        # format before printing
        optimal_model = model_dropout.trainable_variables[0].numpy()
        model_with_zeros_d = np.where(np.abs(optimal_model) > 1e-2, optimal_model, 0.0)
        print(np.round(model_with_zeros_d, 2))


if __name__ == "__main__":
    # change the choices for different data sets. For choices 1 and 3 (with message given by testData)
    CHOICES = 1
    # TrainerAlgorithm(step_size, batch_size,
    #                 input_sizes).dropout_example(dict_datasets)
    # t2_stop_cnt = time.perf_counter()
    # print(F'The count is: {t2_stop_cnt - t2_start}')
    dict_all_datasets = processed_data(CHOICES)
    TRAIN_X = tf.cast(dict_all_datasets.get("x_train"), tf.float32)
    TRAIN_Y = tf.cast(dict_all_datasets.get("y_train"), tf.float32)
    # ---------------------------------Testing---------------------------
    TEST_X = tf.cast(dict_all_datasets.get("x_test"), tf.float32)
    TEST_Y = tf.cast(dict_all_datasets.get("y_test"), tf.float32)
    print("===================== The shape of label and input data ==================")
    print(TRAIN_X.shape, TRAIN_Y.shape)
    step_size, input_sizes = TRAIN_X.shape[0], TRAIN_X.shape[1]
    batch_size = TRAIN_X.shape[0]
    t2_start = time.perf_counter()
    model = TrainerAlgorithm(step_size, batch_size, input_sizes).training_models(dict_all_datasets)
    t2_stop_cnt = time.perf_counter()
    # =========== Classifier analysis ==============================
    analyze_labels(model, dict_all_datasets)