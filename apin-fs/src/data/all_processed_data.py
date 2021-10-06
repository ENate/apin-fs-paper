import tensorflow as tf
from sequence_rna import func_inputs_and_label  # noqa
from make_dataset import (
    WCDSPreprocessing,
    ArtificialDataset,
    LSTVData,
    HeartData,
)  # noqa

def processed_data(CHOICES):
    N_CLASSES = 3
    # change the choices for different data sets. For choices 1 and 3 (with message given by testData)
    if CHOICES == 1:
        DATA_FILE = "~/apin-fs-paper/apin-fs/data/raw/data.csv"
        (
            TRAINING_INPUTS,
            TESTING_INPUTS,
            TRAINING_OUTPUT,
            TESTING_OUTPUT,
        ) = WCDSPreprocessing(DATA_FILE).wcds_preprocess()
    elif CHOICES == 2:
        LSTV_DATA = (
            "~/theFinalProject/rwth_ssh_cluster/src/data/LSVT_voice_rehabilitation.xlsx"
        )
        (
            TRAINING_INPUTS,
            TESTING_INPUTS,
            TRAINING_OUTPUT,
            TESTING_OUTPUT,
        ) = LSTVData().prepare_lstv_data(LSTV_DATA)
    elif CHOICES == 3:
        DATA_MSG = "testData"
        if DATA_MSG == "solardatanorm":
            M_LAB_FILE = "~/Desktop/NewFolder2112/SparseNet12ab/solardatanorm.mat"
        else:
            M_LAB_FILE = "~/apin-fs-paper/apin-fs/data/raw/testData.mat"
        (
            TRAINING_INPUTS,
            TESTING_INPUTS,
            TRAINING_OUTPUT,
            TESTING_OUTPUT,
        ) = ArtificialDataset().load_solar_data(M_LAB_FILE, DATA_MSG)
    elif CHOICES == 4:
        (
            TRAINING_INPUTS,
            TESTING_INPUTS,
            TRAINING_OUTPUT,
            TESTING_OUTPUT,
        ) = func_inputs_and_label("data.csv", "labels.csv", N_CLASSES)
    elif CHOICES == 5:
        HEART_DATA = "~/forLenovoUbuntu/datfile/heartdisease/Integrated.csv"
        TRAINING_INPUTS, TESTING_INPUTS, TRAINING_OUTPUT, TESTING_OUTPUT = HeartData(
            HEART_DATA
        ).main_cleaning_74()
    else:
        TRAINING_INPUTS, TESTING_INPUTS, TRAINING_OUTPUT, TESTING_OUTPUT = send_data()

    TRAIN_X = tf.cast(TRAINING_INPUTS, tf.float32)
    TRAIN_Y = tf.cast(TRAINING_OUTPUT, tf.float32)
    # ---------------------------------Testing---------------------------
    TEST_X = tf.cast(TESTING_INPUTS, tf.float32)
    TEST_Y = tf.cast(TESTING_OUTPUT, tf.float32)
    print("===================== The shape of label and input data ==================")
    print(TRAIN_X.shape, TRAIN_Y.shape)
    step_size, input_sizes = TRAIN_X.shape[0], TRAIN_X.shape[1]
    batch_size = TRAIN_X.shape[0]
    dict_datasets = {
        "x_train": TRAIN_X,
        "y_train": TRAIN_Y,
        "x_test": TEST_X,
        "y_test": TEST_Y,
    }
    return dict_datasets