# visualize
import time
import numpy as np
import tensorflow as tf
from deepNetImpl import NeuralNetwork
from drawingNetsformatting import paramreshape
from results_classifier import func_prediction_analysis


def analyze_labels(model, data_sets):
    x = model(data_sets.get('x_train'))
    some_testy = data_sets.get('y_train')[0:10, :]
    y = x.numpy()
    one_hot_tensor = y
    label_train_prob = tf.argmax(one_hot_tensor, axis = 1)
    label2_train = tf.argmax(data_sets.get('y_train'), axis=1)
    y = np.where(y < 0, 0, 1)
    print("10 point classifier output: ")
    print(y[0:10, :])
    print("10 point Test set output: ")
    print(some_testy)
    
    print("Confusion matrix for training set:")
    print(tf.math.confusion_matrix(label2_train, label_train_prob, num_classes=2).numpy())
    
    x_test = model(data_sets.get('x_test'))
    some_testy = data_sets.get('y_train')[0:10, :]
    # 
    y_test = x_test.numpy()
    one_hot_tensor_test = y_test
    label_test_prob = tf.argmax(one_hot_tensor_test, axis = 1)
    label2_test = tf.argmax(data_sets.get('y_test'), axis=1)
    print("The Confusion matrix for the test set: ")
    print(tf.math.confusion_matrix(label2_test, label_test_prob, num_classes=2).numpy())
    #  t2_start = time.perf_counter()
    # t2_stop_cnt = time.perf_counter()
    # print(f"The count is: {t2_stop_cnt - t2_start}")
    func_prediction_analysis(label_test_prob, label2_test)

def visualize_network(all_net_draw_mat):
    network = NeuralNetwork()
    # loop via formatted matrix as layers in network
    for idx_params in all_net_draw_mat:
        network.add_layer(idx_params.shape[1], idx_params)
        print(idx_params.shape)
    # last layer to output
    nh = np.ones((1, all_net_draw_mat[-1].shape[0]))
    print(nh.shape)
    network.add_layer(nh.shape[1], nh)
    network.draw()


if __name__ == "__main__()":
    p_theta, w_b_shapes, w_b_sizes, m_hidden = 1, 1, 1, 2
    # all_net_draw_mat = paramreshape(p_theta, w_b_shapes, w_b_sizes, m_hidden)
