# visualize
import numpy as np
from deepNetImpl import NeuralNetwork
from drawingNetsformatting import paramreshape


def analyze_labels(y_prob, y_label):
    pass

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
    all_net_draw_mat = paramreshape(p_theta, w_b_shapes, w_b_sizes, m_hidden)
