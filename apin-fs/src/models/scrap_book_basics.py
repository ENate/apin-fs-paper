import tensorflow as tf


def l2_condition(
    input_hid, hid_group, model, threshold_parameter, col_shape, normed_tf
):
    # check for condition:
    if normed_tf[hid_group] < threshold_parameter:
        # Define the row index to update in model trainable params
        indices = tf.constant([[hid_group]])
        # generate a row of a zero matrix tensor
        updates = tf.zeros([1, col_shape])
        tf.tensor_scatter_nd_update(model.trainable_variables[0], indices, updates)
        bias_normed = tf.norm(model.trainable_variables[0], axis=1)
        col_shape_bias = (tf.shape(model.trainable_variables[2]))[1]
        if bias_normed < threshold_parameter:
            idx_norm = bias_normed.get_shape()[0]
            index = tf.constant([[idx_norm]])
            # what to substitue in the indices
            update = tf.zeros([1, col_shape_bias])
    return index, update


def l1_condition(hidden_layer_params_idx, the_rest_parameters):
    left_sign = tf.math.multiply(
        the_rest_parameters[0][hidden_layer_params_idx],
        the_rest_parameters[1][hidden_layer_params_idx],
    )
    right_sign = tf.math.multiply(
        the_rest_parameters[1][hidden_layer_params_idx],
        the_rest_parameters[2][hidden_layer_params_idx],
    )
    left_sign = left_sign < 0.0
    print(left_sign)
    right_sign = right_sign < 0.0
    print(right_sign)
    print(" Signum indices")
    osc_tensor = tf.where(tf.math.logical_and(left_sign, right_sign))
    # set or fix those parameters to 0!
    # find the dimension of osc_tensor
    tf_shape_r = tf.shape(osc_tensor)[0]
    # print(osc_tensor)
    # # Define your zero update tensor w_matrix
    print("tf_shape to use as update container")
    updates_l1 = tf.zeros(tf_shape_r)
    print(osc_tensor)
    return osc_tensor, updates_l1
