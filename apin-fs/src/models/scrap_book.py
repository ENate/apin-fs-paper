import argparse


def make_argparse():
    """An argparse function"""
    parsed = argparse.ArgumentParser()
    parsed.add_argument("--time_delta_n", action="store", type=int, required=True)
    arguments = parsed.parse_args()
    return arguments


def function_call_argparse():
    def_values = make_argparse()
    print(def_values.time_delta_n)


function_call_argparse()


def structured_l2_penalty(self, w_matrix, b_matrix):
    """Define the l2 penalty for input feature optimization

    Args:
        w_matrix (intArray): An array of input to first hidden layer weights
        b_matrix (intArray): An array of input to first hidden layer bias.

    Returns:
        structured_l2 : Structured l2 penalty to be incorporated to the input layer.
    """
    combine_wb_matrices = tf.concat([w_matrix, b_matrix], axis=0)
    square_parameters = tf.square(combine_wb_matrices)
    column_wise_addition = tf.reduce_sum(square_parameters, axis=1)
    sqrt_row_wise_params = tf.sqrt(column_wise_addition)
    self.gen_sum_parameters = tf.reduce_sum(sqrt_row_wise_params)
    return self.gen_sum_parameters


def l1_parameter_norm(self, other_parameters):
    """Read model parameters connecting input -> hidden layer 1
    and compute the LASSO norm.

    Args:
        other_parameters ([Tensor]): [Tensor lists of matrix parameters]
    """
    self.sum_lasso = 0.0
    # for k in range(len(other_parameters)):
    for k, _ in enumerate(other_parameters):
        self.sum_lasso = self.sum_lasso + tf.reduce_sum(tf.abs(other_parameters[k]))
    return self.sum_lasso
