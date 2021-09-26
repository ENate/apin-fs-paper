import tensorflow as tf


class RegularizationParameters:
    """Define regularization functions"""

    def __init__(self):
        self.sum_lasso = 0.0
        self.gen_sum_parameters = None
        self.square_parameters = None
        self.column_wise_addition_b = None

    def structured_penalty(self, w_matrix_tensor, b_matrix_tensor):
        """Computes Regularizers

        Args:
            w_matrix_tensor (Tensor): Matrix tensor connecting input -> hidden layer 1
            b_matrix_tensor (Tensor): Bias tensor connecting input -> hidden layer 1 bias

        Returns:
            [type]: [description]
        """
        self.square_parameters = tf.square(w_matrix_tensor)
        square_parameters_b_matrix = tf.square(b_matrix_tensor)
        column_wise_addition = tf.reduce_sum(self.square_parameters, axis=1)
        self.column_wise_addition_b = tf.reduce_sum(square_parameters_b_matrix)
        sqrt_row_wise_params = tf.sqrt(column_wise_addition)
        sqrt_row_wise_params_b = tf.sqrt(self.column_wise_addition_b)
        self.gen_sum_parameters = tf.reduce_sum(sqrt_row_wise_params)
        self.gen_sum_parameters_b = tf.reduce_sum(sqrt_row_wise_params_b)
        return self.gen_sum_parameters, self.gen_sum_parameters_b

    def l1_parameter_norm(self, other_parameters):
        """Read model parameters connecting input -> hidden layer 1
        and compute the LASSO norm.

        Args:
            other_parameters ([Tensor]): [Tensor lists of matrix parameters]
        """
        for k in range(len(other_parameters)):
            self.sum_lasso = self.sum_lasso + tf.reduce_sum(tf.abs(other_parameters[k]))
        return self.sum_lasso


class CustomCategoricalLoss(tf.keras.losses.Loss):
    """A class to compute loss"""

    def __init__(
        self,
        model_matrix,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="c_mean_sq_error",
    ):
        super(CustomCategoricalLoss, self).__init__(reduction=reduction, name=name)
        self.reg_factor_l2 = 0.0005
        self.reg_factor_l1 = 0.0001
        self.model_matrix = model_matrix

    def call(self, y_true, y_pred):
        """Call to logits loss function

        Args:
            y_true (Tensor): Tensor true label
            y_pred (Tensor): Tensor model for training loss

        Returns:
            Tensor: Tensor loss object
        """
        # Selects the y_pred which corresponds to y_true equal to 1.
        lasso = RegularizationParameters().l1_parameter_norm(self.model_matrix[0::2])
        reg, _ = RegularizationParameters().structured_penalty(
            self.model_matrix[0], self.model_matrix[1]
        )
        logits_y = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        training_loss = logits_y + self.reg_factor_l1 * lasso + self.reg_factor_l2 * reg
        return training_loss

    def residuals(self, y_true, y_pred):
        """Residuals for training and testing labels.

        Args:
            y_true (Tensor): Labels Tensor.
            y_pred (Tensor): A tensor models

        Returns:
            Tensor: Loss tensor for training and testing.
        """
        # Selects the y_pred which corresponds to y_true equal to 1.
        lasso = RegularizationParameters().l1_parameter_norm(self.model_matrix[2:])
        reg, reg_bias = RegularizationParameters().structured_penalty(
            self.model_matrix[0], self.model_matrix[1]
        )
        # prediction = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=1)
        # old_mse = 1.0 - prediction
        logits_y = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        training_loss = logits_y + self.reg_factor_l1 * lasso + self.reg_factor_l2 * reg
        return training_loss


class CustomMSE(tf.keras.losses.Loss):
    def __init__(self, all_model_params, reduction=tf.keras.losses.Reduction.SUM):
        super(CustomMSE, self).__init__(reduction=reduction)
        self.reg_factor_l2 = 0.02
        self.reg_factor_l1 = 0.008
        self.all_model_params = all_model_params

    def call(self, y_true, y_pred):
        """Computes regression loss

        Args:
            y_true (Tensor): A tensor label matrix
            y_pred (Tensor): A tensor model label matrix
        """
        mse0 = y_true - y_pred
        lasso_mse = RegularizationParameters().l1_parameter_norm(
            self.all_model_params[0::2]
        )
        reg_mse, reg_mse_bias = RegularizationParameters().structured_penalty(
            self.all_model_params[0], self.all_model_params[1]
        )
        return mse0 + self.reg_factor_l2 * reg_mse + self.reg_factor_l1 * lasso_mse

    def residuals(self, y_true, y_pred):
        """Residuals for model and true tensors for regression.

        Args:
            y_true (Tensor): Tensor true Labels
            y_pred (Tensor): Tensor model labels
        """
        mse0 = y_true - y_pred
        lasso_mse = RegularizationParameters().l1_parameter_norm(
            self.all_model_params[0::2]
        )
        reg_mse, _ = RegularizationParameters().structured_penalty(
            self.all_model_params[0], self.all_model_params[1]
        )
        return mse0 + self.reg_factor_l2 * reg_mse + self.reg_factor_l1 * lasso_mse


class ReducedOutputMSE(tf.keras.losses.Loss):
    """Provides mean squared error metrics: loss / residuals.

    Consider using this reduced outputs mean squared error loss for regression
    problems with a large number of outputs or at least more then one output.
    This loss function reduces the number of outputs from N to 1, reducing both
    the size of the jacobian matrix and backpropagation complexity.
    Tensorflow, in fact, uses backward differentiation which computational
    complexity is  proportional to the number of outputs.
    """

    def __init__(
        self,
        all_model_params,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="reduced_outputs_mean_squared_error",
    ):
        super(ReducedOutputMSE, self).__init__(reduction=reduction, name=name)
        self.all_model_parameters = all_model_params
        self.reg_factor_l2 = 0.006
        self.reg_factor_l1 = 0.003

    def call(self, y_true, y_pred):
        lasso_mse = RegularizationParameters().l1_parameter_norm(
            self.all_model_parameters[0::2]
        )
        reg_mse, _ = RegularizationParameters().structured_penalty(
            self.all_model_parameters[0], self.all_model_parameters[1]
        )
        sq_diff = tf.math.squared_difference(y_true, y_pred)
        return (
            tf.math.reduce_mean(sq_diff, axis=1)
            + self.reg_factor_l1 * lasso_mse
            + self.reg_factor_l2 * reg_mse
        )

    def residuals(self, y_true, y_pred):
        lasso_mse = RegularizationParameters().l1_parameter_norm(
            self.all_model_parameters[0::2]
        )
        reg_mse, _ = RegularizationParameters().structured_penalty(
            self.all_model_parameters[0], self.all_model_parameters[1]
        )
        sq_diff = tf.math.squared_difference(y_true, y_pred)
        eps = tf.keras.backend.epsilon()
        return (
            tf.math.sqrt(eps + tf.math.reduce_mean(sq_diff, axis=1))
            + self.reg_factor_l1 * lasso_mse
            + self.reg_factor_l2 * reg_mse
        )


class CategoricalMeanSquaredErr(tf.keras.losses.Loss):
    """Provides mean squared error metrics: loss / residuals.

    Use this categorical mean squared error loss for classification problems
    with two or more label classes. The labels are expected to be provided in a
    `one_hot` representation and the output activation to be softmax.
    """

    def __init__(
        self,
        all_model_param,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="categorical_mean_squared_error",
    ):
        super(CategoricalMeanSquaredErr, self).__init__(reduction=reduction, name=name)
        self.all_model_parameters = all_model_param
        self.reg_factor_l2 = 0.005
        self.reg_factor_l1 = 0.001

    def call(self, y_true, y_pred):
        lasso_classif = RegularizationParameters().l1_parameter_norm(
            self.all_model_parameters[0::2]
        )
        reg_classif, _ = RegularizationParameters().structured_penalty(
            self.all_model_parameters[0], self.all_model_parameters[1]
        )
        # Selects the y_pred which corresponds to y_true equal to 1.
        prediction = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=1)
        return (
            tf.math.squared_difference(1.0, prediction)
            + self.reg_factor_l1 * lasso_classif
            + self.reg_factor_l2 * reg_classif
        )

    def residuals(self, y_true, y_pred):
        lasso_classif = RegularizationParameters().l1_parameter_norm(
            self.all_model_parameters[0::2]
        )
        reg_classif, _ = RegularizationParameters().structured_penalty(
            self.all_model_parameters[0], self.all_model_parameters[1]
        )
        # Selects the y_pred which corresponds to y_true equal to 1.
        prediction = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=1)
        return (
            1.0
            - prediction
            + self.reg_factor_l1 * lasso_classif
            + self.reg_factor_l2 * reg_classif
        )


class CategoricalCELoss(tf.keras.losses.CategoricalCrossentropy):
    """Provides cross-entropy metrics: loss / residuals.

    Use this cross-entropy loss for classification problems with two or more
    label classes. The labels are expected to be provided in a `one_hot`
    representation.
    """

    def __init__(self, model_params):
        self.reg_factor_l2 = 0.005
        self.reg_factor_l1 = 0.001
        self.all_model_params = model_params

    def residuals(self, y_true, y_pred):
        lasso_classif = RegularizationParameters().l1_parameter_norm(
            self.all_model_params[0::2]
        )
        reg_classif, _ = RegularizationParameters().structured_penalty(
            self.all_model_params[0], self.all_model_params[1]
        )
        eps = tf.keras.backend.epsilon()
        return (
            tf.math.sqrt(eps + self.fn(y_true, y_pred, **self._fn_kwargs))
            + self.reg_factor_l1 * lasso_classif
            + self.reg_factor_l2 * reg_classif
        )
