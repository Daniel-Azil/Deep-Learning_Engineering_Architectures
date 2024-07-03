"""
    Linear Regression (Forward)

    Implements linear regression, a statistical model for predicting outcomes based on input data.

    Methods:
    - get_model_prediction(input_data, weights, bias=0.0):
      Predicts output values using linear regression.
      Args:
      - input_data (NDArray[np.float64]): Input dataset with shape (n, 3).
      - weights (NDArray[np.float64]): Model weights with shape (3,).
      - bias (float, optional): Bias term (default is 0.0).
      Returns:
      - NDArray[np.float64]: Predicted values with shape (n,).

    - get_loss(model_prediction, target_data):
      Computes mean squared error (MSE) between model predictions and target values.
      Args:
      - model_prediction (NDArray[np.float64]): Predicted values with shape (n,).
      - target_data (NDArray[np.float64]): Actual target values with shape (n,).
      Returns:
      - float: Mean squared error (MSE).
    """


import numpy as np
from numpy.typing import NDArray

class Solution:
    """
    	A class that retrieves the models prediction and the loss function of the model.
    """
    def get_model_prediction(self, input_data: NDArray[np.float64], weights: NDArray[np.float64], bias: float = 0.0) -> NDArray[np.float64]:
        """
        Predict output values using linear regression.
        """
        prediction = np.matmul(input_data, weights) + bias
        return np.round(prediction, 5)

    def get_loss(self, model_prediction: NDArray[np.float64], target_data: NDArray[np.float64]) -> float:
        """
        Calculate mean squared error (MSE) between model predictions and target values.
        """
        loss = np.mean(np.square(model_prediction - target_data))
        return np.round(loss, 5)
