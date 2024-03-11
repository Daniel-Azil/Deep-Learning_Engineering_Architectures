# Pytorch Implementation of Machine Learning Algorithms.

## 1-One_Dimension_Linear_Regression_Training.ipynb

### Implementation of 1D Linear Regression with PyTorch

#### Overview

This Jupyter Notebook contains a simple implementation of 1D linear regression using PyTorch. The goal is to showcase the process of training a linear regression model with varying learning rates and observing the training and validation errors.

#### Contents

- **Data Generation:** Artificial data is created with outliers, providing both training and validation datasets.
- **Linear Regression Model:** A PyTorch model is defined for 1D linear regression.
- **Training Loop:** The model is trained with different learning rates using stochastic gradient descent.
- **Results:** Training and validation errors are recorded for each learning rate.




## 2-Linear_Regression_Multiple_Inputs.ipynb

### Linear Regression with Multiple Inputs

#### Overview
Demonstrates the construction of intricate models using PyTorch's built-in capabilities.

#### Contents

Import essential libraries.
- Initialize a random seed for consistency.
- Define a visualization function for 2D plotting.
- Make Some Data:

- Design a dataset class tailored for two-dimensional features.
- Instantiate a dataset object for model training.
- Model & Setup:

- Construct a specialized linear regression module.
- Establish a linear regression model configured for two inputs and one output.
- Configure an optimizer and select a loss function.
- Training via Mini-Batch Gradient Descent:

- Implement a training mechanism using Mini-Batch Gradient Descent.
- Visualize the model's performance pre and post-training.
- Chart the progression of the loss function throughout iterations.



## 3-Linear_Regression_Multiple_Outputs.ipynb

### Linear Regression with Multiple Outputs

#### Overview
Demonstrate the creation of complex models in PyTorch for multiple linear regression with multiple outputs.

#### Contents
- Import Dependencies:

- Import essential libraries for the task.
- Data Preparation:

- Create a dataset class with two-dimensional features and two targets.
- Generate a dataset object using the defined class.
- Model & Setup:

- Define a custom linear regression module for multiple outputs.
- Initialize a linear regression model.
- Instantiate an optimizer with a specified learning rate.
- Define the Mean Squared Error (MSE) loss function.
- Training:

- Use Mini-Batch Gradient Descent for training.
- Store the total loss for each iteration across epochs.
- Visualization:

- Plot the progression of the loss function over iterations.


## 4-Logistic_Regression_Prediction.ipynb
### Logistic Regression with PyTorch
A guide and demonstration on logistic regression using PyTorch. Dive into the intricacies of binary classification and how PyTorch simplifies its implementation.

### Overview
Logistic regression is a foundational technique in the realm of machine learning, predominantly utilized for binary classification problems. In this repository, we explore the basics of logistic regression and its PyTorch implementation, catering to both beginners and seasoned practitioners.

#### Key Features
Sigmoid Function: Understand the foundational sigmoid function and its role in logistic regression.
PyTorch's nn.Sequential: Dive deep into PyTorch's utilities for crafting logistic regression models.
Custom Modules: Craft bespoke logistic regression implementations for specialized tasks.

- Basic Sigmoid Function
A demonstration showcasing the sigmoid function's role in transforming raw scores into probabilities. Understand its significance and visualize its impact.

- Using nn.Sequential
An illustrative example detailing the construction of logistic regression models using PyTorch's nn.Sequential container. Get hands-on experience with code samples and accompanying explanations.

- Custom Modules
For those keen on a more tailored approach or seeking greater flexibility, this section delves into creating custom PyTorch modules for logistic regression. Uncover advanced techniques and best practices.
