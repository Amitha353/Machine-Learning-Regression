# Machine-Learning-Regression

---
## Simple Regression
* 1 input and fit a line to data. (intercept and the slope coefficients).

#### Cost of the line
* Residual sum of squares (RSS) - Sum of the square of difference between the original value and the predicted value.
* Use RSS to asses different fits to the model.
* Choose the best fit on the training data that minimizes over the "intercept" and "slope".

### Gradient Descent
* Iterative Algorithm that moves in the direction of the negative gradient.
* for convex functions it converges to the optimum.
---
## Multiple Regression
* Allows to fit more complicated relationships between single input and output. Example - polynomial regression, seasonality, etc.
* It also incorporates more inputs and features and using these various inputs to compute the prediction.
* It is the sum of the weighted collection of features h of inputs xi + epsilon (error / noise term).

#### Cost -> RSS for multiple regression
* RSS for the coefficients -> sum of the square of the difference between the output and the predicted value.
* Predicted value = transpose of the feature matrix and coeffcients.

### Gradient Descent
* The gradient is used for the closed-form solution as well.
* Gradient of the RSS.
