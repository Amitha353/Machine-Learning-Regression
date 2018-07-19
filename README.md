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
* The gradient is used for the closed-form solution as well. Complexity of inverse: O(D^3) -> D - #features.
* Gradient of the RSS.
* Requires a step-size.

---
## Assesing Performance
* Variours measure to assess the efficieny of the model fit.

#### Measuring Loss
* It is a measure of how good the fit is performing.
* It is the cost of using estimated parameters w-hat at x when y is true.
* Absolute error - symmetric error - Absolute difference between true and predicted values.
* Squared error - symmetric error - Squared difference between the actual and predicted values.

#### 3 Measures of errors
1. Training Error - Average over the loss measure pf the training dataset. Not a good predictive performance on the model.
2. Generalization / True Error - Measure of how well the error is being predicted for every possible observation available. It can't be computed.
3. Test Error - Examines the traing data fit on the test set. It is a noisy approximation to the generalization error.

#### Error xs. Model complexity
* Training error - decreases with model complexity.
* Generalization error - decreases and then increases with model complexity.
* Test error - noisy generalization of the true error.

### Overfit
* If the training error is decrease below certain amount and the true error increases.
* At this point the magnitude of the coefficients increases.

#### 3 source of prediction error
1. Noise - inherent to the model, cannot be controlled.
2. Bias - Measure of how well the model fits the true prediction / relationship by averaging over all possible training data sets.
3. Variance - Measure of how a fitted function vary from the training data set to training set of all size and observations.


### Bias-Variance tradeoff
* Require low bias and low variance to have good predictive performance.
* Model complexity increases -> bias decreases and variance increases.
* Mean Square Error (MSE) = bias-variance tradeoff = bias^2 + variance.

### Model selection and Assessment
* Fit the model on the training data set.
* Select between different models on the validation set.
* Test the performance on the test data.

---
## Ridge Regression
* As model complexity increases, the models become overfit.
* Symptom of overfitting -> magnitude of coefficients increases.
* It trades of between the bias and the variance.
* Ridge total cost = measure of fit(RSS on training data) + measure of the magnitude of the coefficients.
* It is the L2 regularization parameter = Rss(w) + lambda * ||w||^2

#### Coefficient path
* The magnitude of the coefficients decreases with increases in the tuning parameter "lambda".

#### Ridge closed-form solution  -> complexity O(D^3);

#### Cross-Validation
* In case of insuuffient data to form a separate validation set.
* Then perform k-fold cross validation.
* Here the training set is divided into blocks and each block is treated as the validation set.
  - training block -> parameters or coefficients are extimated.
  - validation block the error is computed.
* The average error across all validation set is computed.  


