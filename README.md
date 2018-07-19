# Machine-Learning-Regression

## MODELS
1. Linear Regression - Simple and Multiple
2. Regularization - Ridge (L2), Lasso (L1)
3. Nearest neighbors and kernel regression

## ALGORITHMS
1. Gradient Descent
2. Coordinate Descent

## GENERAL CONCEPTS
1. Loss function
2. Bias-Variance Trade off
3. Cross Validation
4. Sparsity
5. Overfitting
6. Model selection
7. Feature selection

---

|Information | Modules |
|---|---|
| Simple Regression | [Module 1](https://github.com/Amitha353/Machine-Learning-Regression/blob/master/Week%201%20-%20Simple%20Linear%20Regression.pdf) |
| Multiple Regression | [Module 2](https://github.com/Amitha353/Machine-Learning-Regression/blob/master/Week%202%20-%20Multiple%20Regression.pdf) |
| Assesing Performance | [Module 3](https://github.com/Amitha353/Machine-Learning-Regression/blob/master/Week%203%20-%20Assessing%20Performance.pdf) |
| Ridge Regression | [Module 4](https://github.com/Amitha353/Machine-Learning-Regression/blob/master/Week%204%20-%20Ridge%20Regression.pdf) |
| Feature Selection & Lasso | [Module 5](https://github.com/Amitha353/Machine-Learning-Regression/blob/master/Week%205%20-%20Feature%20Selection%20%26%20Lasso.pdf) |
| Nearest Neighbor & Kernel Regression | [Module 6](https://github.com/Amitha353/Machine-Learning-Regression/blob/master/Week%206%20-%20Nearest%20Neighbors%20%26%20Kernel%20Regression.pdf) |
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

### Cross-Validation
* In case of insuuffient data to form a separate validation set.
* Then perform k-fold cross validation.
* Here the training set is divided into blocks and each block is treated as the validation set.
  - training block -> parameters or coefficients are extimated.
  - validation block the error is computed.
* The average error across all validation set is computed.  

---
## Feature Selection & Lasso
Various methods to search over models with different number of features.
* **All Subset** - exhaustive approach, where feature combinations with least RSS is chosen.
* **Greedy Algorithm** - forward selection - suboptimal solution but eventually provides the desired model set and is more efficient.

#### Lasso objective function - L1 regularized regression
* It leads to sparse solutions.
* L1 norm = RSS(w) + lambda ||w||

#### Coefficient path
* Here the coefficient path becomes sparser with increasing lambda value. This provideds better feature solutions.

### Coordinate Descent
* Better model since it is difficult to find the derivate of an absolute value. Need to use sub-gradients, alternative is coordinate descent.
* Iterate through the different dimensions of the objective or different features of the regression model.
* The coefficients for lasso was setup based on "soft-thresholding" - provides sparse solutions.

---
## Nearest Neighbor & Kernel Regression - Nonparametric fits
### 1-NN - simple procedure
- Look for the most similar dataset observation and base the predictions on it.

### Weighted k-NN
- weigh the more similar observations more than those less similar in the list of k-NN.
- Average across the rating to form the estimated prediction.

### Kernel Regression
* Weight all the points rather than just weighting NN.
* The kernels have a bandwidth - lambda, outside which the observations are 0. Within the range/bandwidth also the observations can decay based on how far they are from the target point.
* It leads to local constant fits.
* Parametric fits -> global constant fits.

---
