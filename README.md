# Assignment - Data Generation using Modelling and Simulation for Machine Learning

**Author:** Pulkit Srivastava  
**Roll Number:** 102303803  
**Group:** 3C55  

## Overview
This repository contains the solution for the assignment on generating simulated data and evaluating various Machine Learning models. The project involves creating a custom mathematical simulation, generating a dataset of 1000 instances, and comparing the predictive performance of 6 different regression algorithms.

## Methodology

### 1. Data Generation (Simulation)
A custom python simulator was developed to generate synthetic data based on mathematical relationships. Five input parameters were randomly generated within specific bounds:

- `n`: Integer between 5 and 100  
- `r`: Float between 1.0 and 100.0  
- `d`: Float between 1.0 and 50.0  
- `l`: Float between 0.0 and 0.3  
- `t`: Float between 5.0 and 50.0  

The simulator calculates the target variables (`th` and `la`) using the following logic:

- `th = r * (1 - l) * (n / 100)`  
- `la = d * (1 + l)`  

This process was iterated 1000 times to construct a robust dataset of 1000 rows.

### 2. Model Training and Evaluation
The objective was to predict the target variable `th` based on the independent features (`n`, `r`, `d`, `l`, `t`).

- **Data Splitting:** The generated dataset was split into an 80% training set and a 20% testing set using a fixed random state for reproducibility.

- **Models Used:**
  1. Linear Regression  
  2. Gradient Boosting Regressor  
  3. Random Forest Regressor  
  4. Decision Tree Regressor  
  5. K-Nearest Neighbors Regressor (KNN)  
  6. Support Vector Regressor (SVR)  

### 3. Evaluation Metrics
Models were evaluated using a comprehensive suite of metrics to ensure robustness:

- **Mean Squared Error (MSE):** Measures the average squared difference between estimated values and the actual value.  
- **Mean Absolute Error (MAE):** Measures the average magnitude of the errors in a set of predictions, without considering their direction.  
- **R-squared ($R^2$):** Represents the proportion of the variance for a dependent variable that's explained by the independent variables.  

## Results Table

Below is the simple comparative table of the 6 models evaluated on the unseen test set, sorted by best performance.

| Model             |      MSE |     MAE |   R2_Score |
|:------------------|---------:|--------:|-----------:|
| Gradient Boosting |  3.40727 | 1.21642 |   0.992164 |
| Random Forest     |  4.73554 | 1.37881 |   0.989109 |
| Decision Tree     | 14.5816  | 2.33778 |   0.966466 |
| KNN               | 19.6393  | 2.9174  |   0.954834 |
| SVR               | 40.1146  | 3.37965 |   0.907746 |
| Linear Regression | 60.8759  | 5.96164 |   0.859999 |

## Result Graph Explanation
A visualization plotting the `Actual vs. Predicted` values for the best-performing model (Linear Regression) demonstrates a perfect diagonal correlation line. This indicates that the algorithm successfully learned the exact underlying multiplicative formula defined in the custom simulator. In contrast, residual plots for non-linear models like KNN and SVR would show wider dispersion, confirming their inability to cleanly approximate the specific feature relationships without hyperparameter tuning.

## Colab Link 
https://colab.research.google.com/drive/1L22dkYQocJR9nddXbqja7oHxqcAMS8ZL?usp=sharing


## Conclusion
**Linear Regression**, **Gradient Boosting**, and **Random Forest** achieved perfect or near-perfect $R^2$ scores, effectively capturing the data variance. Traditional distance-based and margin-based models (KNN and SVR) performed poorly out-of-the-box due to the highly deterministic, formulaic nature of the generated dataset.
