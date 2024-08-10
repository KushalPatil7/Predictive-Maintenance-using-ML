# predictive-maintenance-streamlit-app
Project Overview
This project aims to evaluate and compare the performance of various machine learning classification models on a given dataset. The goal is to determine which model performs best based on accuracy metrics and to visualize the margin difference between training and testing accuracies.

# Models Used
The following classification models have been evaluated:

# Logistic Regression: A linear model for binary classification tasks.
# K-Neighbors Classifier: A non-parametric model that classifies based on the majority vote of nearest neighbors.
# Decision Tree: A model that makes decisions based on feature values, creating a tree-like structure.
# Random Forest Classifier: An ensemble method that combines multiple decision trees to improve performance.
# Gradient Boosting: An ensemble method that builds trees sequentially, correcting the errors of previous trees.
# XGBoost: An optimized gradient boosting library designed to be highly efficient and scalable.
# CatBoost: A gradient boosting library that handles categorical features effectively.
# AdaBoost Classifier: An ensemble method that combines weak classifiers to create a strong classifier.
Methods
# Data Preparation:

The dataset is split into training and testing sets.
Models are trained on the training set and evaluated on the testing set.
# Model Training:

Each model is trained using the fit method.
Predictions are made on both training and testing data.
# Evaluation:

Accuracy scores for both training and testing data are calculated using accuracy_score.
The margin difference between training and testing accuracies is computed to assess overfitting or underfitting.
# Visualization:

Pie charts are generated to visualize the margin difference between training and testing accuracy scores for each model.
The pie charts highlight the proportion of training versus testing accuracy for each model.
# Model Selection:

The model with the highest testing accuracy is selected as the best model.
# Saving the Best Model:

The best-performing model is saved to a file using pickle for future use.
