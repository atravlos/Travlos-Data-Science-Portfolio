# ML Streamlit App - Interactive Supervised Machine Learning Classifier

#### By Anthony Travlos

## Project Description

This project is an interactive supervised machine learning classifier that allows the user to upload a dataset (or choose from the samples provided), walk through data preprocessing steps, and perform and evaluate either a K-Nearest Neighbor or Decision Tree Model. Users are invited to explore with the machine learning model setup by choosing different target/feature columns and tuning hyperparameters (k, max depth, minimum sample splits, etc.) to see how these factors effect model performance. Finally, users will be able to evaluate their model with cross-validation (GridSearchCV), confusion matrices, and classification reports.

## Supervised Machine Learning Overview

__Supervised Machine Learning__  uses labeled data to train a model that makes a prediction on new, unseen data. This app allows users to choose from 2 different supervised machine learning classification models:

- **K-Nearest Neighbors** finds the "k" nearest data points to a new data point and makes predictions based on the class labels.
- **Decision Tree** predicts the value of a target variable by making a series of decisions based on the input features. Each decision made is a node and each possible outcome is a branch.
