# Interactive Supervised Machine Learning Classifier

#### By Anthony Travlos

## Project Description

This project is an interactive supervised machine learning classifier that allows the user to upload a dataset (or choose from the samples provided), walk through data preprocessing steps, and perform and evaluate either a K-Nearest Neighbor or Decision Tree Model. Users are invited to explore with the machine learning model setup by choosing different target/feature columns and tuning hyperparameters (k, max depth, minimum sample splits, etc.) to see how these factors effect model performance. Finally, users will be able to evaluate their model with cross-validation (GridSearchCV), confusion matrices, and classification reports.

## Supervised Machine Learning Overview

__Supervised Machine Learning__  uses labeled data to train a model that makes a prediction on new, unseen data. This app allows users to choose from 2 different supervised machine learning classification models:

- **K-Nearest Neighbors** finds the "k" nearest data points to a new data point and makes predictions based on the class labels.
- **Decision Tree** predicts the value of a target variable by making a series of decisions based on the input features. Each decision made is a node and each possible outcome is a branch.

## App Instructions

1. In terminal, navigate to working directory and run "streamlit run ML_Streamlit_App.py". The app will open in the local browser.
2. Upload .csv file or choose from sample dataset.
3. After reviewing the data and preprocessing steps, choose to run classification model.
4. Choose Supervised Machine Learning Model (K-Nearest Neighbors or Decision Tree).
5. Select Target Column and Feature Column(s).
6. Set Test Set Size.
7. Choose whether to use unscaled/scaled data.
8. Choose Hyperparameters.
9. Run GridSearchCV Optimization.
10. The Confusion Matrix and Classification Report will generate. Feel free to explore with different parameters/datasets to see how they affect evaluation criteria!

## Libraries

- Streamlit
- Numpy
- Pandas
- Seaborn
- Matplotlib
- Graphviz
- sklearn

## Additional Resources

- [One-Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [K-Nearest Neighbors](https://www.ibm.com/think/topics/knn#:~:text=The%20k%2Dnearest%20neighbors%20(KNN)%20algorithm%20is%20a%20non,used%20in%20machine%20learning%20today.)
- [Decision Trees](https://www.ibm.com/think/topics/decision-trees)
- [Scaling Data](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Confusion Matrix](https://www.datacamp.com/tutorial/what-is-a-confusion-matrix-in-machine-learning)
- [Classification Report](https://www.nb-data.com/p/breaking-down-the-classification)


*“A baby learns to crawl, walk and then run.  We are in the crawling stage when it comes to applying machine learning.” ~Dave Waters*
