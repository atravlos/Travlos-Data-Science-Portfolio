import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

# Application Setup

st.title("Interactive Unsupervised Machine Learning App")

# About Section

# Table of Contents

st.markdown("""### Choose Your Dataset""")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    st.info("No file uploaded. Please select one of the demo datasets below.")
    demo_dataset = st.selectbox("Select Demo Dataset", ["Titanic", "Iris", "Wine"])
    if demo_dataset == "Titanic":
        df = sns.load_dataset("titanic")
    elif demo_dataset == "Iris":
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["target"] = iris.target
    elif demo_dataset == "Wine":
        from sklearn.datasets import load_wine
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df["target"] = wine.target

st.caption("Titanic: Lists the attributes of members on the Titanic and if they survived the shipwreck or not.")
st.caption("Iris: Lists sepal and petal lengths/widths of various Iris flowers.")
st.caption("Wine: Lists attributes of different types of wine.")

# Preprocessing

st.subheader("Raw Dataset Preview")
st.dataframe(df.head())

st.markdown("***NOTE: This application removes missing values by default!***")
df.dropna(inplace=True)
# I chose to simply drop NAs. Another option could include giving the user an option to impute.

st.markdown("***NOTE: This application converts categorical data to numerical data using one-hot encoding!***")
st.markdown("*What is one-hot encoding?*")
st.caption("- One-hot encoding is a method that replaces categorical data with numerical binary data. We do this because machine learning models process numerical data types.")

# For more information on one-hot encoding: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

# The next 3 blocks of code ensure that all columns are converted to numeric data. The boolean block was added because I ran into trouble with converting boolean data to integer.

categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
if categorical_columns:
   st.write("One-hot encoding the following categorical columns:", categorical_columns)
   df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
else:
   st.caption("No categorical columns detected for one-hot encoding.")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

bool_cols = df.select_dtypes(include=['bool']).columns
if not bool_cols.empty:
    st.write("Converting the following boolean columns to integers:", bool_cols.tolist())
    df[bool_cols] = df[bool_cols].astype(int)
else:
    st.caption("Data successfully converted to numerical data type!")

# Display Dataset Information

st.markdown("""### Preprocessing Dataset Overview <a id="preprocessing"></a>""", unsafe_allow_html=True)
st.caption("This is the data format the machine learning model will use!")
st.dataframe(df.head())

st.write("### Statistical Summary")
st.dataframe(df.describe())

with st.expander("Click to view full Dataset Information"):
    st.write("**Dataset Shape:**", df.shape)
    st.write("**Column Names:**", df.columns.tolist())

# Unsupervised Machine Learning Model

if st.checkbox("Run Unsupervised Machine Learning Model"):
    st.markdown("""### Unsupervised Machine Learning Setup""")
   
    # Let the user choose the classifier
    classifier_choice = st.selectbox("Select Machine Learning Model", ["KMeans Clustering", "Hierarchical Clustering"])
    #st.caption("K-Nearest Neighbors: Finds the 'k' nearest data points to classify each data point.")

    






