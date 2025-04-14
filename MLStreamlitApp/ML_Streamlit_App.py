import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn import tree

# -----------------------------------------------
# Application Information
# -----------------------------------------------
st.title("Interactive Supervised Machine Learning Classifier")
st.markdown("""
<a id="app-info"></a>
### About This Application
This application allows you to either upload your own CSV file or choose a sample dataset (Titanic, Iris, Wine). It provides:
- **Dataset Overview:** Displays the first few rows and key statistics.
- **Interactive Supervised Machine Learning Classifier:** Choose a supervised machine learning model, configure model hyperparameters, select target/feature columns.
""",
unsafe_allow_html=True)

st.markdown("""
### Instructions:
1. Upload .csv file or choose from sample dataset.
2. After reviewing the data and preprocessing steps, choose to run classification model.
3. Choose Supervised Machine Learning Model (K-Nearest Neighbors or Decision Tree).
4. Select Target Column and Feature Column(s).
5. Set Test Set Size.
6. Choose whether to use unscaled/scaled data.
7. Choose Hyperparameters.
8. Run GridSearchCV Optimization.
9. The Confusion Matrix and Classification Report will generate. Feel free to explore with different parameters/datasets to see how they affect evaluation criteria!
""")

# Main content with anchor tags for navigation

# Sidebar: Table of Contents using markdown
st.sidebar.markdown("""
# Table of Contents
- [About this Application](#app-info)
- [Choose Your Dataset](#data-upload)
- [Preprocessing](#preprocessing)
- [Model Setup](#setup)
- [Model Evaluation](#evaluation)
""", unsafe_allow_html=True)

# -----------------------------------------------
# Data Upload / Demo Dataset Option
# -----------------------------------------------
st.markdown("""### Choose Your Dataset <a id="data-upload"></a>""", unsafe_allow_html=True)
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
# -----------------------------------------------
# Preprocessing: Drop Missing Values and One-Hot Encode
# -----------------------------------------------
st.subheader("Raw Dataset Preview")
st.dataframe(df.head())

st.markdown("***NOTE: This application removes missing values by default!***")
df.dropna(inplace=True)
st.markdown("***NOTE: This application converts categorical data to numerical data using one-hot encoding!***")
st.markdown("*What is one-hot encoding?*")
st.caption("- One-hot encoding is a method that replaces categorical data with numerical binary data. We do this because machine learning models process numerical data types.")

# One-hot encode all categorical columns (object or category types)
categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
if categorical_columns:
   st.write("One-hot encoding the following categorical columns:", categorical_columns)
   df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
else:
   st.caption("No categorical columns detected for one-hot encoding.")

# Convert columns to numeric where possible
for col in df.columns:
    # Attempt to convert each column to numeric, ignoring errors
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Convert boolean columns to integers
bool_cols = df.select_dtypes(include=['bool']).columns
if not bool_cols.empty:
    st.write("Converting the following boolean columns to integers:", bool_cols.tolist())
    df[bool_cols] = df[bool_cols].astype(int)
else:
    st.caption("Data successfully converted to numerical data type!")


# -----------------------------------------------
# Display Dataset Information
# -----------------------------------------------

st.markdown("""### Preprocessing Dataset Overview <a id="preprocessing"></a>""", unsafe_allow_html=True)
st.caption("This is the data format the machine learning model will use!")
st.dataframe(df.head())

st.write("### Statistical Summary")
st.dataframe(df.describe())

with st.expander("Click to view full Dataset Information"):
    st.write("**Dataset Shape:**", df.shape)
    st.write("**Column Names:**", df.columns.tolist())

# -----------------------------------------------
# Classification Section
# -----------------------------------------------
if st.checkbox("Run Supervised Machine Learning Model"):
    st.markdown("""### Supervised Machine Learning Setup <a id="setup"></a>""", unsafe_allow_html=True)
    
    # Let the user choose the classifier
    classifier_choice = st.selectbox("Select Machine Learning Model", ["K-Nearest Neighbors", "Decision Tree"])
    st.caption("K-Nearest Neighbors: Finds the 'k' nearest data points to classify each data point.")
    st.caption("Decision Tree: Predicts the target by making a series of sequential decisions.")
    
    # Let the user choose the target column
    columns = df.columns.tolist()
    target = st.selectbox("Select target column", columns)
    
    # For feature selection, show only numeric columns (excluding the target)
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    if target in numeric_features:
        numeric_features.remove(target)
    
    features = st.multiselect("Select feature columns (choose at least one)", numeric_features)
    
    if not features:
        st.error("Please select at least one numeric feature column for classification.")
        st.stop()
        
    X = df[features]
    y = df[target]
    
    # Split the data into training and testing sets
    test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
    
    # Option for scaling features
    data_type = st.radio("Data Type", options=["Unscaled", "Scaled"])
    st.caption("Unscaled data is the numerical data as it exists in the dataset.")
    st.caption("Scaled data transforms the data into a range (Ex: 0 to 1, standard deviations, etc.).")
    if data_type == "Scaled":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    # Train the selected classifier based on user's choice
    if classifier_choice == "K-Nearest Neighbors":
        k = st.slider("Select number of neighbors (k, odd values only)", min_value=1, max_value=21, step=2, value=5)
        model = KNeighborsClassifier(n_neighbors=k)
    elif classifier_choice == "Decision Tree":
        max_depth = st.slider("Select maximum depth for decision tree", min_value=1, max_value=20, value=5)
        min_samples_split = st.slider("Select minimum number of samples required to split an internal node", min_value=2, max_value=20, value=2)
        min_samples_leaf = st.slider("Select minimum number of samples required to be at a leaf node", min_value=1, max_value=20, value=1)
        model = DecisionTreeClassifier(max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=42)

    # Option to perform GridSearchCV for hyperparameter tuning
    if st.checkbox("Run GridSearchCV Optimization (This may take a minute...)"):
        st.caption("GridSearchCV splits the data into folds and iteratively trains on these folds (through cross-validation) to determine the best parameters.")
        st.subheader("Grid Search Hyperparameter Tuning")
        if classifier_choice == "K-Nearest Neighbors":
            param_grid = {"n_neighbors": list(range(1, 22, 2))}
            grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        elif classifier_choice == "Decision Tree":
            param_grid = {
                "max_depth": list(range(1, 21)),
                "min_samples_split": list(range(2, 21)),
                "min_samples_leaf": list(range(1, 21))
            }
            grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                                param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_params = grid.best_params_
        best_score = grid.best_score_
        st.write("Best Parameters:", best_params)
        st.write("Best Cross Validation Score:", best_score)

        apply_best = st.checkbox("Apply best parameters from GridSearchCV")
        if apply_best:
            model = grid.best_estimator_
            
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy_val = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy: {accuracy_val:.2f}**")
    
    # After evaluating the model, add the following for Decision Tree visualization:

    if classifier_choice == "Decision Tree":
        st.subheader("Decision Tree Visualization")
        # Generate DOT data
        dot_data = tree.export_graphviz(
            model,
            out_file=None,
            feature_names=features,
            class_names=[str(cls) for cls in np.unique(y)],
            filled=True,
            rounded=True,
            special_characters=True
        )
        # Display the DOT data using Streamlit's graphviz_chart
        st.graphviz_chart(dot_data)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.markdown("""### Confusion Matrix <a id="evaluation"></a>""", unsafe_allow_html=True)
    st.caption("A confusion matrix is used to evaluate classification. The rows represent the actual classifications. The columns represent the predicted classifications." \
    " The diagonal (top left to bottom right) represents correct predictions. Off the diagonal represents incorrect predictions.")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix ({data_type} Data)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report (displayed as a cleaner table)
    st.subheader("Classification Report")
    st.caption("**Precision:** Percentage of positive classifications that are actually positive. This is crucial when it is especially important for positive predictions to be accirate.")
    st.caption("**Recall:** Percentage of actual positives that were classified as positive. This is crucial when false negatives are more costly than false positives.")
    st.caption("**F1 Score:** Harmonic average of precision and recall. Formula: 2 * ((Precision * Recall) / (Precision + Recall))")
    from sklearn.metrics import classification_report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)
