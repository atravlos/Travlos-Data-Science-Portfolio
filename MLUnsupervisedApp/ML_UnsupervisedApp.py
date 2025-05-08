import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --------------------
# ABOUT SECTION
# --------------------


st.title("Interactive Unsupervised Machine Learning App")
st.markdown("""
<a id="app-info"></a>
### About This Application
This simple, user-friendly application allows you to either upload your own CSV file or choose a sample dataset. Then, you may choose which unsupervised machine learning clustering model you would like to run (K-Means or Hierarchical Clustering). From there, you can explore different parameters, plots and evalutation criteria!
            
### Instructions:
1. In terminal, navigate to working directory and run "streamlit run ML_Streamlit_App.py". The app will open in the local browser.
2. Upload .csv file or choose from sample dataset.
3. Choose Unsupervised Machine Learning Model (K-Means or Hierarchical Clustering).
4. Choose number of clusters.
5. Choose whether to see plots or evaluate for the optimal k.
6. Feel free to explore with different parameters/datasets to see how they affect evaluation criteria!          

""", unsafe_allow_html=True)


# --------------------
# SIDEBAR - TABLE OF CONTENTS
# --------------------


st.sidebar.markdown("""
# Table of Contents
- [About this Application](#app-info)
- [Choose/Upload Your Dataset](#data-upload)
- [Choose Clustering Method](#model)
""")


# --------------------
# UPLOAD / CHOOSE SAMPLE DATASET
# --------------------


st.markdown("""### Choose Your Dataset <a id="data-upload"></a>""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.write(df.head())

    # Ask user to select the label column (optional)
    label_column = st.selectbox("Select the true label column (optional)", options=["None"] + list(df.columns))
    if label_column != "None":
        y = df[label_column].values
        df = df.drop(columns=[label_column])
        target_names = np.unique(y)
    else:
        y = None
        target_names = None
else:
    demo_dataset = st.selectbox("Or select a demo dataset:", ["Iris", "Wine", "Breast Cancer"])
    if demo_dataset == "Iris":
        from sklearn.datasets import load_iris
        data = load_iris()
    elif demo_dataset == "Wine":
        from sklearn.datasets import load_wine
        data = load_wine()
    else:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    y = data.target
    target_names = data.target_names
    st.success(f"✅ Loaded demo dataset: {demo_dataset}")
    if st.checkbox("See Raw Dataset"):
        st.dataframe(df, height=300)

# --------------------
# PREPROCESSING
# --------------------

# Drop non-numeric columns (necessary for the machine learning models this app uses.)
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
if len(non_numeric_cols) > 0:
    st.warning(f"The following non-numeric columns will be dropped automatically: {', '.join(non_numeric_cols)}")
    df = df.drop(columns=non_numeric_cols)

# Drop target variable
if "target" in df.columns:
    df = df.drop(columns=["target"])

# Drop missing values
# I chose to simply drop NAs. Another option could include giving the user an option to impute.
if y is not None:
    # Combine into one DataFrame to drop NAs across both features and labels
    df['__temp_y__'] = y
    df = df.dropna()
    y = df['__temp_y__'].values
    df = df.drop(columns=['__temp_y__'])
else:
    df = df.dropna()

# Standardize/Scale features
# Scaling is critical here because KMeans relies on distance calculations and can be biased by the scale of features.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(X_scaled, columns=df.columns)

st.success("✅ Preprocessing complete. Data ready for clustering.")
st.dataframe(df_scaled, height=300)


# --------------------
# UNSUPERVISED MACHINE LEARNING - CLUSTERING
# --------------------


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

st.markdown("""### Choose Clustering Method <a id="model"></a>""",unsafe_allow_html=True)

clustering_method = st.radio("Select clustering method", ["K-Means", "Hierarchical"])
st.caption("KMeans clustering partitions the data into k clusters by iteratively assigning points to the nearest cluster centroid and then updating the centroids based on the cluster’s mean.")
st.caption("Hierarchical clustering starts with each data point as an individual cluster and iteratively merges the closest clusters until a single cluster remains. ")

clusters = st.slider("Choose number of clusters (k)", min_value=2, max_value=10, value=3)

# --------------------
# K-MEANS
# --------------------


if clustering_method == "K-Means":
    model = KMeans(n_clusters=clusters, random_state=42)
    cluster_labels = model.fit_predict(X_scaled)
    # --------------------
    # PCA for Visualization
    # --------------------
    # PCA, also known as principal component analysis reduces the dimensions of the dataset to include less variables that still contain most of the original variables' data
    # For more information on PCA: https://builtin.com/data-science/step-step-explanation-principal-component-analysis
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    if st.checkbox("Show PCA Cluster Plot"):
        st.write("Visualization helps us understand how well KMeans has partitioned the data. However, our dataset is high-dimensional, so we first reduce it to 2 dimensions using PCA for visualization. We then plot the PCA scores with colors corresponding to the cluster assignments.")
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(clusters):
            ax.scatter(
                X_pca[cluster_labels == i, 0],
                X_pca[cluster_labels == i, 1],
                alpha=0.7,
                edgecolor='k',
                s=60,
                label=f'Cluster {i}'
            )
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('KMeans Clustering: 2D PCA Projection')
        ax.legend(loc='best')
        ax.grid(True)

        st.pyplot(fig)

    if st.checkbox("Plot Clusters with True Labels"):
        st.write("Even though KMeans is unsupervised, we can compare its cluster assignments with the actual labels to gauge performance.")
        fig, ax = plt.subplots(figsize=(8, 6))

        # Use a dynamic color palette that adjusts to the number of classes
        palette = sns.color_palette("tab10", n_colors=len(np.unique(y)))

        for i, target_name in enumerate(target_names):
            ax.scatter(
                X_pca[y == i, 0],
                X_pca[y == i, 1],
                color=palette[i],
                alpha=0.7,
                edgecolor='k',
                s=60,
                label=target_name
            )

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('True Labels: 2D PCA Projection')
        ax.legend(loc='best')
        ax.grid(True)

        st.pyplot(fig)

            # --------------------
            # Evaluation - Accuracy Score
            # --------------------
            # There are other metrics like Adjusted Rand Index (ARI) that are also used to evaluate clusters to true values.
        if st.checkbox("Show Accuracy Score") and y is not None:
            from sklearn.metrics import accuracy_score
            from scipy.stats import mode

            st.write("Accuracy is calculated as the number of correctly labeled predictions divided by the total number of predictions.")
            def map_clusters_to_labels(clusters, true_labels):
                labels_map = {}
                for i in np.unique(clusters):
                    match = mode(true_labels[clusters == i], keepdims=True).mode[0]
                    labels_map[i] = match
                return np.array([labels_map[c] for c in clusters])

            mapped_labels = map_clusters_to_labels(cluster_labels, y)
            accuracy = accuracy_score(y, mapped_labels)
            st.success(f"✅ Accuracy Score: **{accuracy:.2%}**")

    if st.checkbox("Evaluate for Best Number of Clusters"):
        from sklearn.metrics import silhouette_score
        st.markdown("""
    There are 2 popular methods to evaluate the best number of clusters:

    - **Elbow Method:**
    We plot the Within-Cluster Sum of Squares (WCSS) against different values of k. The "elbow" point—where the rate of decrease sharply changes—suggests an optimal value for k.

    - **Silhouette Score:**
    This score quantifies how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better clustering. We compute the average silhouette score for different values of k and select the one with the highest score.
                    """)
        
        ks = range(2, 11)
        wcss = []
        silhouette_scores = []

        for k in ks:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X_scaled)
            labels_k = km.labels_
            wcss.append(km.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels_k))

        st.write("📉 WCSS Values:", wcss)
        st.write("🎯 Silhouette Scores:", silhouette_scores)

        # Plot both metrics side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(ks, wcss, marker='o')
        axes[0].set_title("Elbow Method for Optimal k")
        axes[0].set_xlabel("Number of clusters (k)")
        axes[0].set_ylabel("WCSS")
        axes[0].grid(True)

        axes[1].plot(ks, silhouette_scores, marker='o', color='green')
        axes[1].set_title("Silhouette Score for Optimal k")
        axes[1].set_xlabel("Number of clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].grid(True)

        plt.tight_layout()
        st.pyplot(fig)


# --------------------
# HIERARCHICAL CLUSTERING
# --------------------


else:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA

        # Let user pick column for labeling (if available)
    label_col = None
    if len(df.columns) > 0:
        label_col = st.selectbox("Choose column for dendrogram labels:", options=["Row index"] + list(df.columns))

    # Prepare labels
    if label_col == "Row index" or label_col is None:
        dendro_labels = [f"Sample {i}" for i in range(len(df))]
    else:
        dendro_labels = df[label_col].astype(str).tolist()

    # Build Lineage Matrix
    Z = linkage(X_scaled, method="ward")

    if st.checkbox("Show Hierarchical Tree"):
        st.markdown("""
* **Ward linkage** merges clusters that yield the *smallest* increase in total within‑cluster variance.  
* The dendrogram gives us two insights:  
  1. Similarity structure (who merges early).  
  2. Reasonable cut heights (horizontal line) for k clusters.  
We truncate to the last 30 merges to keep the plot readable.                     
                    """)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z, 
                   labels=dendro_labels,
                   truncate_mode='level', p=5, ax=ax)
        st.pyplot(fig)

        n_clusters = st.slider("Choose number of clusters", min_value=2, max_value=10, value=3)

        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        cluster_labels = agg.fit_predict(X_scaled)
        df["Cluster"] = cluster_labels
        
        # --------------------
        # PCA for Visualization
        # --------------------
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        if st.checkbox("Show PCA Cluster Plot (Agglomerative)"):

            fig, ax = plt.subplots(figsize=(10, 7))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                                c=cluster_labels,
                                cmap='viridis',
                                s=60,
                                edgecolor='k',
                                alpha=0.7)
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title('Agglomerative Clustering on Data (via PCA)')
            ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.grid(True)

            st.pyplot(fig)

        if st.checkbox("Evaluate for Optimal k Using the Silhouette Elbow Method"):
            from sklearn.metrics import silhouette_score
            st.markdown("""
            This plot shows the average silhouette score for different numbers of clusters (k).
            
            - A **higher score** indicates better-defined clusters.
            - The **best k** maximizes the silhouette score.
            """)

            k_range = range(2, 11)
            sil_scores = []

            for k in k_range:
                labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                sil_scores.append(score)

            # Plot silhouette scores
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(list(k_range), sil_scores, marker="o")
            ax.set_xticks(list(k_range))
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Average Silhouette Score")
            ax.set_title("Silhouette Analysis for Agglomerative Clustering")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Show best k
            best_k = list(k_range)[np.argmax(sil_scores)]
            best_score = max(sil_scores)
            st.success(f"✅ Best k by silhouette: **{best_k}** (score = {best_score:.3f})")







    
    