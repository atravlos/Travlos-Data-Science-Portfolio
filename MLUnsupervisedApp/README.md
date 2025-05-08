# Interactive Unsupervised Machine Learning Clustering App

#### By Anthony Travlos

## 🗒️ Project Description

This project is an interactive unsupervised machine learning clustering app that allows the user to upload a dataset (or choose from the samples provided), walk through data preprocessing steps, and perform and evaluate either a K-Means or Hierarchical Clustering model. Users are invited to explore with the machine learning model parameters by choosing different k values to see how these factors effect model performance. Finally, users will be able to evaluate their model by comparing clusters to the true labels, calculating an accuracy score, and evaluating for the optimal number of clusters (k).

## 📋 Unsupervised Machine Learning Overview

__Unsupervised Machine Learning__  uses unlabeled data to train a model that identifies patterns and groups within the data. This app allows users to choose from 2 different unsupervised machine learning classification models:

- **K-Means Clustering** partitions the data into k clusters by iteratively assigning points to the nearest cluster centroid and then updating the centroids based on the cluster’s mean.
- **Hierarchical Clustering** starts with each data point as an individual cluster and iteratively merges the closest clusters until a single cluster remains.
## ✍️ App Instructions

1. In terminal, navigate to working directory and run "streamlit run ML_Streamlit_App.py". The app will open in the local browser.
2. Upload .csv file or choose from sample dataset.
3. Choose Unsupervised Machine Learning Model (K-Means or Hierarchical Clustering).
4. Choose number of clusters.
5. Choose whether to see plots or evaluate for the optimal k.
6. Feel free to explore with different parameters/datasets to see how they affect evaluation criteria!          


## 📚 Libraries

- Streamlit
- Numpy
- Pandas
- Seaborn
- Matplotlib
- sklearn

## 🗄️ Additional Resources

- [K-Means](https://www.geeksforgeeks.org/k-means-clustering-introduction/)
- [Hierarchical Clustering](https://www.displayr.com/what-is-hierarchical-clustering/#:~:text=Hierarchical%20clustering%2C%20also%20known%20as,broadly%20similar%20to%20each%20other.)
- [Scaling Data](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)
- [Accuracy Score](https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall#:~:text=Accuracy%20is%20a%20metric%20that,often%20the%20model%20is%20right%3F)
- [Principal Component Analysis](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
- [Determining Optimal k](https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/)


*“A baby learns to crawl, walk and then run.  We are in the crawling stage when it comes to applying machine learning.” ~Dave Waters*
