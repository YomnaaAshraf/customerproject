import streamlit as st

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from sklearn_extra.cluster import KMedoids

from sklearn.metrics import silhouette_score



def cluster_and_evaluate(rfm_features, algorithm):

    if algorithm == "K-Means":

        model = KMeans(n_clusters=5, n_init=10)

    elif algorithm == "Hierarchical Clustering":

        model = AgglomerativeClustering(n_clusters=5)

    elif algorithm == "DBSCAN":

        model = DBSCAN(eps=0.5, min_samples=10)

    elif algorithm == "K-Medoids":

        model = KMedoids(n_clusters=5)



    labels = model.fit_predict(rfm_features)

    score = silhouette_score(rfm_features, labels)

    return labels, score



st.title("Customer Segmentation with RFM Analysis")



labels, score = cluster_and_evaluate(rfm_df[['Recency', 'Frequency', 'Monetary']], "K-Means")





rfm_df['Cluster'] = labels



st.write("RFM Data:")

st.dataframe(rfm_df)



st.write("Cluster Labels:")

st.write(rfm_df[['CustomerID', 'Cluster']])