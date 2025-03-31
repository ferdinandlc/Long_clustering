import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pacmap
import hdbscan

from embassy import align_and_measure
from sklearn.preprocessing import StandardScaler
np.random.seed(42)

path = os.getcwd()

def snapshot_clusters(embeddings_path, file_name, min_cluster_size = 50, min_samples = 10, cluster_selection_epsilon = 0.1):
    
    
    embeddings = np.load(embeddings_path, allow_pickle = True)
    embeddings_std = [StandardScaler().fit_transform(embeddings[i]) for i in range(len(embeddings))]
    X_transformed = []
    for i in range(len(embeddings_std)):
        embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
        X_transformed.append(embedding.fit_transform(embeddings_std[i], init="pca"))
        
    del embeddings_std
        
    new_X = []
    new_X.append(align_and_measure(X_transformed[0],X_transformed[1])['emb1'])
    new_X.append(align_and_measure(X_transformed[0],X_transformed[1])['emb2'])
    for t in range(2, len(X_transformed)):
        new_X.append(align_and_measure(new_X[-1],X_transformed[t])['emb2'])
    
    del X_transformed
    
    cluster_labels = []
    for X in new_X:
      clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples = min_samples, cluster_selection_method = 'leaf',cluster_selection_epsilon=cluster_selection_epsilon, prediction_data=True)
      clusterer.fit(X)
      cluster_labels.append(list(clusterer.labels_))
      

    np.save(path+'data/clustering/' + file_name, cluster_labels)