import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from data_loading.loader import data_load
from visualization.plotter import plot_clustering_metrics, compute_metrics, compute_mean_metric

def main():
    
    ## config_names = ['newcom25', 'mergecom25', 'chgnode25']
    
    # Load data and perform clustering
    dic_clus, commu, labels, Lref = data_load('newcom25')

    # Visualize results
    dic_AMI, dic_ARI, dic_H, dic_C, dic_FMI, dic_NBCOM = compute_metrics(dic_clus, commu)
    plot_clustering_metrics(dic_AMI, dic_ARI, dic_H, dic_C, dic_FMI, dic_NBCOM)
    compute_mean_metric(dic_clus, dic_AMI, dic_ARI, dic_H, dic_C, dic_FMI, dic_NBCOM)

if __name__ == "__main__":
    main()
