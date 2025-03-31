import numpy as np
import json
import os

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
from clustering.analyzer import long_clustering
from sklearn.metrics.cluster import adjusted_mutual_info_score

path = os.getcwd()

def load_sensitivity(config_name, method_name):
    with open(path+'/data/json_config/config_'+config_name+'.json', 'r') as f:
        config_data = json.load(f)
    dic_clus = config_data['dic_clus']
    
    new_dic_clus = dic_clus[method_name]
    cluster_labels = np.load(path+'/data/clustering/'+config_name+'/'+method_name)
    
    lst_clusters = []
    if new_dic_clus[2] != []:
      lst_ind = new_dic_clus[2]
    else:
      lst_ind = range(len(cluster_labels))
    for i in lst_ind:
      dic = {}
      for j in range(len(cluster_labels[i])):
        if cluster_labels[i][j] not in dic:
          dic[cluster_labels[i][j]] = [j]
        else:
          dic[cluster_labels[i][j]].append(j)
      lst_clusters.append(dic)
      
    commu = [np.load(path+'/data/label/'+config_name+'/node_communities_{}.npy'.format(i)) for i in range(25)]

    return new_dic_clus, lst_clusters, commu

def evaluate_model(X , new_dic_clus, lst_clusters, commu, nb_steps = 25):
  new_d = long_clustering(lst_clusters, new_dic_clus[1],  similarity_treshold = X[0], min_instance = int(X[1]), t_periods = int(X[2]))
  new_dic_clus.append(new_d)
  lst_AMI = []
  for t in range(nb_steps):
    labelling = []
    true_label = []
    coms = np.unique([new_dic_clus[-1][i][t] for i in range(500)])
    coms = np.setdiff1d(coms,np.array([-1]))
    for i in range(500):
      if len(new_dic_clus[1]) > 0:
        node = int(new_dic_clus[1][t][i])
      else:
        node = i
      true_label.append(commu[t][node])
      labelling.append(new_dic_clus[-1][i][t])
    lst_AMI.append(adjusted_mutual_info_score(true_label, labelling))
  return np.mean(lst_AMI)

def to_int(a):
    return np.array([a[0], round(a[1]), round(a[2])])

def evaluate_sensitivity(problem, n_sobol, new_dic_clus, lst_clusters, commu):
    param_values = saltelli.sample(problem, 64)
    param_values = np.apply_along_axis(to_int, 1, param_values)
    
    Y = np.zeros([param_values.shape[0]])

    for i, X in enumerate(param_values):
      Y[i] = evaluate_model(X, new_dic_clus, lst_clusters, commu)
    Si = sobol.analyze(problem, Y)
     
    return Si, param_values[np.argwhere(Y == Y.max)]

