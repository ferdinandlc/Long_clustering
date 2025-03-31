import os
import sys
import numpy as np
import pandas as pd
import json


path = os.getcwd()
#os.chdir(path)

from clustering.analyzer import long_clustering

config_names = ['newcom25', 'mergecom25', 'chgnode25']

def data_load(config_name, print_assign = False):
    
    with open(path+'/data/json_config/config_'+config_name+'.json', 'r') as f:
        config_data = json.load(f)
    
    dic_clus = config_data['dic_clus']
    
    for key in dic_clus.keys():
      if dic_clus[key][1] != []:
        map_int = np.apply_along_axis(lambda X: [int(x) for x in X], 1, dic_clus[key][1])
        dic_clus[key][1] = map_int
    
    lst_naive = []
    
    for clus_name in dic_clus.keys():
      cluster_labels = np.load(path+'/data/clustering/'+config_name+'/'+clus_name)
      lst_clusters = []
      if dic_clus[clus_name][2] != []:
        lst_ind = dic_clus[clus_name][2]
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
      if print_assign:
        print(clus_name)
      
      new_d = long_clustering(lst_clusters, dic_clus[clus_name][1])
    
      dic_clus[clus_name].append(new_d)
    
    commu = [np.load(path+'/data/label/'+config_name+'/node_communities_{}.npy'.format(i)) for i in range(25)]
    raw_labels = [[np.load(path+'/data/label/'+config_name+'/perturbations_{}.npy'.format(i)),
                  np.load(path+'/data/label/'+config_name+'/dyn_change_nodes_{}.npy'.format(i))] for i in range(25)]
    
    labels = []
    already_seen = []
    for i in range(25):
        l = []
        for node in range(500):
            if node in raw_labels[i][0]:
                l.append('perturbation')
            elif node in raw_labels[i][1]:
                l.append('dynchange')
            elif node in already_seen:
                l.append('perturbation')
            else:
                l.append('none')
        for node in list(raw_labels[i][0]) + list(raw_labels[i][1]):
            if node not in already_seen:
                already_seen.append(node)
        labels.append(l)
    
    Lref = []
    for node in range(500):
      for step in range(25):
        Lref.append(labels[step][node])
        
    return dic_clus, commu, labels, Lref

