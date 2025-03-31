import os
import sys
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import jaccard_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_mutual_info_score


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

def jaccard_similarity_map(list1, list2, t1, t2, full_map):
  l_1 = [full_map[t1][node] for node in list1]
  l_2 = [full_map[t2][node] for node in list2]
  intersection = len(list(set(l_1).intersection(l_2)))
  union = (len(set(l_1)) + len(set(l_2))) - intersection
  return float(intersection) / union

def extract_t_c(node_name):
  index = node_name.index('-')
  t = int(node_name[1:index])
  com = int(node_name[index+1:len(node_name)])

  return t, com


def build_union(lst_labels, com, treshold = 1, timeshift = 0):

  l_union = set()
  lst_com = list(com)

  if treshold == 1:
    for commu in lst_com:
      t, c = extract_t_c(commu)
      l_union = l_union.union(set(lst_labels[t-timeshift][c]))

  else:
    dic_union = {}
    for commu in lst_com:
      t, c = extract_t_c(commu)
      for node in lst_labels[t-timeshift][c]:
        if node not in dic_union:
          dic_union[node] = 1
        else:
          dic_union[node] += 1

    l_union = set([k for k,v in dic_union.items() if v>treshold])


  return list(l_union)

def build_commu(lst_clusters, l_com, min_instances = 2):

  l_glob_com = []
  nodes = set()
  dic_nodes_commu = {}
  for t in range(len(lst_clusters)):
    for c in lst_clusters[t].keys():
      nodes = nodes.union(set(lst_clusters[t][c]))
  nodes = list(nodes)
  for node in nodes:
    dic_nodes_commu[node] = [-1 for t in range(len(lst_clusters))]

  id_com = 0
  for com in l_com:
    if len(com) >= min_instances:
      lst_com=list(com)
      for commu in lst_com:
        t, c = extract_t_c(commu)
        for node in lst_clusters[t][c]:
          dic_nodes_commu[node][t] = id_com
      id_com += 1
      l_glob_com.append(com)

  return dic_nodes_commu, l_glob_com

def simple_aggregate_commu(dic_nodes_commu, print_assign = False):
  num_assigned = 0
  num_corrected = 0
  num_unknown = 0
  num_last_case = 0
  new_dic = {}
  for node in dic_nodes_commu:
    new_l = []
    for t in range(len(dic_nodes_commu[node])):
      if dic_nodes_commu[node][t] != -1:
        new_l.append(dic_nodes_commu[node][t])
        num_assigned += 1
      elif t<len(dic_nodes_commu[node])-1:
        if dic_nodes_commu[node][t+1] != -1:
            new_l.append(dic_nodes_commu[node][t+1])
            num_corrected += 1
        elif t>0:
          if new_l[t-1] != -1:
            new_l.append(new_l[t-1])
            num_corrected += 1
          else:
            new_l.append(-1)
            num_unknown += 1
        else:
          new_l.append(-1)
          num_unknown += 1
      else:
        new_l.append(new_l[t-1])
        num_last_case += 1
    new_dic[node] = new_l

  if print_assign:
    print('{} nodes were assigned, {} corrected, {} unlabelled and {} last case'.format(num_assigned, num_corrected, num_unknown, num_last_case))
  return new_dic





def long_clustering(lst_clusters, full_map, similarity_treshold = 0.15, min_instance = 2, t_periods = 2):

  G = nx.Graph()
  if len(full_map) > 0:
    need_map = True
  else:
    need_map = False

  for y in range(len(lst_clusters)):
    if -1 in lst_clusters[y].keys():
      for i in range(len(lst_clusters[y])-1):
        G.add_node('C{}-{}'.format(y,i), timestep = y)
      else:
        for i in range(len(lst_clusters[y])):
          G.add_node('C{}-{}'.format(y,i), timestep = y)


  for y in range(len(lst_clusters)-1):
    if -1 in lst_clusters[y].keys():
      nb_clus = len(lst_clusters[y])-1
    else:
      nb_clus = len(lst_clusters[y])
    for i in range(nb_clus):

      for t in range(min(t_periods,len(lst_clusters)-y-1)):
        if -1 in lst_clusters[y+1+t].keys():
          nb_clus_t = len(lst_clusters[y+1+t])-1
        else:
          nb_clus_t = len(lst_clusters[y+1+t])
        for j in range(nb_clus_t):

          if need_map == True:
            similarity = jaccard_similarity_map(lst_clusters[y][i],lst_clusters[y+1+t][j], y, y+1+t, full_map)
          else:
            similarity = jaccard_similarity(lst_clusters[y][i],lst_clusters[y+1+t][j])
          if similarity > similarity_treshold:
            G.add_edge('C{}-{}'.format(y,i),'C{}-{}'.format(y+1+t,j), weight =similarity)

  l_com = nx.community.greedy_modularity_communities(G)

  d, l = build_commu(lst_clusters, l_com, min_instances = min_instance)
  new_d = simple_aggregate_commu(d)
  return new_d