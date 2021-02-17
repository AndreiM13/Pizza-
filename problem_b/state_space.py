from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
import numpy as np
from mcts import MCTS
from pizza_requirements import *
import argparse
import re
import pandas
import os

df = pandas.DataFrame(columns=['method', 'value', 'point', 'cluster', 'parent', 'parent_method'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--fuzzy", type=bool, required=True, default=1)
    return parser.parse_args()

args = parse_args()

def run_mcts(clusters, MAX_CLUSTERS=5, action_count=5):
    
    global args
    _HCC = namedtuple("HierarchicalCluster", "cluster point value method terminal")
    file = open(args.file, 'r')

    def find_rows():
        contents = file.read()
        lines = re.split(r"\n", contents)
        rows = []
        for l in lines:
            rows.append(l.split(" "))
        
        return rows

    rows = find_rows()

    r = rows[1:]
    # requirements
    d = []
    for i in r:
        d.append(i[1:])

    # find the hierarchy for such a solution
    class HierarchicalCluster(_HCC, ABC):

        # @staticmethod
        # def has_children(cluster, n):
        #     cluster_idxs = np.where(clusters==cluster)[0]
        #     return n == np.min(noise[cluster_idxs])
        
        @staticmethod
        def last_child():
            return HierarchicalCluster(cluster=5, point=4, value=5, terminal=True)
        
        def find_children(self):
            global df, args
            counts = self.find_child()
            if self.cluster == MAX_CLUSTERS:
                return {}
            cluster_idx = np.where(clusters==(self.cluster+1))[0][0]
            methods = ['cluster_score', 'fuzzy_intersection']
            df = df.append({"method": 'cluster_score', "point": counts[0], "value": counts[1], "cluster": clusters[cluster_idx], 'parent': self.cluster, 'parent_method': self.method}, ignore_index=True)
            if(args.fuzzy == 1):
                df = df.append({"method": 'fuzzy_intersection', "point": counts[0], "value": counts[2], "cluster": clusters[cluster_idx], 'parent': self.cluster, 'parent_method': self.method}, ignore_index=True)
            l = range(1,3) if args.fuzzy == 1 else range(1,2)
            return {
                HierarchicalCluster(cluster=clusters[cluster_idx], point=counts[0], value=counts[idx], method=methods[idx-1],
                    terminal=HierarchicalCluster.set_terminal(
                        clusters[cluster_idx])) for idx in l
            }

        def find_random_child(self):
            global df, args
            counts = self.find_child()
            end = 2 if args.fuzzy == 1 else 1
            m = np.random.choice([1,end])
            method = 'cluster_score' if (m == 1 and args.fuzzy == 0) else 'fuzzy_intersection'
            cluster_no = np.random.choice(list(set(np.arange(self.cluster,action_count+1,1).tolist()) - set([self.cluster])))
            # df = df.append({"method": method, "point": counts[0], "value": counts[m], "cluster": cluster_no, 'parent': self.cluster, 'mode': 'random_child'}, ignore_index=True)
            return HierarchicalCluster(cluster=cluster_no, point=counts[0], value=counts[m], method=method,
            terminal=HierarchicalCluster.set_terminal(cluster_no))

        @staticmethod
        def set_terminal(cluster):
            return (cluster == MAX_CLUSTERS)
        
        def is_terminal(self):
            return self.terminal

        def find_child(self):
            # pizzas
            p = d
            counts = []
            # ing = list(ingredients.values())
            
            # map requirements to pizza
            i = d[self.cluster-1]
            individual_counts = []
            def intersection(arr1, arr2):
                arr3 = [v1 for v1 in arr1 for v2 in arr2 if v1 == v2]
                return arr3
            
            ic = []
            for ii,j in enumerate(p):
                # calculate count
                individual_counts.append((len(j) + (int(len(intersection(i, j)) == len(i)))+1)*1e-2)
                ic.append(int(len(intersection(i, j)) == len(i)))

            # implementing fuzzy logic
            d_values = dict(zip(list(range(1,len(d))),[0]*(len(d)-1)))
            fuzzy_intersection = 0
            for ij in list(set(list(range(0,len(d)))) - set([self.cluster-1])):
                if ij not in d_values:
                    d_values[ij] = 0
                d_values[ij] += len(intersection(i, d[ij]))

            fuzzy_intersection = sum(list(d_values.values()))
            
            # index-based, value-based
            # exact match
            counts = (np.argmax(ic), sum(individual_counts), fuzzy_intersection*1e-2)

            return counts
        
        # floating point number
        def reward(self):
            return self.value
        
    return HierarchicalCluster

def cluster_model(MAX_CLUSTERS):
    def non_deterministic_hierarchical_clustering(n_clusters=MAX_CLUSTERS):
        clusters = list(range(1,MAX_CLUSTERS+1))
        
        return clusters
    
    def the_model(clusters, MAX_CLUSTERS, env=None):
        node = run_mcts(clusters, MAX_CLUSTERS)(cluster=1, point=0, value=0, method='cluster_score', terminal=False)
        mcts = MCTS(env=env)
        global df
        
        while True:
            for i in range(25):
                mcts.do_rollout(node)
                # print(mcts.print_nodes())
            
            node, score = mcts.choose(node)
            if node.terminal:
                print(mcts.print_nodes(df, df_export=True), score)
                break
        
        state_selected = np.where((clusters == node.cluster))[0][0]
        return state_selected, score
    
    return non_deterministic_hierarchical_clustering, the_model

if __name__ == "__main__":

    if os.path.exists("output.csv"):
        os.unlink("output.csv")

    clusters, the_model = cluster_model(5)
    the_model(np.array(clusters()), 5)