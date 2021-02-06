from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
import numpy as np
from mcts import MCTS
from pizza_requirements import *
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    return parser.parse_args()

def run_mcts(clusters, scaler, MAX_CLUSTERS=5, action_count=5):
    
    args = parse_args()
    _HCC = namedtuple("HierarchicalCluster", "cluster point value terminal")
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
            counts = self.find_child()
            if self.cluster == MAX_CLUSTERS:
                return {}
            cluster_idx = np.where(clusters==(self.cluster+1))[0][0]
            return {
                HierarchicalCluster(cluster=clusters[cluster_idx], point=counts[0], value=counts[idx],
                    terminal=HierarchicalCluster.set_terminal(
                        clusters[cluster_idx])) for idx in range(1,3)
            }

        def find_random_child(self):
            counts = self.find_child()
            cluster_no = np.random.choice(list(set(np.arange(1,action_count+1,1).tolist()) - set([self.cluster])))
            return HierarchicalCluster(cluster=cluster_no, point=counts[0], value=counts[1], 
            terminal=HierarchicalCluster.set_terminal(cluster_no))

        @staticmethod
        def set_terminal(cluster):
            return (cluster == MAX_CLUSTERS)
        
        def is_terminal(self):
            return self.terminal

        def find_child(self):
            # pizzas
            p = list(pizza_requirements.values())
            counts = []
            ing = list(ingredients.values())
            
            # map requirements to pizza
            i = d[self.cluster-1]
            individual_counts = []
            def intersection(arr1, arr2):
                arr3 = [v1 for v1 in arr1 for v2 in arr2 if v1 == v2]
                return arr3
            
            ic = []
            for ii,j in enumerate(p):
                # calculate count
                individual_counts.append((len(j) * sum(ing[ii]) + (int(len(intersection(i, j)) == len(i))+1)))
                ic.append(int(len(intersection(i, j)) == len(i)))

            # implementing fuzzy logic
            d_values = dict(zip(list(range(1,len(d))),[0]*(len(d)-1)))
            fuzzy_intersection = 0
            for ij in range(self.cluster,len(d)):
                if ij not in d_values:
                    d_values[ij] = 0
                d_values[ij] += len(intersection(i, d[ij]))

            fuzzy_intersection += (d_values[ii] if ii in d_values else 0);
            
            # index-based, value-based
            # exact match
            counts = (np.argmax(ic), sum(individual_counts) * 1e-2, fuzzy_intersection * 1e-2)

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
        node = run_mcts(clusters, MAX_CLUSTERS)(cluster=1, point=0, value=0, terminal=False)
        mcts = MCTS(env=env)
        
        while True:
            for i in range(25):
                mcts.do_rollout(node)
                # print(mcts.print_nodes())
            
            node, score = mcts.choose(node)
            if node.terminal:
                print(mcts.print_nodes())
                break
        
        state_selected = np.where((clusters == node.cluster))[0][0]
        return state_selected, score
    
    return non_deterministic_hierarchical_clustering, the_model

if __name__ == "__main__":

    clusters, the_model = cluster_model(5)
    the_model(np.array(clusters()), 5)