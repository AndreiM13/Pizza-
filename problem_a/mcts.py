from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
import math
import numpy as np

class MCTS:

    def __init__(self, exploration_weight=1, env=None):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.env = env

    def score(self, n):
        if self.N[n] == 0:
            return float("-inf")  # avoid unseen moves
        return self.Q[n] / self.N[n]  # average reward

    def print_nodes(self):
        agg = []
        score = []

        for n,c in self.children.items():
            agg.append(str(list(c)[0].point) if len(c) > 0 else 'Remaining')
            score.append(self.score(n))
        return "-".join(agg), score
    
    def choose(self, node):
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        return max(self.children[node], key=self.score), np.max(list(map(self.score, self.children[node])))

    def do_rollout(self, node):
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            # generate a new child node to explore
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        while True:
            if node.is_terminal():
                reward = node.reward()
                return reward
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        for ii,node in enumerate(reversed(path)):
            self.N[node] += 1
            self.Q[node] += reward
            reward /= 1.01 # 1% of the rewards are always reduced, discounted returns

    def _uct_select(self, node):
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        
        log_N_vertex = math.log(self.N[node])
        
        def uct(item):
            return self.Q[item] / self.N[item] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[item]
            )
        
        return max(self.children[node], key=uct)