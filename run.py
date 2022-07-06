# %%
import os
from platform import node
import sys
import time
from numba import jit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx
import math
import random
import graphviz

# 
from mcts import *
from node import Node
from state import State
from evaluator import ThermoEvaluator as Evaluator

from contextlib import contextmanager

#%%
@contextmanager
def timer():
    t = time.perf_counter()
    yield None
    print('Elapsed:', time.perf_counter() - t)

# %%
g = nx.Graph()
nodes = ['start', 'C', 'O', 'N', 'c1ccc(cc1)', 'end']
edges = list(combinations(nodes, 2))
print(edges)
g.add_nodes_from(nodes)
g.add_edges_from(edges)
view = nx.draw_networkx(g)

mcts = MCTS(g, max_step=5)
mcts.run()
