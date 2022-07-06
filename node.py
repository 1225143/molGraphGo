import math
import numpy as np

from evaluator import JoabckEvaluator as Evaluator
# %%
class Node:
    def __init__(self, state):
        self.state = state
        self.w = 0
        self.n = 0
        self.child_nodes = None
        self.evaluator = Evaluator()
        self.smiles = None
  
    def evaluate(self, num_expand=1):
        if self.state.is_done():
            value = self.evaluator.evaluate(self.smiles)
            self.w += value
            self.n += 1
            return value

        if not self.child_nodes:
            value = self.playout(self.state)

            self.w += value
            self.n += 1
            if self.n == num_expand:
              self.expand()
            return value
        else:
            value = -self.next_child_node().evaluate()

            self.w += value
            self.n += 1
            return value
        
    def expand(self):
        legal_actions = self.state.legal_actions()
        self.child_nodes = []
        for action in legal_actions:
            self.child_nodes.append(Node(self.state.next(action)))
      
    def next_child_node(self):

        for child_node in self.child_nodes:
            if child_node.n == 0:
                return child_node

        t = 0
        for c in self.child_nodes:
            t += c.n
        ucb1_values = []
        for child_node in self.child_nodes:
            ucb1_values.append(-child_node.w/child_node.n + (2*math.log(t)/child_node.n)**0.5)
          
        return self.child_nodes[np.argmax(ucb1_values)]

    def playout():
        pass
  # %%
