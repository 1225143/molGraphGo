import math
import numpy as np
from regex import W
from .state import State
from .evaluator import JobackEvaluator as Evaluator
# %%
class Node:
    def __init__(self, state=None, parent=None):
        self.state = state
        self.parent = parent 

        self.w = 0
        self.n = 0
        self.child_nodes = None
    def __str__(self):
        return f'{self.state} {str(self)}'

    def evaluate(self, num_expand=1):
        if self.state.is_done():
            state_history = []
            _node = self
            while True:
                state_history.append(_node.state.current_state)
                if _node.parent is None:
                    break
                _node = _node.parent
            state_history = state_history[::-1]
            score = self.state.game.get_score(state_history)
            self.w += score
            self.n += 1
            return score

        if not self.child_nodes:
            score = self.playout(self.state.copy())  # Play the game with random agent.
            self.w += score
            self.n += 1
            if self.n == num_expand:
                legal_states = [_[1] for _ in self.state.legal_actions()]
                self.expand(legal_states)
            return score
        else:
            score = self.next_child_node().evaluate()
            self.w += score
            self.n += 1
            return score

    def get_child_node(self, state):
        for _node in self.child_nodes:
            if _node.state.current_state == state:
                return _node
        else:
            return None
        
    def expand(self, legal_states):
        if self.child_nodes:
            return self.child_nodes
        self.child_nodes = []
        for _state in legal_states:
            state = self.state.copy()
            state.update(_state)
            self.child_nodes.append(Node(state=state, parent=self))
      
    def next_child_node(self):

        for child_node in self.child_nodes:
            if child_node.n == 0:
                return child_node

        t = 0
        for c in self.child_nodes:
            t += c.n
        ucb1_values = []
        for child_node in self.child_nodes:
            ucb1_values.append(child_node.w/child_node.n + (2*child_node.w*math.log(t)/child_node.n)**0.5)
          
        return self.child_nodes[np.argmin(ucb1_values)]

    def playout(self, state):
        if state.is_done():
            return state.evaluate()
        legal_states = [_[1] for _ in state.legal_actions()]
        action = np.random.choice(legal_states)
        next = state.next(action)
        return self.playout(state.update(next))
    
    def get_root_node(self):
        if self.state.is_initial:
            return self 
        return self.parent.get_root_node()