import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from numbers import Real
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from IPython.display import HTML
from thermo import Joback
softmax = lambda x:np.exp(x)/np.nansum(np.exp(x))

def main():
    n = 1000
    edges = [
        ('s', 'C'), 
        ('s', 'c1cc(ccc1)'), 
        ('s', 'O'),
        ('C', 'C'), 
        ('C', 'c1cc(ccc1)'), 
        ('C', 'O'),
        ('C', 'g'),
        ('c1cc(ccc1)', 'c1cc(ccc1)'), 
        ('c1cc(ccc1)', 'C'), 
        ('c1cc(ccc1)', 'O'), 
        ('c1cc(ccc1)', 'g'), 
        ('O', 'C'), 
        ('O', 'c1cc(ccc1)'), 
        ('O', 'O'), 
        ('O', 'g'), 
        ]
    game = Game(edges)
    nx.draw_networkx(game.g, with_labels=True)
    nx.nx_agraph.view_pygraphviz(game.g, prog='fdp') 
    plt.show()
    for i in tqdm(range(n)):
        game.initialize()
        game.play()
        print(game.reward, game.smiles, game.state_action_history)
        game.update()
    print(game.state_action_history)
    for node in game.g.nodes:
        print(node, game.g.nodes[node]['p'])
    print(game.state_action_history)
    plt.plot(game.reward_history, marker='o')
    plt.show()

class Game():
    def __init__(self, edges=None):

        self.current_node = 's'
        self.smiles = ''
        self.state_action_history = [[self.current_node, np.nan]]
        self.reward = None 
        self.reward_history = []

        if edges is not None:
            self.g = nx.DiGraph()
            self.g.add_edges_from(edges)

            successors = [list(self.g.successors(n)) for n in self.g.nodes]
            self.params_state_action = dict()
            for n, _ in zip(self.g.nodes, successors):
                self.g.nodes[n]['params'] = dict(list(zip(_, np.ones(len(_)))))
                self.g.nodes[n]['p'] = dict(list(zip(_, softmax(np.ones(len(_))))))

    def evaluate(self, property='Tb', scale=1.0):
        nodes = [_[1] for _ in self.state_action_history] ## Sequence of nodes wo 'start' and 'goal'.
        nodes, next_node, _ = nodes[:-2], nodes[-2], nodes[-1]
        smiles = ''.join(nodes)
        try:
            estimator = Joback(smiles)
            props = estimator.estimate()
            prop = props.get(property)
        except ValueError as p:
            print(p)
            prop = 0.0
 
        if isinstance(scale, Real):
            reward = scale * prop
        else:
            reward = (reward - scale[0])/(scale[1]-scale[0])
        return reward

    def next(self, node, maxstep=5, max_mass=1000):
        if len(self.state_action_history) > maxstep:
            return 'g'

        mol = Chem.MolFromSmiles(self.smiles)
        mass = sum([_.GetMass() for _ in mol.GetAtoms()])
        if mass > max_mass:
            return 'g'

        probs = self.g.nodes[node]['p']
        if len(probs)==0:
            return None
        next_nodes = list(probs.keys())
        next_probs = [probs[n] for n in next_nodes]
        return np.random.choice(next_nodes, p=next_probs)
 
    def initialize(self):
        self.current_node = 's'
        self.smiles = ''
        self.state_action_history = [[self.current_node, np.nan]]
        self.reward = None 

    def update(self, learning_rate=0.001):
        lr = learning_rate
        sa = np.array(self.state_action_history) 
        num_steps = len(sa)
        delta = dict()
        for node in self.g.nodes:
            prob = self.g.nodes[node]['p']
            for _next, _prob in prob.items():
                _sa = [_ for _ in sa if (_[0], _[1]) == (node, _next)]
                _s = [_ for _ in sa if _[0] == node] 
                count_sa = len(_sa)
                count_s = len(_s)
                delta[(node, _next)] = lr * self.reward * (count_sa - _prob * count_s)/num_steps
            
        for node in self.g.nodes:
            prob = self.g.nodes[node]['params']
            for _next, _param in prob.items():
                self.g.nodes[node]['params'][_next] = _param + delta[(node, _next)]
            _successors = prob.keys()
            _params = [self.g.nodes[node]['params'][_] for _ in _successors]
            self.g.nodes[node]['p']  = dict(list(zip(_successors, softmax(_params))))
    
    def play(self):
        while True:
            n = self.next(self.current_node)
            self.state_action_history[-1][1] = n
            self.state_action_history.append([n, np.nan])
            if len(list(self.g.successors(n)))==0:
                reward = self.evaluate()
                self.reward = reward
                self.reward_history.append(self.reward)
                break 
            else:
                self.smiles += n
                self.current_node = n
main()