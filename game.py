import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from rdkit import Chem
from rdkit.Chem import AllChem
from regex import W
from IPython.display import HTML
from itertools import chain
from thermo import Joback

def softmax(x):
    return np.exp(x)/np.nansum(np.exp(x))

class BaseGame():
    def __init__(self):
        pass

    def initialize(self):
        pass

    def next(self, state=None, action=None):
        return None 

    def legal_actions(self, state=None):
        return []

    def update(self):
        pass

    def evaluate(self, state_history=None):
        return 0.0
    
class MolGraphGame(BaseGame):
    def __init__(self, edges=None):
        self.final_state = 'goal'
        self.initial_state = 'start'
        self.max_mass = 1000
        self.max_step = 10
        self.current_state = self.initial_state
        self.smiles = ''
        self.reward = None 

        self.state_action_history = [[self.current_state, np.nan]]
        self.reward_history = []


        if edges is not None: 
            self.g = nx.DiGraph()
            self.g.add_edges_from(edges, weight=1.0)
            if not (self.initial_state in self.g and self.final_state in self.g):
                msg = f"'{self.initial_state}' and '{self.final_state}' are required for the molecular fragment graph."
                raise ValueError(msg)
 
    def initialize(self):
        self.current_state = self.initial_state
        self.smiles = ''
        self.reward = None

    def get_mass(self, add_Hs=True):

        mol = Chem.MolFromSmiles(self.smiles)
        if add_Hs:
            mol = AllChem.AddHs(mol)
        mass = sum([_.GetMass() for _ in mol.GetAtoms()])
        return mass

    def legal_actions(self, state):
    
        if len(self.state_action_history) > self.max_step:
            return [self.final_state]
    
        if self.get_mass() > self.max_mass:
            return [self.final_state]

        return list(self.g.edges(state))

    def next(self, state=None, action=None):
        # Deterministic transition. The action is always accepted.
        if action is None:
            return None
        if isinstance(action, str):
            return action
        else:
            return action[-1]
 
    def get_score(self, state_history, property='Tb'):
        
        if state_history[-1] != self.final_state:
            return 0.0
        
        smiles = ''.join(state_history[1:-1])
        try:
            estimator = Joback(smiles)
            props = estimator.estimate()
            return props.get(property)
        except ValueError as p:
            print(p)
            return 0.0


    @property
    def playcount(self):
        return len(self.reward_history)

