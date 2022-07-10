import numpy as np
from copy import deepcopy

class History:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def copy(self):

        history = __class__()

        history.states = deepcopy(self.states)
        history.actions = deepcopy(self.actions)
        history.rewards = deepcopy(self.rewards)
        history.state_history = deepcopy(self.state_history)
        history.action_history = deepcopy(self.action_history)
        history.reward_history = deepcopy(self.reward_history)

        return history

    def update(self, state=None, action=None, reward=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def update_history(self):
        self.state_history.append(self.states.copy())
        self.action_history.append(self.actions.copy())
        self.reward_history.append(self.rewards.copy())

    def initialize(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def plot(self, show=True):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        y = [_[-1] for _ in self.reward_history]
        x = np.arange(len(y))
        ax.scatter(x, y, marker="o", linewidth=None)
        if show:
            plt.show()

        return fig, ax
