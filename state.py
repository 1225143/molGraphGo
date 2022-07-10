
# %%
from regex import W
from numbers import Real
from copy import deepcopy
class State():

    def __init__(self, current_state=None, game=None, initial_state=None, final_state=None, history=None):
        self.current_state = current_state 
        self.history = history or [] 
        self.game = game
        if initial_state or game is None:
            self.initial_state = initial_state
        else:
            self.initial_state = self.game.initial_state 
        if final_state or game is None:
            self.final_state = final_state
        else:
            self.final_state = self.game.final_state 

    def legal_actions(self):
        return self.game.legal_actions(self.current_state)

    def next(self, action):
        return self.game.next(action=action)

    def copy(self):
        state = __class__(game=self.game, initial_state=self.initial_state, final_state=self.final_state)
        state.current_state = self.current_state
        if self.history is not None:
            state.history = deepcopy(self.history)
        return state

    def initialize(self):
        self.current_state = self.initial_state
        self.history = [self.initial_state]

    def is_initial(self):
        return self.current_state == self.initial_state
    def is_done(self):
        return self.current_state == self.final_state 

    def update(self, next):
        if next is None:
            self.initialize()
        else:
            self.current_state = next
            self.history.append(next)
        return self

    def evaluate(self):
        return self.game.get_score(self.history)