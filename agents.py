import numpy as np
from regex import W
from .mcts import Node
from .state import State
from .history import History


def softmax(x):
    return np.exp(x) / np.nansum(np.exp(x))


class BaseAgent(object):
    def __init__(self, game=None, state=None, history=None, max_step=5):
        self.game = game
        self.history = history
        self.state = state
        self.max_step = max_step

        if self.history is None:
            self.history = History()
        if self.state is None and self.game is not None:
            self.state = State(
                game=self.game,
                initial_state=self.game.initial_state,
                final_state=self.game.final_state,
            )

    def copy(self):
        agent = __class__()
        agent.game = self.game
        agent.state = self.state.copy()
        agent.history = self.history.copy()
        agent.max_step = self.max_step
        return agent

    def update(self, state, action=None, next=None, reward=None):
        self.history.update(state, action, reward)
        self.state.update(next)

    def play(self, new_game=True):
        if new_game:
            self.game.initialize()
            self.state.initialize()
            self.history.initialize()

        while True:
            state = self.state.current_state
            if self.state.is_done():
                reward = self.evaluate(state, action=None, next=None)
                self.update(state, action=None, next=None, reward=reward)
                self.history.update_history()
                break
            action = self.action()
            next = self.game.next(self.state.current_state, action)
            reward = self.evaluate(state, action, next)
            self.update(state, action, next, reward)

        return reward

    def evaluate(self, state=None, action=None, next=None):
        if self.state.is_done():
            state_history = self.history.states + [self.state.current_state]
            score = self.game.get_score(state_history)
            return score
        else:
            return 0.0

    def action(self):
        return None

    def next(self, state=None, action=None):
        return self.game.next(state, action)


class RandomAgent(BaseAgent):
    def action(self):
        actions = self.game.legal_actions(self.state.current_state)
        if len(actions) == 0:
            return None
        idx = np.random.choice(len(actions))
        return actions[idx]


class MCTSAgent(BaseAgent):
    def __init__(self, game=None, state=None, history=None, max_step=5):
        super().__init__(game=game, state=state, history=history, max_step=max_step)
        self.node = None
        
    def action(self, n=100):
        state = self.state.current_state
        if self.node is None:
            self.node = Node(state=self.state)

        if self.node.child_nodes is None:
            legal_actions = self.state.legal_actions()
            legal_states = [_[1] for _ in legal_actions]
            self.node.expand(legal_states)
        else:
            legal_actions = [_.state.current_state for _ in self.node.child_nodes]
        
        for _ in range(n):
            self.node.evaluate()

        n_list = []
        for c in self.node.child_nodes:
            n_list.append(c.n)
        return legal_actions[np.argmax(n_list)]

    def update(self, state, action, next, reward):
        self.history.update(state, action, reward)
        self.state.initialize()
        if next is None:
            self.node = self.node.get_root_node()
        else:
            self.node = self.node.get_child_node(next)

    def playout(self):
        random_agent = RandomAgent(
            game=self.game,
            state=self.state.copy(),
            history=self.history.copy(),
            max_step=self.max_step,
        )
        score = random_agent.play(new_game=False)
        return score

    def random_action(self, state):
        actions = self.game.legal_actions(state)
        if len(actions) == 0:
            return None
        idx = np.random.choice(len(actions))
        return actions[idx]

    # %%
    def argmax(self, collection, key=None):
        return collection.index(max(collection))


class MinMaxAgent(BaseAgent):
    def action(self, state):
        best_action = 0
        best_score = -float("inf")
        strings = ["", ""]
        for action in state.legal_actions():
            score = -self.mini_max_plus(state.next(action), -best_score)
            if score > best_score:
                best_action = action
                best_score = score

            strings[0] = f"{strings[0]}{action:2d},"
            strings[1] = f"{strings[1]}{score:2d},"
        #  print(f'action: {strings[0]}\nscore: {strings[1]}\n')
        return best_action

    def mini_max_plus(self, state, limit):
        if state.is_lose():
            return -1

        if state.is_draw():
            return 0

        best_score = -float("inf")
        for action in state.legal_actions():
            score = -self.mini_max_plus(state.next(action), -best_score)
            if score > best_score:
                best_score = score

            if best_score >= limit:
                return best_score

        return best_score


class AlphaBetaAgent(BaseAgent):

    # %%
    def alpha_beta(self, state, alpha, beta):
        if state.is_lose():
            return -1
        if state.is_draw():
            return 0

        for action in state.legal_actions():
            score = -self.alpha_beta(state.next(action), -beta, -alpha)
            if score > alpha:
                alpha = score
            if alpha >= beta:
                return alpha
        return alpha

    def alpha_beta_action(self, state):
        best_action = 0
        alpha = -float("inf")
        strings = ["", ""]
        for action in state.legal_actions():
            score = -self.alpha_beta(state.next(action), -float("inf"), -alpha)
            if score > alpha:
                best_action = action
                alpha = score
            strings[0] = f"{strings[0]}{action:2d}"
            strings[1] = f"{strings[1]}{score:2d}"
        #  print(f'action: {strings[0]},\nscore: {strings[1]}\n')
        return best_action


#   # %%
#   %%time
#     EP_GAME_COUNT=100

#     def first_player_point(self, ended_state):
#         if ended_state.is_lose():
#             return 0 if ended_state.is_first_player() else 1
#         return 0.5

#     def play(self, next_actions):
#         state = State()

#         while True:
#             if state.is_done():
#                 break
#             next_action = next_actions[0] if state.is_first_player() else next_actions[1]
#             action = next_action(state)

#             state = state.next(action)

#       return first_player_point(state)

#     def evaluate_algorithm_of(label, next_actions):

#         total_point=0
#         for i in range(EP_GAME_COUNT):
#           if i  % 2 == 0:
#             total_point += play(next_actions)
#           else:
#             total_point += 1 - play(list(reversed(next_actions)))

#         print(f'\rEvaluate {i+1}/{EP_GAME_COUNT}', end=' ')
#         print('')

#         average_point = total_point / EP_GAME_COUNT
#         print(label.format(average_point))

#       next_actions = (mcts_action, random_action)
#       %time evaluate_algorithm_of('VS_Random {:.3f}', next_actions)
#       print()

#       next_actions = (mcts_action, mini_max_action)
#       %time evaluate_algorithm_of('VS_MiniMax {:.3f}', next_actions)
#       print()

#       next_actions = (mcts_action, alpha_beta_action)
#       %time evaluate_algorithm_of('VS_AlphaBeta {:.3f}', next_actions)
#       print()

#   # %%
#   next_actions = (mcts_action, alpha_beta_action)

#   for EP_GAME_COUNT in 5, 10, 15, 20, 25, 50:
#       %time evaluate_algorithm_of('VS_AlphaBeta {:.3f}', next_actions)
#       print()

#   # %%

