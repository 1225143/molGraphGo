from .node import Node
from math import random

def MCTS():
    def __init__(self, game, max_step=5):
        self.game = game
        self.max_step = max_step
  
    def run():
        self.mcts_action()
  
    def mcts_action(self, state, n=100):
    
        root_node = Node(state)
        root_node.expand()

        for _ in range(n):
            root_node.evaluate()
          
        legal_actions = state.legal_actions()
        n_list = []
        for c in root_node.child_nodes:
            n_list.append(c.n)
        return legal_actions[argmax(n_list)]


    def playout(self, state):
        if state.is_lose():
            return -1
        if state.is_draw():
            return 0
        return -playout(state.next(random_action(state)))

    # %%
    def argmax(self, collection, key=None):
        return collection.index(max(collection))

    # %%
    def random_action(self, state):
        legal_actions = state.legal_actions()
        return legal_actions[random.randint(0, len(legal_actions)-1)]

  # %%
    def mini_max_action(self, state):
        best_action = 0
        best_score = -float('inf')
        strings = ['', '']
        for action in state.legal_actions():
            score = -mini_max_plus(state.next(action), -best_score)
            if score > best_score:
                best_action = action
                best_score = score
              
            strings[0] = f'{strings[0]}{action:2d},'
            strings[1] = f'{strings[1]}{score:2d},'
        #  print(f'action: {strings[0]}\nscore: {strings[1]}\n')
        return best_action

    def mini_max_plus(self, state, limit):
        if state.is_lose():
            return -1
        
        if state.is_draw():
            return 0
        
        best_score = -float('inf')
        for action in state.legal_actions():
            score = -mini_max_plus(state.next(action), -best_score)
            if score > best_score:
                best_score = score

            if best_score >= limit:
                return best_score
          
        return best_score


  # %%
    def alpha_beta(self, state, alpha, beta):
        if state.is_lose():
            return -1
        if state.is_draw():
            return 0
      
        for action in state.legal_actions():
            score = -alpha_beta(state.next(action), -beta, -alpha)
            if score > alpha:
                alpha = score
            if alpha >= beta:
                return alpha
        return alpha

    def alpha_beta_action(self, state):
        best_action = 0
        alpha = -float('inf')
        strings = ['', '']
        for action in state.legal_actions():
            score = -alpha_beta(state.next(action), -float('inf'), -alpha)
            if score > alpha:
                best_action = action
                alpha = score
            strings[0] = f'{strings[0]}{action:2d}'
            strings[1] = f'{strings[1]}{score:2d}'
        #  print(f'action: {strings[0]},\nscore: {strings[1]}\n')
        return best_action

  # %%
  %%time
    EP_GAME_COUNT=100

    def first_player_point(self, ended_state):
        if ended_state.is_lose():
            return 0 if ended_state.is_first_player() else 1
        return 0.5

    def play(self, next_actions):
        state = State()

        while True:
            if state.is_done():
                break
            next_action = next_actions[0] if state.is_first_player() else next_actions[1]
            action = next_action(state)

            state = state.next(action)

      return first_player_point(state)

    def evaluate_algorithm_of(label, next_actions):

        total_point=0
        for i in range(EP_GAME_COUNT):
          if i  % 2 == 0:
            total_point += play(next_actions)
          else:
            total_point += 1 - play(list(reversed(next_actions)))
          
        print(f'\rEvaluate {i+1}/{EP_GAME_COUNT}', end=' ')
        print('')

        average_point = total_point / EP_GAME_COUNT
        print(label.format(average_point))

      next_actions = (mcts_action, random_action)
      %time evaluate_algorithm_of('VS_Random {:.3f}', next_actions)
      print()

      next_actions = (mcts_action, mini_max_action)
      %time evaluate_algorithm_of('VS_MiniMax {:.3f}', next_actions)
      print()

      next_actions = (mcts_action, alpha_beta_action)
      %time evaluate_algorithm_of('VS_AlphaBeta {:.3f}', next_actions)
      print()

  # %%
  next_actions = (mcts_action, alpha_beta_action)

  for EP_GAME_COUNT in 5, 10, 15, 20, 25, 50:
      %time evaluate_algorithm_of('VS_AlphaBeta {:.3f}', next_actions)
      print()

  # %%


