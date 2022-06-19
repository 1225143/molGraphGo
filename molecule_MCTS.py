#!/usr/bin/env python

# %%
import os
from platform import node
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx
import math
import random
import graphviz

# %%
g = nx.Graph()
nodes = ['start', 'C', 'O', 'N', 'c1ccc(cc1)', 'end']
edges = list(combinations(nodes, 2))
print(edges)
g.add_nodes_from(nodes)
g.add_edges_from(edges)
view = nx.draw_networkx(g)

# %%
class State():
  def __init__(self, molGraph=None, pieces=None, enemy_pieces=None):
    self.g = self.molGraph = molGraph
    self.current_node = node
    self.allowed_nodes = None
    self.reward = None
 
  def is_done(self):
    return len(self.legal_s()) == 0
  
  def next(self, action):
    pieces = self.pieces.copy()
    pieces[action] = 1
    return State(self.enemy_pieces, pieces)
  
  def legal_actions(self):
    actions = []
    for i in range(9):
      if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
        actions.append(i)
    return actions
  
  def is_first_player(self):
    return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)
  
  def __str__(self):
    ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
    string = ''
    for i in range(9):
      if self.pieces[i] == 1:
        string += ox[0]
      elif self.enemy_pieces[i] == 1:
        string += ox[1]
      else:
        string +=  '-'
      if i % 3 == 2:
        string += '\n'
    return string

# %%
class Node:
  def __init__(self, state):
    self.state = state
    self.w = 0
    self.n = 0
    self.child_nodes = None

  def evaluate(self):
    if self.state.is_done():
      value = self._evaluate_md()
      self.w += value
      self.n += 1
      return value

    if not self.child_nodes:
      value = playout(self.state)

      self.w += value
      self.n += 1
      if self.n == 10:
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
    
    return self.child_nodes[argmax(ucb1_values)]
  

# %%
def mcts_action(state):
   
  root_node = Node(state)
  root_node.expand()

  for _ in range(100):
    root_node.evaluate()
  
  legal_actions = state.legal_actions()
  n_list = []
  for c in root_node.child_nodes:
    n_list.append(c.n)
  return legal_actions[argmax(n_list)]


# %%
def playout(state):
  if state.is_lose():
    return -1
  if state.is_draw():
    return 0
  return -playout(state.next(random_action(state)))

# %%
def argmax(collection, key=None):
  return collection.index(max(collection))

# %%
def random_action(state):
  legal_actions = state.legal_actions()
  return legal_actions[random.randint(0, len(legal_actions)-1)]

# %%
def mini_max_action(state):
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

def mini_max_plus(state, limit):
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
def alpha_beta(state, alpha, beta):
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

def alpha_beta_action(state):
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

def first_player_point(ended_state):
  if ended_state.is_lose():
    return 0 if ended_state.is_first_player() else 1
  return 0.5

def play(next_actions):
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



