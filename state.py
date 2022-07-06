
# %%
class State():
  def __init__(self, molGraph=None, pieces=None, enemy_pieces=None):
    self.g = self.molGraph = molGraph
    self.current_node = None 
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

  