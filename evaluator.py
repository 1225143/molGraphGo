
from thermo import Joback

class JobackEvaluator():
  def __init__(self, prop='Tb'):
    self.prop = prop

  def evaluate(self, smiles):
    props_joback = Joback(smiles).estimate()
    return props_joback.get(self.prop)
  