import gurobipy
import numpy as np
import pandas as pd
import gurobipy as gb
from gurobipy import GRB
import gurobi_logtools as glt

from solver import Solver
import sys
import time


class GurobiSolver(Solver):
  def __str__(self):
    return 'gurobi'

  def __init__(self):
    self.logs = None
    self.mps_path = None
    self.model: gurobipy.Model | None = None

  @staticmethod
  def varname(var):
    return var.VarName

  @staticmethod
  def varval(var):
    return var.X

  def get_vars(self):
    return self.model.getVars()

  def hide_output_to_console(self):
    gb.setParam('LogToConsole', 0)

  def create_integer_var(self, name, lower_bound=-GRB.INFINITY, upper_bound=GRB.INFINITY):
    v = self.model.addVar(name=name, lb=lower_bound, ub=upper_bound, vtype=GRB.INTEGER)
    return v

  def create_real_var(self, name, lower_bound=-GRB.INFINITY, upper_bound=GRB.INFINITY):
    v = self.model.addVar(name=name, lb=lower_bound, ub=upper_bound, vtype=GRB.CONTINUOUS)
    return v

  def create_binary_var(self, name):
    v = self.model.addVar(name=name, vtype=GRB.BINARY)
    return v

  def set_objective_function(self, equation, maximize=True):
    self.model.setObjective(equation)
    self.model.ModelSense = GRB.MAXIMIZE if maximize else GRB.MINIMIZE

  def add_constraint(self, cns, name):
    self.model.addConstr(cns, name)

  def load_model(self, mip_path: str):
    self.model = gb.read(mip_path)
    self.mps_path = mip_path

  def solve(self, means='gurobi', log_file='', data_mode="test", time_limit=3600, threads=1, max_solutions=50, search_mode=2) -> pd.DataFrame:
    if log_file != '':
      with open(log_file, 'wb'):
        pass
    self.model.Params.LogFile = log_file
    self.model.Params.TimeLimit = time_limit
    self.model.Params.Threads = threads
    # if mode is 'test', use the default configuration
    if data_mode == "train":
        self.model.Params.PoolSolutions = max_solutions
        self.model.Params.PoolSearchMode = search_mode

    self.model.optimize()
    if log_file:
      self.logs = glt.parse(log_file).progress('nodelog')
      self.logs['Means'] = pd.Series(np.repeat(means, len(self.logs)), index=self.logs.index)
      self.logs['MpsPath'] = pd.Series(np.repeat(self.mps_path, len(self.logs)), index=self.logs.index)
    print('Done solving with ' + means)
    return self.logs

  def disable_presolver(self):
    self.model.Params.Presolve = 0

  def disable_cuts(self):
    self.model.Params.Cuts = 0

  def disable_heuristics(self):
    self.model.Params.Heuristics = 0

  def set_aggressive(self):
    self.model.Params.MIPFocus = 1

  def get_sol_data(self):
    sols = []
    objs = []
    solc = self.model.getAttr('SolCount')
    for sn in range(solc):
      self.model.Params.SolutionNumber = sn
      sols.append(np.array(self.model.Xn))
      objs.append(self.model.PoolObjVal)

    sols = np.array(sols, dtype=np.float32)
    objs = np.array(objs, dtype=np.float32)
    return sols, objs
