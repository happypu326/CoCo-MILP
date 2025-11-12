from solver.gurobi import GurobiSolver
from solver.scip import SCIPSolver
SOLVER_CLASSES = {'scip': SCIPSolver, 'gurobi' : GurobiSolver}