import sys

import cvxpy as cp
import gurobipy as gp
import numpy as np
from cvxopt import matrix
from gurobipy import GRB
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.consumption_matrix import ConsumptionMatrix
from pm4py.objects.petri_net.utils.incidence_matrix import IncidenceMatrix
from pm4py.util.lp import solver as lp_solver

from util.constants import SplitPointSolvers


def compute_exact_heuristics_with_splits(initial_vec: np.ndarray, final_vec: np.ndarray, cost_vec: list[int],
                                         incidence_matrix: IncidenceMatrix, consumption_matrix: ConsumptionMatrix,
                                         split_points: list[PetriNet.Transition],
                                         solver: SplitPointSolvers = SplitPointSolvers.GUROBI):
    """This method solves the Extended Marking Equation for Classic Alignments as defined in [1].

    [1] Van Dongen, B. F. (2018, January). Efficiently computing alignments: using the extended marking equation
    """
    if solver == SplitPointSolvers.GUROBI:
        return _compute_init_heuristic_with_split(initial_vec, final_vec, np.asarray(cost_vec), split_points,
                                                  incidence_matrix, consumption_matrix)
    else:
        mip = solver in [SplitPointSolvers.CBC_MI, SplitPointSolvers.GLPK_MI]
        return _compute_exact_heuristics_with_splits_cvx(initial_vec, final_vec, matrix(cost_vec), incidence_matrix,
                                                         consumption_matrix, split_points, solver, mip)


def _compute_exact_heuristics_with_splits_cvx(initial_vec: np.ndarray, final_vec: np.ndarray, cost_vec: matrix,
                                              incidence_matrix: IncidenceMatrix, consumption_matrix: ConsumptionMatrix,
                                              split_points: list[PetriNet.Transition],
                                              solver: SplitPointSolvers, mip: bool = True):
    """This method solves the Extended Marking Equation for Classic Alignments as defined in [1] using cvx notation.

    [1] Van Dongen, B. F. (2018, January). Efficiently computing alignments: using the extended marking equation
    """
    C = np.asmatrix(incidence_matrix.a_matrix)
    C_p = np.asmatrix(consumption_matrix.c_matrix)
    m_i = initial_vec
    m_f = final_vec
    k = len(split_points)
    len_t = len(incidence_matrix.transitions)

    # Initialize x and y
    if mip:
        x = cp.Variable(shape=(k + 1, len_t), integer=True)
        y = cp.Variable(shape=(k + 1, len_t), integer=True)
    else:
        x = cp.Variable(shape=(k + 1, len_t), nonneg=True)
        y = cp.Variable(shape=(k + 1, len_t), nonneg=True)

    # set objective
    objective = cp.Minimize(sum(cost_vec.T @ x[i, :] for i in range(k + 1)) +
                            sum(cost_vec.T @ y[i, :] for i in range(k + 1)))

    constraints = []

    # not mentioned in the paper, but necessary because of the initialization of y
    for t_idx in range(len_t):
        constraints += [y[0, t_idx] == 0]

    # add constraint 1 (marking equation)
    constraints += [m_i + C @ x[0, :] + sum(C @ (x[a, :] + y[a, :]) for a in range(1, k + 1)) == m_f]

    # constraint 2 (extended marking equation)
    constraint_2_tmp = m_i + C @ x[0, :]
    for a in range(1, k + 1):
        for b in range(1, a):
            if b == a - 1:
                constraint_2_tmp += C @ (x[b, :] + y[b, :])
        constraints += [constraint_2_tmp + C_p @ y[a, :] >= 0]

    # constraint 3 (x in N) and
    # constraint 4 (y in {0,1})
    if mip:
        for a in range(k + 1):
            for t in range(len_t):
                constraints += [x[a, t] >= 0]
                constraints += [y[a, t] >= 0]

    # constraint 5 (element with value 1 corresponds to the starting transition of sigma_a in the SPN) and
    # constraint 6 (exactly one element of y has value 1)
    for y_col, split_point in enumerate(split_points, 1):
        y_index = 0
        for t, t_idx in incidence_matrix.transitions.items():
            if t.name[0] == split_point:  # transition is the starting transition of sigma_a in the SPN
                y_index += y[y_col, t_idx]
        constraints += [y_index == 1]
        constraints += [sum(y[y_col, :]) == 1]  # constraint 6

    prob = cp.Problem(objective, constraints)

    match solver:
        case SplitPointSolvers.CBC_MI | SplitPointSolvers.CBC_LP:
            opt_obj = prob.solve(solver=cp.CBC)
        case SplitPointSolvers.GLPK_MI:
            opt_obj = prob.solve(solver=cp.GLPK_MI)
        case _:
            opt_obj = prob.solve(solver=cp.GLPK)

    return opt_obj, list(np.array(x.value).sum(axis=0) + np.array(y.value).sum(axis=0))


def _compute_init_heuristic_with_split(initial_vec: np.ndarray, final_vec: np.ndarray, cost_vec: np.ndarray,
                                       split_points: list[PetriNet.Transition], incidence_matrix: IncidenceMatrix,
                                       consumption_matrix: ConsumptionMatrix):
    """This method solves the Extended Marking Equation for Classic Alignments as defined in [1] using Gurobi.

    [1] Van Dongen, B. F. (2018, January). Efficiently computing alignments: using the extended marking equation
    """
    # Define lp problem
    m = gp.Model()
    m.Params.LogToConsole = 0
    # Create two 2-D arrays of integer variables X and Y, 0 to k+1
    k = len(split_points)
    t_index = len(incidence_matrix.transitions)
    # constraint 3, note that the variable type can be non-integer
    x = m.addMVar((k + 1, t_index), vtype=GRB.INTEGER, lb=0)
    y = m.addMVar((k + 1, t_index), vtype=GRB.INTEGER, lb=0)
    C = np.array(incidence_matrix.a_matrix)
    C_p = np.array(consumption_matrix.c_matrix)
    # Set objective
    m.setObjective(sum(cost_vec @ x[i, :] + cost_vec @ y[i, :] for i in range(k + 1)), GRB.MINIMIZE)
    # Add constraint 1
    cons_one = np.array(initial_vec)
    for i in range(k + 1):
        sum_x = C @ x[i, :]
        sum_y = C @ y[i, :]
        cons_one += sum_x + sum_y
    m.addConstr(cons_one == np.array(final_vec), "constraint_1")
    # Add constraint 2
    cons_two_temp = np.array(initial_vec) + C @ x[0, :]
    for a in range(1, k + 1):
        for b in range(1, a):
            if b == a - 1:
                cons_two_temp += C @ x[b, :] + C @ y[b, :]
        m.addConstr(cons_two_temp + C_p @ y[a, :] >= 0, "constraint_2")
    # Add constraints 4, 5 and 6:
    m.addConstr(y[0, :].sum() == 0, "y_0_equal_0")  # not a mentioned constraint
    for y_col, split_point in enumerate(split_points, 1):
        y_index = 0
        for t, t_idx in incidence_matrix.transitions.items():
            if t.name[0] == split_point:  # transition is the starting transition of sigma_a in the SPN
                y_index += y[y_col, t_idx]
        m.addConstr(y_index == 1, "y_eq_1")
        m.addConstr(y[y_col, :].sum() == 1, "y_sum_eq_1")  # constraint 5 / 6
    # optimize model
    m.optimize()
    return m.objVal, list(np.array(x.X).sum(axis=0) + np.array(y.X).sum(axis=0))


def compute_exact_heuristics_with_splits_order_aware(initial_vec: np.ndarray, final_vec: np.ndarray,
                                                     cost_vec: list[int], incidence_matrix: IncidenceMatrix,
                                                     consumption_matrix: ConsumptionMatrix,
                                                     split_points: list[PetriNet.Transition],
                                                     solver: SplitPointSolvers = SplitPointSolvers.GUROBI):
    """This method solves the Extended Marking Equation for Order-Aware Alignments as defined in [1].

    [1] Van Dongen, B. F. (2018, January). Efficiently computing alignments: using the extended marking equation
    """
    if solver == SplitPointSolvers.GUROBI:
        return _compute_init_heuristic_with_split_order_aware(initial_vec, final_vec, np.asarray(cost_vec),
                                                              split_points, incidence_matrix, consumption_matrix)
    else:
        mip = solver in [SplitPointSolvers.CBC_MI, SplitPointSolvers.GLPK_MI]
        return _compute_exact_heuristics_with_splits_cvx_order_aware(initial_vec, final_vec, matrix(cost_vec),
                                                                     incidence_matrix, consumption_matrix, split_points,
                                                                     solver, mip)


def _compute_exact_heuristics_with_splits_cvx_order_aware(initial_vec: np.ndarray, final_vec: np.ndarray,
                                                          cost_vec: matrix, incidence_matrix: IncidenceMatrix,
                                                          consumption_matrix: ConsumptionMatrix,
                                                          split_points: list[PetriNet.Transition],
                                                          solver: SplitPointSolvers, mip: bool = True):
    """This method solves the Extended Marking Equation for Order-Aware Alignments as defined in [1] using cvx notation.

    [1] Van Dongen, B. F. (2018, January). Efficiently computing alignments: using the extended marking equation
    """
    C = np.asmatrix(incidence_matrix.a_matrix)
    C_p = np.asmatrix(consumption_matrix.c_matrix)
    m_i = initial_vec
    m_f = final_vec
    k = len(split_points)
    len_t = len(incidence_matrix.transitions)

    # Initialize x and y
    if mip:
        x = cp.Variable(shape=(k + 1, len_t), integer=True)
        y = cp.Variable(shape=(k + 1, len_t), integer=True)
    else:
        x = cp.Variable(shape=(k + 1, len_t), nonneg=True)
        y = cp.Variable(shape=(k + 1, len_t), nonneg=True)

    # set objective
    objective = cp.Minimize(sum(cost_vec.T @ x[i, :] for i in range(k + 1)) +
                            sum(cost_vec.T @ y[i, :] for i in range(k + 1)))

    constraints = []

    # not mentioned in the paper, but necessary because of the initialization of y
    for t_idx in range(len_t):
        constraints += [y[0, t_idx] == 0]

    # add constraint 1 (marking equation)
    constraints += [m_i + C @ x[0, :] + sum(C @ (x[a, :] + y[a, :]) for a in range(1, k + 1)) == m_f]

    # constraint 2 (extended marking equation)
    constraint_2_tmp = m_i + C @ x[0, :]
    for a in range(1, k + 1):
        for b in range(1, a):
            if b == a - 1:
                constraint_2_tmp += C @ (x[b, :] + y[b, :])
        constraints += [constraint_2_tmp + C_p @ y[a, :] >= 0]

    # constraint 3 (x in N) and
    # constraint 4 (y in {0,1})
    if mip:
        for a in range(k + 1):
            for t in range(len_t):
                constraints += [x[a, t] >= 0]
                constraints += [y[a, t] >= 0]

    # constraint 5 (element with value 1 corresponds to the starting transition of sigma_a in the SPN) and
    # constraint 6 (exactly one element of y has value 1)
    for y_col, transitions_to_be_enabled in enumerate(split_points, 1):
        y_index = 0
        for t, t_idx in incidence_matrix.transitions.items():
            if t.name[0] in transitions_to_be_enabled:  # transition is the starting transition of sigma_a in the SPN
                y_index += y[y_col, t_idx]
        constraints += [y_index == 1]
        constraints += [sum(y[y_col, :]) == 1]  # constraint 6

    prob = cp.Problem(objective, constraints)

    match solver:
        case SplitPointSolvers.CBC_MI | SplitPointSolvers.CBC_LP:
            opt_obj = prob.solve(solver=cp.CBC)
        case SplitPointSolvers.GLPK_MI:
            opt_obj = prob.solve(solver=cp.GLPK_MI)
        case _:
            opt_obj = prob.solve(solver=cp.GLPK)

    return opt_obj, list(np.array(x.value).sum(axis=0) + np.array(y.value).sum(axis=0))


def _compute_init_heuristic_with_split_order_aware(ini_vec, fin_vec, cost, split_lst, incidence_matrix: IncidenceMatrix,
                                                   consumption_matrix: ConsumptionMatrix):
    """This method solves the Extended Marking Equation for Order-Aware Alignments as defined in [1] using Gurobi.

    [1] Van Dongen, B. F. (2018, January). Efficiently computing alignments: using the extended marking equation
    """
    # Define lp problem
    m = gp.Model()
    m.Params.LogToConsole = 0
    # Create two 2-D arrays of integer variables X and Y, 0 to k+1
    k = len(split_lst)
    t_index = len(incidence_matrix.transitions)
    # constraint 3, note that the variable type can be non-integer
    x = m.addMVar((k + 1, t_index), vtype=GRB.INTEGER, lb=0)
    y = m.addMVar((k + 1, t_index), vtype=GRB.INTEGER, lb=0)
    C = np.array(incidence_matrix.a_matrix)
    C_p = np.array(consumption_matrix.c_matrix)
    # Set objective
    m.setObjective(sum(cost @ x[i, :] + cost @ y[i, :] for i in range(k + 1)), GRB.MINIMIZE)
    # Add constraint 1
    cons_one = np.array(ini_vec)
    for i in range(k + 1):
        sum_x = C @ x[i, :]
        sum_y = C @ y[i, :]
        cons_one += sum_x + sum_y
    m.addConstr(cons_one == np.array(fin_vec), "constraint_1")
    # Add constraint 2
    cons_two_temp = np.array(ini_vec) + C @ x[0, :]
    for a in range(1, k + 1):
        for b in range(1, a):
            if b == a - 1:
                cons_two_temp += C @ x[b, :] + C @ y[b, :]
        m.addConstr(cons_two_temp + C_p @ y[a, :] >= 0, "constraint_2")
    # Add constraints 4, 5 and 6:
    m.addConstr(y[0, :].sum() == 0, "y_0_equal_0")  # not a mentioned constraint
    for y_col, transitions_to_be_enabled in enumerate(split_lst, 1):
        y_index = 0
        for t, t_idx in incidence_matrix.transitions.items():
            if t.name[0] in transitions_to_be_enabled:  # transition is the starting transition of sigma_a in the SPN
                y_index += y[y_col, t_idx]
        m.addConstr(y_index == 1, "y_eq_1")
        m.addConstr(y[y_col, :].sum() == 1, "y_sum_eq_1")  # constraint 5 / 6
    # optimize model
    m.optimize()
    return m.objVal, list(np.array(x.X).sum(axis=0) + np.array(y.X).sum(axis=0))


def vectorize_initial_final_cost(incidence_matrix: IncidenceMatrix, initial_marking: Marking, final_marking: Marking,
                                 cost_function: dict[PetriNet.Transition, int]):
    ini_vec = incidence_matrix.encode_marking(initial_marking)
    fini_vec = incidence_matrix.encode_marking(final_marking)
    cost_vec = [0] * len(cost_function)
    for t, costs in cost_function.items():
        cost_vec[incidence_matrix.transitions[t]] = costs
    return ini_vec, fini_vec, cost_vec


def vectorize_matrices(incidence_matrix: IncidenceMatrix, sync_net: PetriNet):
    a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
    g_matrix = -np.eye(len(sync_net.transitions))
    h_cvx = np.matrix(np.zeros(len(sync_net.transitions))).transpose()

    a_matrix = matrix(a_matrix)
    g_matrix = matrix(g_matrix)
    h_cvx = matrix(h_cvx)

    return a_matrix, g_matrix, h_cvx


def derive_heuristic(incidence_matrix: IncidenceMatrix, cost_vec, x: list[float], t: PetriNet.Transition,
                     h: int):
    x_prime = x.copy()
    x_prime[incidence_matrix.transitions[t]] -= 1
    return max(0, h - cost_vec[incidence_matrix.transitions[t]]), x_prime


def derive_heuristic_split_point(incidence_matrix: IncidenceMatrix, costs: int, z: list[float],
                                 t: PetriNet.Transition, h: int):
    z_prime = z.copy()

    if z_prime[incidence_matrix.transitions[t]] >= 1:
        h_new = h - costs
        z_prime[incidence_matrix.transitions[t]] -= 1
        feasible = True
    else:
        h_new = max(0, h - costs)
        z_prime = None
        feasible = False

    return h_new, z_prime, feasible


def is_solution_feasible(x: list[float]):
    for v in x:
        if v < -0.001:
            return False
    return True


def compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix, marking, fin_vec):
    # compute diff marking
    m_vec = incidence_matrix.encode_marking(marking)
    b_term = [i - j for i, j in zip(fin_vec, m_vec)]
    b_term = np.matrix([x * 1.0 for x in b_term]).transpose()
    b_term = matrix(b_term)

    sol = lp_solver.apply(cost_vec, g_matrix, h_cvx, a_matrix, b_term, parameters={"solver": "glpk"})
    prim_obj = lp_solver.get_prim_obj_from_sol(sol)
    points = lp_solver.get_points_from_sol(sol)

    prim_obj = prim_obj if prim_obj is not None else sys.maxsize
    points = points if points is not None else [0.0] * len(sync_net.transitions)

    return prim_obj, points
