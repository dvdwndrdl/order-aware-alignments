import heapq
import time

from cvxopt import matrix
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.incidence_matrix import construct as construct_incidence_matrix
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, decorate_transitions_prepostset

from algo.search_algorithm import SearchAlgorithm
from util.alignment_utils import get_transitions_to_visit, OrderAwareAlignmentResult
from util.constants import DependencyTypes
from util.heuristic_utils import compute_exact_heuristic, vectorize_initial_final_cost, derive_heuristic, \
    is_solution_feasible, vectorize_matrices
from util.search_tuples import OrderAwareSearchTuple, StateSpaceItem


class AStarOrderAware(SearchAlgorithm):
    def __init__(self, sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                 cost_function: dict[PetriNet.Transition, int], dependency_cost_function: dict[DependencyTypes, int]):
        self.start_time = time.time()

        self.cost_function = cost_function
        self.dep_cost_function = dependency_cost_function

        self.sync_net = sync_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking

        decorate_transitions_prepostset(self.sync_net)
        decorate_places_preset_trans(self.sync_net)
        self.trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

        self.incidence_matrix = construct_incidence_matrix(sync_net)
        self.ini_vec, self.fin_vec, self.cost_vec = \
            vectorize_initial_final_cost(self.incidence_matrix, initial_marking, final_marking, cost_function)
        self.a_matrix, self.g_matrix, self.h_cvx = vectorize_matrices(self.incidence_matrix, self.sync_net)
        self.cost_vec = matrix([1.0 * x for x in self.cost_vec])

        self.open_set: list[OrderAwareSearchTuple] = None
        self.explored_state_space: list[StateSpaceItem] = None
        self.traversed: int = None
        self.visited: int = None
        self.queued: int = None
        self.lp_solved: int = None

    def search(self) -> OrderAwareAlignmentResult:
        self._init_search()

        while self.open_set:
            curr = heapq.heappop(self.open_set)

            if curr.h < 0.01 and curr.m == self.final_marking:
                # OPTIMAL ALIGNMENT FOUND
                return OrderAwareAlignmentResult.from_unidirectional(
                    curr, self.visited, self.queued, self.traversed, time.time() - self.start_time, 0, self.lp_solved)

            # REQUEUE IF NON-REALIZABLE FIRING SEQUENCE
            if not curr.feasible:
                h, x = compute_exact_heuristic(self.sync_net, self.a_matrix, self.h_cvx, self.g_matrix, self.cost_vec,
                                               self.incidence_matrix, curr.m, self.fin_vec)
                self.lp_solved += 1
                curr.h = h
                curr.x = x
                curr.feasible = True
                heapq.heappush(self.open_set, curr)

                continue

            self.visited += 1
            self._expand_from_current_marking(curr)

    def _init_search(self):
        h, x = compute_exact_heuristic(self.sync_net, self.a_matrix, self.h_cvx, self.g_matrix, self.cost_vec,
                                       self.incidence_matrix, self.initial_marking, self.fin_vec)

        initial_search_tuple = OrderAwareSearchTuple(0, h, self.initial_marking, None, None, x, True)
        self.open_set = [initial_search_tuple]
        heapq.heapify(self.open_set)
        self.explored_state_space = [initial_search_tuple.state_space_item]

        self.queued = 1
        self.visited = 0
        self.traversed = 0
        self.lp_solved = 1

    def _expand_from_current_marking(self, current: OrderAwareSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset, self.cost_function):
            self._expand_transition_from_marking(current, t, cost)

    def _expand_transition_from_marking(self, current: OrderAwareSearchTuple,
                                        transition: PetriNet.Transition, cost: int):
        h, x = derive_heuristic(self.incidence_matrix, self.cost_vec, current.x, transition, current.h)
        feasible = is_solution_feasible(x)
        tp = OrderAwareSearchTuple.from_current_search_tuple(
            current, transition, cost, h, x, feasible, self.dep_cost_function)
        self.traversed += 1

        if tp.state_space_item in self.explored_state_space:
            return

        self.explored_state_space.append(tp.state_space_item)
        heapq.heappush(self.open_set, tp)
        self.queued += 1
