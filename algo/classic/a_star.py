import heapq
import time

from cvxopt import matrix
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.align_utils import add_markings
from pm4py.objects.petri_net.utils.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, decorate_transitions_prepostset

from algo.search_algorithm import SearchAlgorithm
from util.alignment_utils import get_transitions_to_visit, AlignmentResult
from util.heuristic_utils import compute_exact_heuristic, vectorize_initial_final_cost, derive_heuristic, \
    is_solution_feasible, vectorize_matrices
from util.search_tuples import SearchTuple


class AStar(SearchAlgorithm):
    def __init__(self, sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                 cost_function: dict[PetriNet.Transition, int]):
        self.start_time = time.time()

        self.cost_function = cost_function
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.sync_net = sync_net

        decorate_transitions_prepostset(self.sync_net)
        decorate_places_preset_trans(self.sync_net)
        self.trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

        self.incidence_matrix = inc_mat_construct(sync_net)
        self.ini_vec, self.fin_vec, self.cost_vec = \
            vectorize_initial_final_cost(self.incidence_matrix, initial_marking, final_marking, cost_function)
        self.a_matrix, self.g_matrix, self.h_cvx = vectorize_matrices(self.incidence_matrix, self.sync_net)
        self.cost_vec = matrix([x * 1.0 for x in self.cost_vec])

        self.closed = None
        self.open_set = None
        self.visited = None
        self.queued = None
        self.traversed = None
        self.lp_solved = None

    def search(self) -> AlignmentResult:
        self._init_search()

        while self.open_set:
            curr = heapq.heappop(self.open_set)

            # FINAL MARKING FOUND
            if curr.h < 0.01 and curr.m == self.final_marking:
                return AlignmentResult.from_unidirectional(curr, self.visited, self.queued, self.traversed,
                                                           time.time() - self.start_time, simple_lp=self.lp_solved)

            # ALREADY CLOSED
            if curr.m in self.closed:
                continue

            # REQUEUE IF NON-REALIZABLE FIRING SEQUENCE
            if not curr.feasible:
                h, x = compute_exact_heuristic(self.sync_net, self.a_matrix, self.h_cvx, self.g_matrix, self.cost_vec,
                                               self.incidence_matrix, curr.m, self.fin_vec)
                tp = SearchTuple(curr.g, h, curr.m, curr.p, curr.t, x, True)
                heapq.heappush(self.open_set, tp)

                self.lp_solved += 1
                continue

            self.closed.add(curr.m)
            self.visited += 1
            self._expand_from_current_marking(curr)

    def _init_search(self):
        h, x = compute_exact_heuristic(self.sync_net, self.a_matrix, self.h_cvx, self.g_matrix, self.cost_vec,
                                       self.incidence_matrix, self.initial_marking, self.fin_vec)

        self.closed = set()
        self.open_set = [SearchTuple(0, h, self.initial_marking, None, None, x, True)]
        heapq.heapify(self.open_set)

        self.visited = 0
        self.queued = 1
        self.traversed = 0
        self.lp_solved = 1

    def _expand_from_current_marking(self, current: SearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset, self.cost_function):
            self._expand_transition_from_marking(current, t, cost)

    def _expand_transition_from_marking(self, current: SearchTuple, transition: PetriNet.Transition, cost: int):
        self.traversed += 1

        new_marking = add_markings(current.m, transition.add_marking)
        if new_marking in self.closed:
            return

        h, x = derive_heuristic(self.incidence_matrix, self.cost_vec, current.x, transition, current.h)
        feasible = is_solution_feasible(x)
        tp = SearchTuple(current.g + cost, h, new_marking, current, transition, x, feasible)
        heapq.heappush(self.open_set, tp)
        self.queued += 1
