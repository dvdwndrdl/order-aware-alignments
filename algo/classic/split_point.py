import heapq
import time

from cvxopt import matrix
from pm4py.objects.petri_net.utils.consumption_matrix import construct as construct_consumption_matrix
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.align_utils import add_markings
from pm4py.objects.petri_net.utils.incidence_matrix import construct as construct_incidence_matrix
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, decorate_transitions_prepostset

from algo.search_algorithm import SearchAlgorithm
from util import constants
from util.alignment_utils import get_transitions_to_visit, AlignmentResult
from util.constants import SplitPointSolvers
from util.heuristic_utils import compute_exact_heuristic, vectorize_initial_final_cost, vectorize_matrices, \
    compute_exact_heuristics_with_splits, derive_heuristic_split_point
from util.search_tuples import SplitPointSearchTuple


class SplitPoint(SearchAlgorithm):
    def __init__(self, sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                 cost_function: dict[PetriNet.Transition, int], variant: SplitPointSolvers = SplitPointSolvers.GUROBI):
        self.start_time = time.time()
        self.variant = variant

        self.cost_function = cost_function
        self.sync_net = sync_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking

        self.trace_net_mapping = {int(t.name[0].split("_")[-1]): t.name[0] for t in sync_net.transitions
                                  if t.name[0] != constants.SKIP and t.name[1] == constants.SKIP}
        self.trace_length = len(self.trace_net_mapping)

        decorate_transitions_prepostset(self.sync_net)
        decorate_places_preset_trans(self.sync_net)
        self.trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

        self.incidence_matrix = construct_incidence_matrix(sync_net)
        self.consumption_matrix = construct_consumption_matrix(sync_net)
        self.ini_vec, self.fin_vec, self.cost_vec = vectorize_initial_final_cost(
            self.incidence_matrix, self.initial_marking, self.final_marking, self.cost_function)
        self.a_matrix, self.g_matrix, self.h_cvx = vectorize_matrices(self.incidence_matrix, self.sync_net)
        self.cost_matrix = matrix([x * 1.0 for x in self.cost_vec])

        self.closed = None
        self.open_set = None

        self.visited = None
        self.queued = None
        self.traversed = None
        self.simple_lp = None
        self.complex_lp = None
        self.complex_duration = None

        self.max_event_explained = None
        self.split_points = None

    def search(self) -> AlignmentResult:
        self._init_search()

        while self.open_set:
            curr = heapq.heappop(self.open_set)

            # FINAL MARKING FOUND
            if curr.h < 0.01 and curr.m == self.final_marking:
                return AlignmentResult.from_unidirectional(curr, self.visited, self.queued, self.traversed,
                                                           time.time() - self.start_time, self.complex_duration,
                                                           self.simple_lp, self.complex_lp, k=len(self.split_points))

            # ALREADY CLOSED
            if curr.m in self.closed:
                if self.closed[curr.m].g > curr.g:
                    del self.closed[curr.m]
                else:
                    continue

            if not curr.feasible:
                # ORIGINALLY COMPUTED SOLUTION CORRESPONDS TO A NON-REALIZABLE FIRING SEQUENCE
                self.max_event_explained = self._get_max_event_not_in_split_list(self.max_event_explained)
                if self.max_event_explained in self.split_points or self.max_event_explained >= self.trace_length or self.max_event_explained < 0:
                    # CONTINUE WITH REGULAR A* SINCE NO NEW SPLIT POINTS
                    h, x = compute_exact_heuristic(self.sync_net, self.a_matrix, self.h_cvx, self.g_matrix,
                                                   self.cost_matrix, self.incidence_matrix, curr.m, self.fin_vec)
                    self.simple_lp += 1
                    tp = SplitPointSearchTuple(curr.g, h, curr.m, curr.p, curr.t, x, True, curr.max_event_explained)
                    heapq.heappush(self.open_set, tp)
                else:
                    # RESTART SEARCH WITH ADDITIONAL SPLIT POINT
                    self._restart_search()

                continue

            # SUCCESSFULLY CLOSED / EXPLORED
            self.closed[curr.m] = curr
            self.visited += 1
            self._expand_from_current_marking(curr)

    def _init_search(self):
        h, x = compute_exact_heuristic(self.sync_net, self.a_matrix, self.h_cvx, self.g_matrix, self.cost_matrix,
                                       self.incidence_matrix, self.initial_marking, self.fin_vec)

        self.closed = {}
        self.open_set = [SplitPointSearchTuple(0, h, self.initial_marking, None, None, x, True, -1)]
        heapq.heapify(self.open_set)

        self.visited = 0
        self.queued = 0
        self.traversed = 0
        self.simple_lp = 1
        self.complex_lp = 0
        self.complex_duration = 0

        self.max_event_explained = -1
        self.split_points = []

    def _get_max_event_not_in_split_list(self, max_event: int):
        if max_event not in self.split_points or max_event < 0:
            return max_event
        return self._get_max_event_not_in_split_list(max_event - 1)

    def _expand_from_current_marking(self, current: SplitPointSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset, self.cost_function):
            self._expand_transition_from_marking(current, t, cost)

    def _expand_transition_from_marking(self, current: SplitPointSearchTuple, transition: PetriNet.Transition,
                                        cost: int):
        self.traversed += 1

        new_marking = add_markings(current.m, transition.add_marking)
        new_g = current.g + cost
        h, x, feasible = derive_heuristic_split_point(self.incidence_matrix, cost, current.x, transition, current.h)

        if new_marking in self.closed:
            if self.closed[new_marking].g > new_g:
                del self.closed[new_marking]
            else:
                return

        # INCREASE MAX EVENTS EXPLAINED
        no_events_explained = current.max_event_explained
        if transition.label[0] != constants.SKIP:
            no_events_explained += 1
            if no_events_explained > self.max_event_explained:
                self.max_event_explained = no_events_explained

        tp = SplitPointSearchTuple(new_g, h, new_marking, current, transition, x, feasible, no_events_explained)
        heapq.heappush(self.open_set, tp)
        self.queued += 1

    def _restart_search(self):
        self.split_points = sorted(self.split_points + [self.max_event_explained])
        split_points = [self.trace_net_mapping[s] for s in self.split_points]
        complex_lp_start = time.time()
        h, z = compute_exact_heuristics_with_splits(self.ini_vec, self.fin_vec, self.cost_vec, self.incidence_matrix,
                                                    self.consumption_matrix, split_points, self.variant)
        self.complex_duration += time.time() - complex_lp_start
        self.complex_lp += 1
        self.closed = {}
        self.open_set = [SplitPointSearchTuple(0, h, self.initial_marking, None, None, z, True, -1)]
        heapq.heapify(self.open_set)
        self.max_event_explained = -1
