import heapq
import math
import time

from cvxopt import matrix
from pm4py.objects.petri_net.utils.consumption_matrix import construct as construct_consumption_matrix
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.incidence_matrix import construct as construct_incidence_matrix
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, decorate_transitions_prepostset

from algo.search_algorithm import SearchAlgorithm
from util import constants
from util.alignment_utils import get_transitions_to_visit, OrderAwareAlignmentResult
from util.constants import DependencyTypes, SplitPointSolvers
from util.heuristic_utils import compute_exact_heuristic, vectorize_initial_final_cost, vectorize_matrices, \
    derive_heuristic_split_point, compute_exact_heuristics_with_splits_order_aware
from util.search_tuples import OrderAwareSplitPointSearchTuple, StateSpaceItem


class SplitPointOrderAware(SearchAlgorithm):
    def __init__(self, sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                 cost_function: dict[PetriNet.Transition, int], dependency_cost_function: dict[DependencyTypes, int],
                 variant: SplitPointSolvers = SplitPointSolvers.GUROBI, trace_dependencies: dict = None):
        self.start_time = time.time()
        self.variant = variant

        self.cost_function = cost_function
        self.dep_cost_function = dependency_cost_function

        self.sync_net = sync_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.trace_dependencies = trace_dependencies

        decorate_transitions_prepostset(self.sync_net)
        decorate_places_preset_trans(self.sync_net)
        self.trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

        self.incidence_matrix = construct_incidence_matrix(sync_net)
        self.consumption_matrix = construct_consumption_matrix(sync_net)
        self.ini_vec, self.fin_vec, self.cost_vec = vectorize_initial_final_cost(
            self.incidence_matrix, self.initial_marking, self.final_marking, self.cost_function)
        self.a_matrix, self.g_matrix, self.h_cvx = vectorize_matrices(self.incidence_matrix, self.sync_net)
        self.cost_matrix = matrix([x * 1.0 for x in self.cost_vec])

        self.open_set: list[OrderAwareSplitPointSearchTuple] = None
        self.explored_state_space: list[StateSpaceItem] = None
        self.best_result_cost: float = None
        self.best_result: OrderAwareSplitPointSearchTuple = None

        self.visited: int = None
        self.queued: int = None
        self.traversed: int = None
        self.simple_lp: int = None
        self.complex_lp: int = None
        self.complex_duration: int = None

        self.max_events_explained = None
        self.split_points = None
        self.is_fallback_a_star = None

    def search(self) -> OrderAwareAlignmentResult:
        self._init_search()

        while self.open_set:
            curr = heapq.heappop(self.open_set)

            if curr.total_cost >= self.best_result_cost:
                # OPTIMAL ALIGNMENT FOUND
                return OrderAwareAlignmentResult.from_unidirectional(
                    self.best_result, self.visited, self.queued, self.traversed, time.time() - self.start_time,
                    self.complex_duration, self.simple_lp, self.complex_lp, len(self.split_points))

            # update best result so far
            if curr.h < 0.01 and curr.m == self.final_marking and self.best_result_cost > curr.total_cost:
                self.best_result_cost = curr.total_cost
                self.best_result = curr

            if not curr.feasible:
                # ORIGINALLY COMPUTED SOLUTION CORRESPONDS TO A NON-REALIZABLE FIRING SEQUENCE
                self.split_points = recompute_split_points(
                    self.trace_dependencies, self.max_events_explained, self.split_points)
                if len(self.split_points) == 0 or self.is_fallback_a_star:
                    # CONTINUE WITH REGULAR A* SINCE NO NEW SPLIT POINTS
                    self.is_fallback_a_star = True
                    h, x = compute_exact_heuristic(self.sync_net, self.a_matrix, self.h_cvx, self.g_matrix,
                                                   self.cost_matrix, self.incidence_matrix, curr.m, self.fin_vec)
                    self.simple_lp += 1
                    curr.h = h
                    curr.x = x
                    curr.feasible = True
                    heapq.heappush(self.open_set, curr)
                else:
                    # RESTART SEARCH WITH ADDITIONAL SPLIT POINT
                    self._restart_search()

                continue

            self.visited += 1
            self._expand_from_current_marking(curr)

    def _init_search(self):
        h, x = compute_exact_heuristic(self.sync_net, self.a_matrix, self.h_cvx, self.g_matrix, self.cost_matrix,
                                       self.incidence_matrix, self.initial_marking, self.fin_vec)
        initial_search_tuple = OrderAwareSplitPointSearchTuple(0, h, self.initial_marking, None, None, x, True)

        self.open_set = [initial_search_tuple]
        heapq.heapify(self.open_set)
        self.explored_state_space = [initial_search_tuple.state_space_item]

        self.best_result_cost = math.inf
        self.visited = 0
        self.queued = 1
        self.traversed = 0
        self.simple_lp = 1
        self.complex_lp = 0
        self.complex_duration = 0

        self.max_events_explained = []
        self.split_points = []
        self.is_fallback_a_star = False

    def _expand_from_current_marking(self, current: OrderAwareSplitPointSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset, self.cost_function):
            self._expand_transition_from_marking(current, t, cost)

    def _expand_transition_from_marking(self, current: OrderAwareSplitPointSearchTuple, transition: PetriNet.Transition,
                                        cost: int):
        h, x, feasible = derive_heuristic_split_point(self.incidence_matrix, cost, current.x, transition, current.h)

        # INCREASE MAX EVENTS EXPLAINED
        events_explained = current.events_explained.copy()
        if transition.label[0] != constants.SKIP:
            events_explained.append(transition.name[0])
            if len(events_explained) > len(self.max_events_explained):
                self.max_events_explained = events_explained

        tp = OrderAwareSplitPointSearchTuple.from_current_search_tuple(
            current, transition, cost, h, x, feasible, events_explained, self.dep_cost_function)
        self.traversed += 1

        if tp.total_cost >= self.best_result_cost or tp.state_space_item in self.explored_state_space:
            return

        self.explored_state_space.append(tp.state_space_item)
        heapq.heappush(self.open_set, tp)
        self.queued += 1

    def _restart_search(self):
        complex_lp_start = time.time()
        h, z = compute_exact_heuristics_with_splits_order_aware(self.ini_vec, self.fin_vec, self.cost_vec,
                                                                self.incidence_matrix, self.consumption_matrix,
                                                                self.split_points, self.variant)
        self.complex_duration += time.time() - complex_lp_start
        self.complex_lp += 1
        initial_search_tuple = OrderAwareSplitPointSearchTuple(0, h, self.initial_marking, None, None, z, True)
        self.open_set = [initial_search_tuple]
        self.explored_state_space = [initial_search_tuple.state_space_item]
        heapq.heapify(self.open_set)
        self.max_events_explained = []


def recompute_split_points(trace_dependencies, max_events_explained, current_split_points) -> list[list[str]]:
    never_seen = [e for e in trace_dependencies if e not in max_events_explained]
    independent_never_seen = [e for e in never_seen
                              if len(list(set(trace_dependencies[e]) - set(max_events_explained))) == 0]

    current_all_split_events = set([e for split_point in current_split_points for e in split_point])
    all_split_events = current_all_split_events.union(independent_never_seen)

    # we did not add a new event to the list of split point events; hence: continue
    if len(all_split_events) == len(current_all_split_events):
        return []

    split_points = []
    while all_split_events:
        # get all split point events without dependencies to other split point events
        new_split_point = [e for e in all_split_events if
                           len(list(set(trace_dependencies[e]) & set(all_split_events))) == 0]

        # remove independent split points from list with all split point event
        all_split_events -= set(new_split_point)

        # extend the new split point with additional, independent events from previous split points
        split_points_to_remove = []
        for split_point in split_points:
            # find sequential and concurrent events
            sequential_events = set()
            concurrent_events = set()
            for e in new_split_point:
                sequential_events_of_e = set(split_point) & set(trace_dependencies[e])
                sequential_events = sequential_events.union(sequential_events_of_e)
                concurrent_events_of_e = set(split_point) - set(trace_dependencies[e])
                concurrent_events = concurrent_events.union(concurrent_events_of_e)

            # if there are no concurrent events, the investigated split point is fully sequential, and we cannot extend the new split point
            if len(concurrent_events) == 0:
                continue

            # check if there are events that
            # (1) are concurrent to at least one event of the new split point AND
            # (2) are NOT sequential to any event of the new split points
            # if yes: extend the new split point with this/these concurrent event(s)
            concurrent_and_not_sequential = concurrent_events - sequential_events
            if len(concurrent_and_not_sequential) > 0:
                new_split_point += concurrent_and_not_sequential
                continue

            # if there are concurrent events AND they are also sequential, then we have to merge both split points,
            # i.e. we add the existing one to the new one and remove the existing one from the list
            new_split_point += split_point
            split_points_to_remove.append(split_point)

        # remove split points that have been merged
        for split_point_to_remove in split_points_to_remove:
            split_points.remove(split_point_to_remove)

        # add the new split point
        split_points.append(new_split_point)

    return split_points
