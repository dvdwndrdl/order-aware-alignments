import heapq
import math
import time

from cvxopt import matrix
from pm4py.objects.petri_net.utils.consumption_matrix import construct as construct_consumption_matrix
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.incidence_matrix import construct as construct_incidence_matrix
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, decorate_transitions_prepostset

from algo.order_aware import recompute_split_points
from algo.search_algorithm import BidirectionalSearchAlgorithm
from util import constants
from util.alignment_utils import get_transitions_to_visit, OrderAwareAlignmentResult
from util.constants import DependencyTypes, SplitPointSolvers, AlternatingMethod, TerminationCriterion
from util.dependency_utils import combine_dependency_matrices, compute_dependency_cost_from_matrix
from util.heuristic_utils import compute_exact_heuristic, vectorize_initial_final_cost, vectorize_matrices, \
    derive_heuristic_split_point, compute_exact_heuristics_with_splits_order_aware
from util.petri_net_utils import invert_petri_net
from util.search_tuples import OrderAwareSplitPointSearchTuple, StateSpaceItem, HashableDict


class SplitPointOrderAwareBidirectional(BidirectionalSearchAlgorithm):
    def __init__(self, sync_net: PetriNet, initial_marking: Marking, final_marking: Marking, trace_dependencies: dict,
                 cost_function: dict[PetriNet.Transition, int], dependency_cost_function: dict[DependencyTypes, int],
                 alternating_method: AlternatingMethod = AlternatingMethod.STRICTLY_ALTERNATE,
                 stopping_criterion: TerminationCriterion = TerminationCriterion.SYMMETRIC_LOWER_BOUNDING,
                 variant: SplitPointSolvers = SplitPointSolvers.GUROBI):
        self.start_time = time.time()
        self.variant = variant
        self.alternating_method = alternating_method
        self.stopping_criterion = stopping_criterion

        self.sync_net = sync_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking

        # create reverse dependencies
        self.trace_dependencies_f = trace_dependencies
        self.trace_dependencies_r = {}
        for e, deps in trace_dependencies.items():
            for d in deps:
                self.trace_dependencies_r.setdefault(d, []).append(e)

        # build reverse product net
        self.sync_net_r, self.final_marking_r, self.initial_marking_r = \
            invert_petri_net(self.sync_net, self.initial_marking, self.final_marking)

        # build reverse cost function
        self.dep_cost_function = dependency_cost_function
        self.cost_function = cost_function
        cost_function_str = {str(t): c for t, c in cost_function.items()}
        self.cost_function_r = {t: cost_function_str[str(t)] for t in self.sync_net_r.transitions}

        decorate_transitions_prepostset(self.sync_net)
        decorate_places_preset_trans(self.sync_net)
        self.trans_empty_preset_f = set(t for t in self.sync_net.transitions if len(t.in_arcs) == 0)

        decorate_transitions_prepostset(self.sync_net_r)
        decorate_places_preset_trans(self.sync_net_r)
        self.trans_empty_preset_r = set(t for t in self.sync_net_r.transitions if len(t.in_arcs) == 0)

        # init forward search
        self.incidence_matrix_f = construct_incidence_matrix(sync_net)
        self.consumption_matrix_f = construct_consumption_matrix(sync_net)
        self.ini_vec_f, self.fin_vec_f, self.cost_vec_f = vectorize_initial_final_cost(
            self.incidence_matrix_f, self.initial_marking, self.final_marking, self.cost_function)
        self.a_matrix_f, self.g_matrix_f, self.h_cvx_f = vectorize_matrices(self.incidence_matrix_f, self.sync_net)
        self.cost_matrix_f = matrix([x * 1.0 for x in self.cost_vec_f])

        # init reverse search
        self.incidence_matrix_r = construct_incidence_matrix(self.sync_net_r)
        self.consumption_matrix_r = construct_consumption_matrix(self.sync_net_r)
        self.ini_vec_r, self.fin_vec_r, self.cost_vec_r = vectorize_initial_final_cost(
            self.incidence_matrix_r, self.initial_marking_r, self.final_marking_r, self.cost_function_r)
        self.a_matrix_r, self.g_matrix_r, self.h_cvx_r = vectorize_matrices(self.incidence_matrix_r, self.sync_net_r)
        self.cost_matrix_r = matrix([x * 1.0 for x in self.cost_vec_r])

        self.open_set_f: list[OrderAwareSplitPointSearchTuple] = None
        self.explored_state_space_f: list[StateSpaceItem] = None
        self.scanned_f: dict[frozenset, dict[HashableDict, OrderAwareSplitPointSearchTuple]] = None
        self.max_event_explained_f = None
        self.split_points_f = None
        self.is_fallback_a_star_f = None

        self.open_set_r: list[OrderAwareSplitPointSearchTuple] = None
        self.explored_state_space_r: list[StateSpaceItem] = None
        self.scanned_r: dict[frozenset, dict[HashableDict, OrderAwareSplitPointSearchTuple]] = None
        self.max_event_explained_r = None
        self.split_points_r = None
        self.is_fallback_a_star_r = None

        self.visited: int = None
        self.queued: int = None
        self.traversed: int = None
        self.simple_lp: int = None
        self.complex_lp: int = None

        self.est_dist_s_t: float = None
        self.mu: float = None
        self.mu_forward_state: OrderAwareSplitPointSearchTuple = None
        self.mu_reverse_state: OrderAwareSplitPointSearchTuple = None

    def search(self) -> OrderAwareAlignmentResult:
        self._init_search()

        match self.alternating_method:
            case AlternatingMethod.LOWEST_G_COST:
                return self._search_lowest_g_first()
            case AlternatingMethod.SMALLER_OPEN_SET:
                return self._search_smaller_open_set_first()
            case _:
                return self._search_alternating()

    def _search_alternating(self) -> OrderAwareAlignmentResult:
        is_forward = True
        while self.open_set_f and self.open_set_r:
            if self._is_optimal():
                return self._get_alignment()

            if is_forward:
                self._forward_search_step()
            else:
                self._reverse_search_step()

            is_forward = not is_forward

        return self._get_alignment()

    def _search_lowest_g_first(self) -> OrderAwareAlignmentResult:
        while self.open_set_f and self.open_set_r:
            if self._is_optimal():
                return self._get_alignment()

            if self.open_set_f[0].g <= self.open_set_r[0].g:
                self._forward_search_step()
            else:
                self._reverse_search_step()

        return self._get_alignment()

    def _search_smaller_open_set_first(self) -> OrderAwareAlignmentResult:
        while self.open_set_f and self.open_set_r:
            if self._is_optimal():
                return self._get_alignment()

            if len(self.open_set_f) <= len(self.open_set_r):
                self._forward_search_step()
            else:
                self._reverse_search_step()

        return self._get_alignment()

    def _is_optimal(self) -> bool:
        match self.stopping_criterion:
            case TerminationCriterion.AVERAGE_FUNCTION:
                top_f = self.open_set_f[0].total_cost_so_far
                top_r = self.open_set_r[0].total_cost_so_far
                return (top_f + top_r) >= (self.mu + self.est_dist_s_t)
            case TerminationCriterion.HALF_AVERAGE_FUNCTION:
                top_f = self.open_set_f[0].total_cost_so_far
                top_r = self.open_set_r[0].total_cost_so_far
                return (top_f + top_r) >= (self.mu + 0.5 * self.est_dist_s_t)
            case _:
                top_f = self.open_set_f[0].total_cost
                top_r = self.open_set_r[0].total_cost
                return top_f >= self.mu or top_r >= self.mu

    def _get_alignment(self):
        return OrderAwareAlignmentResult.from_bidirectional(
            self.mu_forward_state, self.mu_reverse_state, self.visited, self.queued, self.traversed,
            time.time() - self.start_time, self.complex_duration, self.simple_lp, self.complex_lp, 0,
            self.dep_cost_function)

    def _forward_search_step(self):
        curr = heapq.heappop(self.open_set_f)

        if not curr.feasible:
            # ORIGINALLY COMPUTED SOLUTION CORRESPONDS TO A NON-REALIZABLE FIRING SEQUENCE
            self.split_points_f = recompute_split_points(
                self.trace_dependencies_f, self.max_event_explained_f, self.split_points_f)
            if len(self.split_points_f) == 0 or self.is_fallback_a_star_f:
                # CONTINUE WITH REGULAR A* SINCE NO NEW SPLIT POINTS
                self.is_fallback_a_star_f = True
                h, x = compute_exact_heuristic(self.sync_net, self.a_matrix_f, self.h_cvx_f, self.g_matrix_f,
                                               self.cost_matrix_f, self.incidence_matrix_f, curr.m, self.fin_vec_f)
                self.simple_lp += 1
                curr.h = h
                curr.x = x
                curr.feasible = True
                heapq.heappush(self.open_set_f, curr)
            else:
                # RESTART SEARCH WITH ADDITIONAL SPLIT POINT
                self._restart_search_forward()

            return

        self.visited += 1
        self._expand_from_current_marking_forward(curr)

    def _reverse_search_step(self):
        curr = heapq.heappop(self.open_set_r)

        if not curr.feasible:
            # ORIGINALLY COMPUTED SOLUTION CORRESPONDS TO A NON-REALIZABLE FIRING SEQUENCE
            self.split_points_r = recompute_split_points(
                self.trace_dependencies_r, self.max_event_explained_r, self.split_points_r)
            if len(self.split_points_r) == 0 or self.is_fallback_a_star_r:
                # CONTINUE WITH REGULAR A* SINCE NO NEW SPLIT POINTS
                self.is_fallback_a_star_r = True
                h, x = compute_exact_heuristic(self.sync_net_r, self.a_matrix_r, self.h_cvx_r, self.g_matrix_r,
                                               self.cost_matrix_r, self.incidence_matrix_r, curr.m, self.fin_vec_r)
                self.simple_lp += 1
                curr.h = h
                curr.x = x
                curr.feasible = True
                heapq.heappush(self.open_set_r, curr)
            else:
                # RESTART SEARCH WITH ADDITIONAL SPLIT POINT
                self._restart_search_reverse()

            return

        self.visited += 1
        self._expand_from_current_marking_reverse(curr)

    def _init_search(self):
        # init forward search
        h_f, x_f = compute_exact_heuristic(self.sync_net, self.a_matrix_f, self.h_cvx_f, self.g_matrix_f,
                                           self.cost_matrix_f, self.incidence_matrix_f, self.initial_marking,
                                           self.fin_vec_f)

        self.est_dist_s_t = h_f

        ini_state_f = OrderAwareSplitPointSearchTuple(0, h_f, self.initial_marking, None, None, x_f, True)
        self.open_set_f = [ini_state_f]
        heapq.heapify(self.open_set_f)
        self.scanned_f = {ini_state_f.simple_marking: {ini_state_f.simple_parikh_vector: ini_state_f}}
        self.explored_state_space_f = [ini_state_f.state_space_item]

        self.max_event_explained_f = []
        self.split_points_f = []
        self.is_fallback_a_star_f = False

        # init reverse search
        h_r, x_r = compute_exact_heuristic(self.sync_net_r, self.a_matrix_r, self.h_cvx_r, self.g_matrix_r,
                                           self.cost_matrix_r, self.incidence_matrix_r, self.initial_marking_r,
                                           self.fin_vec_r)

        ini_state_r = OrderAwareSplitPointSearchTuple(0, h_r, self.initial_marking_r, None, None, x_r, True)
        self.open_set_r = [ini_state_r]
        heapq.heapify(self.open_set_r)
        self.scanned_r = {ini_state_r.simple_marking: {ini_state_r.simple_parikh_vector: ini_state_r}}
        self.explored_state_space_r = [ini_state_f.state_space_item]

        self.max_event_explained_r = []
        self.split_points_r = []
        self.is_fallback_a_star_r = False

        self.mu = math.inf
        self.visited = 0
        self.queued = 2
        self.traversed = 0
        self.simple_lp = 1
        self.complex_lp = 0
        self.complex_duration = 0

    def _expand_from_current_marking_forward(self, current: OrderAwareSplitPointSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_f, self.cost_function):
            self._expand_transition_from_marking_forward(current, t, cost)

    def _expand_from_current_marking_reverse(self, current: OrderAwareSplitPointSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_r, self.cost_function_r):
            self._expand_transition_from_marking_reverse(current, t, cost)

    def _expand_transition_from_marking_forward(self, current: OrderAwareSplitPointSearchTuple,
                                                transition: PetriNet.Transition, cost: int):
        h, x, feasible = derive_heuristic_split_point(self.incidence_matrix_f, cost, current.x, transition, current.h)

        # INCREASE MAX EVENTS EXPLAINED
        events_explained = current.events_explained.copy()
        if transition.label[0] != constants.SKIP:
            events_explained.append(transition.name[0])
            if len(events_explained) > len(self.max_event_explained_f):
                self.max_event_explained_f = events_explained

        tp = OrderAwareSplitPointSearchTuple.from_current_search_tuple(
            current, transition, cost, h, x, feasible, events_explained, self.dep_cost_function)
        self.traversed += 1

        if tp.total_cost >= self.mu:
            return

        if tp.state_space_item not in self.explored_state_space_f:
            self.explored_state_space_f.append(tp.state_space_item)
            heapq.heappush(self.open_set_f, tp)
            self.queued += 1

        if tp.simple_marking not in self.scanned_f:
            self.scanned_f[tp.simple_marking] = {tp.simple_parikh_vector: tp}
        elif tp.simple_parikh_vector not in self.scanned_f[tp.simple_marking]:
            self.scanned_f[tp.simple_marking][tp.simple_parikh_vector] = tp

        matching = self.scanned_r.get(tp.simple_marking, {})
        for match in matching.values():
            self._match_forward_and_reverse_search(tp, match)

    def _expand_transition_from_marking_reverse(self, current: OrderAwareSplitPointSearchTuple,
                                                transition: PetriNet.Transition, cost: int):
        h, x, feasible = derive_heuristic_split_point(self.incidence_matrix_r, cost, current.x, transition, current.h)

        # INCREASE MAX EVENTS EXPLAINED
        events_explained = current.events_explained.copy()
        if transition.label[0] != constants.SKIP:
            events_explained.append(transition.name[0])
            if len(events_explained) > len(self.max_event_explained_r):
                self.max_event_explained_r = events_explained

        tp = OrderAwareSplitPointSearchTuple.from_current_search_tuple(
            current, transition, cost, h, x, feasible, events_explained, self.dep_cost_function)
        self.traversed += 1

        if tp.total_cost >= self.mu:
            return

        if tp.state_space_item not in self.explored_state_space_r:
            self.explored_state_space_r.append(tp.state_space_item)
            heapq.heappush(self.open_set_r, tp)
            self.queued += 1

        if tp.simple_marking not in self.scanned_r:
            self.scanned_r[tp.simple_marking] = {tp.simple_parikh_vector: tp}
        elif tp.simple_parikh_vector not in self.scanned_r[tp.simple_marking]:
            self.scanned_r[tp.simple_marking][tp.simple_parikh_vector] = tp

        matching = self.scanned_f.get(tp.simple_marking, {})
        for match in matching.values():
            self._match_forward_and_reverse_search(match, tp)

    def _match_forward_and_reverse_search(self, forward_tuple: OrderAwareSplitPointSearchTuple,
                                          reverse_tuple: OrderAwareSplitPointSearchTuple):
        # more detailed test including the connecting dependencies only if total costs so far are lower than current mu value
        if forward_tuple.total_cost + reverse_tuple.total_cost >= self.mu:
            return

        combined_dep_matrix = combine_dependency_matrices(forward_tuple, reverse_tuple)
        new_mu = forward_tuple.g + reverse_tuple.g + compute_dependency_cost_from_matrix(combined_dep_matrix,
                                                                                         self.dep_cost_function)

        if new_mu < self.mu:
            self.mu = new_mu
            self.mu_forward_state = forward_tuple
            self.mu_reverse_state = reverse_tuple

    def _restart_search_forward(self):
        complex_lp_start = time.time()
        h, z = compute_exact_heuristics_with_splits_order_aware(self.ini_vec_f, self.fin_vec_f, self.cost_vec_f,
                                                                self.incidence_matrix_f, self.consumption_matrix_f,
                                                                self.split_points_f, self.variant)
        self.complex_duration += time.time() - complex_lp_start
        self.complex_lp += 1

        ini_state_f = OrderAwareSplitPointSearchTuple(0, h, self.initial_marking, None, None, z, True)
        self.open_set_f = [ini_state_f]
        heapq.heapify(self.open_set_f)
        self.scanned_f = {ini_state_f.simple_marking: {ini_state_f.simple_parikh_vector: ini_state_f}}
        self.explored_state_space_f = [ini_state_f.state_space_item]
        self.max_event_explained_f = []

    def _restart_search_reverse(self):
        complex_lp_start = time.time()
        h, z = compute_exact_heuristics_with_splits_order_aware(self.ini_vec_r, self.fin_vec_r, self.cost_vec_r,
                                                                self.incidence_matrix_r, self.consumption_matrix_r,
                                                                self.split_points_r, self.variant)
        self.complex_duration += time.time() - complex_lp_start
        self.complex_lp += 1

        ini_state_r = OrderAwareSplitPointSearchTuple(0, h, self.initial_marking_r, None, None, z, True)
        self.open_set_r = [ini_state_r]
        heapq.heapify(self.open_set_r)
        self.scanned_r = {ini_state_r.simple_marking: {ini_state_r.simple_parikh_vector: ini_state_r}}
        self.explored_state_space_r = [ini_state_r.state_space_item]
        self.max_event_explained_r = []
