import heapq
import math
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
from util.constants import SplitPointSolvers, TerminationCriterion, AlternatingMethod
from util.heuristic_utils import compute_exact_heuristic, vectorize_initial_final_cost, vectorize_matrices, \
    compute_exact_heuristics_with_splits, derive_heuristic_split_point
from util.petri_net_utils import invert_petri_net
from util.search_tuples import SplitPointSearchTuple


class SplitPointBidirectional(SearchAlgorithm):
    def __init__(self, sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                 cost_function: dict[PetriNet.Transition, int],
                 stopping_criterion: TerminationCriterion = TerminationCriterion.SYMMETRIC_LOWER_BOUNDING,
                 alternating_method: AlternatingMethod = AlternatingMethod.STRICTLY_ALTERNATE,
                 solver: SplitPointSolvers = SplitPointSolvers.GUROBI):
        self.start_time = time.time()
        self.solver = solver
        self.stopping_criterion = stopping_criterion
        self.alternating_method = alternating_method

        self.sync_net = sync_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking

        # build reverse product net
        self.sync_net_r, self.final_marking_r, self.initial_marking_r = invert_petri_net(
            self.sync_net, self.initial_marking, self.final_marking)

        # create reverse cost function
        self.cost_function = cost_function
        cost_function_str = {str(t): c for t, c in cost_function.items()}
        self.cost_function_r = {}
        for t in self.sync_net_r.transitions:
            self.cost_function_r[t] = cost_function_str[str(t)]

        self.trace_net_mapping_f = {int(t.name[0].split("_")[-1]): t.name[0] for t in sync_net.transitions
                                    if t.name[0] != constants.SKIP and t.name[1] == constants.SKIP}
        self.trace_length = len(self.trace_net_mapping_f)
        self.trace_net_mapping_r = {(self.trace_length - 1) - t_idx: t for t_idx, t in self.trace_net_mapping_f.items()}

        decorate_transitions_prepostset(self.sync_net)
        decorate_places_preset_trans(self.sync_net)
        self.trans_empty_preset_f = set(t for t in self.sync_net.transitions if len(t.in_arcs) == 0)

        decorate_transitions_prepostset(self.sync_net_r)
        decorate_places_preset_trans(self.sync_net_r)
        self.trans_empty_preset_r = set(t for t in self.sync_net_r.transitions if len(t.in_arcs) == 0)

        # init forward search
        self.incidence_matrix_f = construct_incidence_matrix(self.sync_net)
        self.consumption_matrix_f = construct_consumption_matrix(self.sync_net)
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

        self.closed_f = None
        self.open_set_f = None
        self.max_event_explained_f = None
        self.split_points_f = None
        self.scanned_markings_f = None

        self.closed_r = None
        self.open_set_r = None
        self.max_event_explained_r = None
        self.split_points_r = None
        self.scanned_markings_r = None

        self.visited = None
        self.queued = None
        self.traversed = None
        self.simple_lp = None
        self.complex_lp = None
        self.complex_duration = None

        self.mu = math.inf
        self.mu_forward_state = None
        self.mu_reverse_state = None

        self.est_dist_s_t = None

    def search(self) -> AlignmentResult:
        self._init_search()

        match self.alternating_method:
            case AlternatingMethod.LOWEST_G_COST:
                return self._search_lowest_g_first()
            case AlternatingMethod.SMALLER_OPEN_SET:
                return self._search_smaller_open_set_first()
            case _:
                return self._search_alternating()

    def _search_alternating(self) -> AlignmentResult:
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

    def _search_lowest_g_first(self) -> AlignmentResult:
        while self.open_set_f and self.open_set_r:
            if self._is_optimal():
                return self._get_alignment()

            if self.open_set_f[0].g <= self.open_set_r[0].g:
                self._forward_search_step()
            else:
                self._reverse_search_step()

        return self._get_alignment()

    def _search_smaller_open_set_first(self) -> AlignmentResult:
        while self.open_set_f and self.open_set_r:
            if self._is_optimal():
                return self._get_alignment()

            if len(self.open_set_f) <= len(self.open_set_r):
                self._forward_search_step()
            else:
                self._reverse_search_step()

        return self._get_alignment()

    def _is_optimal(self) -> bool:
        top_f = self.open_set_f[0].f
        top_r = self.open_set_r[0].f
        match self.stopping_criterion:
            case TerminationCriterion.AVERAGE_FUNCTION:
                return (top_f + top_r) >= (self.mu + self.est_dist_s_t)
            case TerminationCriterion.HALF_AVERAGE_FUNCTION:
                return (top_f + top_r) >= (self.mu + 0.5 * self.est_dist_s_t)
            case _:
                return top_f >= self.mu or top_r >= self.mu

    def _get_alignment(self):
        return AlignmentResult.from_bidirectional(self.mu_forward_state, self.mu_reverse_state, self.visited,
                                                  self.queued, self.traversed, time.time() - self.start_time,
                                                  self.complex_duration, self.simple_lp, self.complex_lp)

    def _forward_search_step(self):
        curr = heapq.heappop(self.open_set_f)

        # ALREADY CLOSED
        if curr.m in self.closed_f and self.closed_f[curr.m].g <= curr.g:
            return

        if not curr.feasible:
            # ORIGINALLY COMPUTED SOLUTION CORRESPONDS TO A NON-REALIZABLE FIRING SEQUENCE
            self.max_event_explained_f = self._get_max_event_not_in_split_list_forward(self.max_event_explained_f)
            if self.max_event_explained_f in self.split_points_f or self.max_event_explained_f >= self.trace_length or self.max_event_explained_f < 0:
                # CONTINUE WITH REGULAR A* SINCE NO NEW SPLIT POINTS
                h, x = compute_exact_heuristic(self.sync_net, self.a_matrix_f, self.h_cvx_f, self.g_matrix_f,
                                               self.cost_matrix_f, self.incidence_matrix_f, curr.m, self.fin_vec_f)
                self.simple_lp += 1
                tp = SplitPointSearchTuple(curr.g, h, curr.m, curr.p, curr.t, x, True, curr.max_event_explained)
                heapq.heappush(self.open_set_f, tp)
            else:
                # RESTART SEARCH WITH ADDITIONAL SPLIT POINT
                self._restart_search_forward()

            return

        # SUCCESSFULLY CLOSED / EXPLORED
        self.closed_f[curr.m] = curr
        self.visited += 1
        self._expand_from_current_marking_forward(curr)

    def _reverse_search_step(self):
        curr = heapq.heappop(self.open_set_r)

        # ALREADY CLOSED
        if curr.m in self.closed_r and self.closed_r[curr.m].g <= curr.g:
            return

        if not curr.feasible:
            # ORIGINALLY COMPUTED SOLUTION CORRESPONDS TO A NON-REALIZABLE FIRING SEQUENCE
            self.max_event_explained_r = self._get_max_event_not_in_split_list_reverse(self.max_event_explained_r)
            if self.max_event_explained_r in self.split_points_f or self.max_event_explained_r >= self.trace_length or self.max_event_explained_r < 0:
                # CONTINUE WITH REGULAR A* SINCE NO NEW SPLIT POINTS
                h, x = compute_exact_heuristic(self.sync_net_r, self.a_matrix_r, self.h_cvx_r, self.g_matrix_r,
                                               self.cost_matrix_r, self.incidence_matrix_r, curr.m, self.fin_vec_r)
                self.simple_lp += 1
                tp = SplitPointSearchTuple(curr.g, h, curr.m, curr.p, curr.t, x, True, curr.max_event_explained)
                heapq.heappush(self.open_set_r, tp)
            else:
                # RESTART SEARCH WITH ADDITIONAL SPLIT POINT
                self._restart_search_reverse()

            return

        # SUCCESSFULLY CLOSED / EXPLORED
        self.closed_r[curr.m] = curr
        self.visited += 1
        self._expand_from_current_marking_reverse(curr)

    def _init_search(self):
        # init forward search
        h_f, x_f = compute_exact_heuristic(self.sync_net, self.a_matrix_f, self.h_cvx_f, self.g_matrix_f,
                                           self.cost_matrix_f, self.incidence_matrix_f, self.initial_marking,
                                           self.fin_vec_f)

        self.est_dist_s_t = h_f

        self.closed_f = {}
        ini_state_f = SplitPointSearchTuple(0, h_f, self.initial_marking, None, None, x_f, True, -1)
        self.open_set_f = [ini_state_f]
        heapq.heapify(self.open_set_f)
        self.scanned_markings_f = {str(self.initial_marking): ini_state_f}

        self.max_event_explained_f = -1
        self.split_points_f = []

        # init reverse search
        h_r, x_r = compute_exact_heuristic(self.sync_net_r, self.a_matrix_r, self.h_cvx_r, self.g_matrix_r,
                                           self.cost_matrix_r, self.incidence_matrix_r, self.initial_marking_r,
                                           self.fin_vec_r)
        self.closed_r = {}
        ini_state_r = SplitPointSearchTuple(0, h_r, self.initial_marking_r, None, None, x_r, True, -1)
        self.open_set_r = [ini_state_r]
        heapq.heapify(self.open_set_r)
        self.scanned_markings_r = {str(self.initial_marking_r): ini_state_r}

        self.max_event_explained_r = -1
        self.split_points_r = []

        self.visited = 0
        self.queued = 0
        self.traversed = 0
        self.simple_lp = 1
        self.complex_lp = 0
        self.complex_duration = 0

    def _get_max_event_not_in_split_list_forward(self, max_event: int):
        if max_event not in self.split_points_f or max_event < 0:
            return max_event
        return self._get_max_event_not_in_split_list_forward(max_event - 1)

    def _get_max_event_not_in_split_list_reverse(self, max_event: int):
        if max_event not in self.split_points_r or max_event < 0:
            return max_event
        return self._get_max_event_not_in_split_list_reverse(max_event - 1)

    def _expand_from_current_marking_forward(self, current: SplitPointSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_f, self.cost_function):
            self._expand_transition_from_marking_forward(current, t, cost)

    def _expand_transition_from_marking_forward(self, current: SplitPointSearchTuple, transition: PetriNet.Transition,
                                                cost: int):
        self.traversed += 1

        new_marking = add_markings(current.m, transition.add_marking)
        new_g = current.g + cost
        h, x, feasible = derive_heuristic_split_point(self.incidence_matrix_f, cost, current.x, transition, current.h)

        if new_marking in self.closed_f and (self.closed_f[new_marking].g <= new_g or not feasible):
            return

        # INCREASE MAX EVENTS EXPLAINED
        no_events_explained = current.max_event_explained
        if transition.label[0] != constants.SKIP:
            no_events_explained += 1
            if no_events_explained > self.max_event_explained_f:
                self.max_event_explained_f = no_events_explained

        tp = SplitPointSearchTuple(new_g, h, new_marking, current, transition, x, feasible, no_events_explained)
        heapq.heappush(self.open_set_f, tp)
        self.queued += 1

        new_marking_str = str(new_marking)
        if new_marking_str not in self.scanned_markings_f or self.scanned_markings_f[new_marking_str].f > tp.f:
            self.scanned_markings_f[new_marking_str] = tp

        if new_marking_str in self.scanned_markings_r:
            matching = self.scanned_markings_r[new_marking_str]
            new_mu = new_g + matching.g
            if new_mu < self.mu:
                self.mu = new_mu
                self.mu_forward_state = tp
                self.mu_reverse_state = matching

    def _expand_from_current_marking_reverse(self, current: SplitPointSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_r, self.cost_function_r):
            self._expand_transition_from_marking_reverse(current, t, cost)

    def _expand_transition_from_marking_reverse(self, current: SplitPointSearchTuple, transition: PetriNet.Transition,
                                                cost: int):
        self.traversed += 1

        new_marking = add_markings(current.m, transition.add_marking)
        new_g = current.g + cost
        h, x, feasible = derive_heuristic_split_point(self.incidence_matrix_r, cost, current.x, transition, current.h)

        if new_marking in self.closed_r and (self.closed_r[new_marking].g <= new_g or not feasible):
            return

        # INCREASE MAX EVENTS EXPLAINED
        no_events_explained = current.max_event_explained
        if transition.label[0] != constants.SKIP:
            no_events_explained += 1
            if no_events_explained > self.max_event_explained_r:
                self.max_event_explained_r = no_events_explained

        tp = SplitPointSearchTuple(new_g, h, new_marking, current, transition, x, feasible, no_events_explained)
        heapq.heappush(self.open_set_r, tp)
        self.queued += 1

        new_marking_str = str(new_marking)
        if new_marking_str not in self.scanned_markings_r or self.scanned_markings_r[new_marking_str].f > tp.f:
            self.scanned_markings_r[new_marking_str] = tp

        if new_marking_str in self.scanned_markings_f:
            matching = self.scanned_markings_f[new_marking_str]
            new_mu = new_g + matching.g
            if new_mu < self.mu:
                self.mu = new_mu
                self.mu_forward_state = matching
                self.mu_reverse_state = tp

    def _restart_search_forward(self):
        self.split_points_f = sorted(self.split_points_f + [self.max_event_explained_f])
        split_points = [self.trace_net_mapping_f[s] for s in self.split_points_f]
        complex_lp_start = time.time()
        h, z = compute_exact_heuristics_with_splits(self.ini_vec_f, self.fin_vec_f, self.cost_vec_f,
                                                    self.incidence_matrix_f, self.consumption_matrix_f,
                                                    split_points, self.solver)
        self.complex_duration += time.time() - complex_lp_start
        self.complex_lp += 1
        self.closed_f = {}
        ini_state_f = SplitPointSearchTuple(0, h, self.initial_marking, None, None, z, True, -1)
        self.open_set_f = [ini_state_f]
        self.scanned_markings_f = {str(self.initial_marking): ini_state_f}
        heapq.heapify(self.open_set_f)
        self.max_event_explained_f = -1

    def _restart_search_reverse(self):
        self.split_points_r = sorted(self.split_points_r + [self.max_event_explained_r])
        split_points = [self.trace_net_mapping_r[s] for s in self.split_points_r]
        complex_lp_start = time.time()
        h, z = compute_exact_heuristics_with_splits(self.ini_vec_r, self.fin_vec_r, self.cost_vec_r,
                                                    self.incidence_matrix_r, self.consumption_matrix_r,
                                                    split_points, self.solver)
        self.complex_duration += time.time() - complex_lp_start
        self.complex_lp += 1
        self.closed_r = {}
        ini_state_r = SplitPointSearchTuple(0, h, self.initial_marking_r, None, None, z, True, -1)
        self.open_set_r = [ini_state_r]
        self.scanned_markings_r = {str(self.initial_marking_r): ini_state_r}
        heapq.heapify(self.open_set_r)
        self.max_event_explained_r = -1
