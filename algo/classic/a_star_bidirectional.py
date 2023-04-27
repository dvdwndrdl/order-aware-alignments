import heapq
import math
import time

from cvxopt import matrix
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, decorate_transitions_prepostset

from util.alignment_utils import get_transitions_to_visit, AlignmentResult
from util.constants import TerminationCriterion, AlternatingMethod
from util.heuristic_utils import vectorize_initial_final_cost, compute_exact_heuristic, derive_heuristic, \
    is_solution_feasible, vectorize_matrices
from util.petri_net_utils import invert_petri_net
from util.search_tuples import SearchTuple


class AStarBidirectional:
    def __init__(self, sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                 cost_function: dict[PetriNet.Transition, int],
                 stopping_criterion: TerminationCriterion = TerminationCriterion.SYMMETRIC_LOWER_BOUNDING,
                 alternating_method: AlternatingMethod = AlternatingMethod.STRICTLY_ALTERNATE
                 ):
        self.start_time = time.time()
        self.stopping_criterion = stopping_criterion
        self.alternating_method = alternating_method

        self.sync_net = sync_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking

        # build reverse product net
        self.sync_net_r, self.final_marking_r, self.initial_marking_r = invert_petri_net(
            self.sync_net, self.initial_marking, self.final_marking)

        self.cost_function_f = cost_function
        cost_function_str = {str(t): c for t, c in cost_function.items()}
        self.cost_function_r = {}
        for t in self.sync_net_r.transitions:
            self.cost_function_r[t] = cost_function_str[str(t)]

        decorate_transitions_prepostset(self.sync_net)
        decorate_places_preset_trans(self.sync_net)
        self.trans_empty_preset_f = set(t for t in self.sync_net.transitions if len(t.in_arcs) == 0)

        decorate_transitions_prepostset(self.sync_net_r)
        decorate_places_preset_trans(self.sync_net_r)
        self.trans_empty_preset_r = set(t for t in self.sync_net_r.transitions if len(t.in_arcs) == 0)

        # init forward search
        self.incidence_matrix_f = inc_mat_construct(self.sync_net)
        self.ini_vec_f, self.fin_vec_f, self.cost_vec_f = vectorize_initial_final_cost(
            self.incidence_matrix_f, self.initial_marking, self.final_marking, self.cost_function_f)
        self.a_matrix_f, self.g_matrix_f, self.h_cvx_f = vectorize_matrices(self.incidence_matrix_f, self.sync_net)
        self.cost_vec_f = matrix([x * 1.0 for x in self.cost_vec_f])

        # init reverse search
        self.incidence_matrix_r = inc_mat_construct(self.sync_net_r)
        self.ini_vec_r, self.fin_vec_r, self.cost_vec_r = vectorize_initial_final_cost(
            self.incidence_matrix_r, self.initial_marking_r, self.final_marking_r, self.cost_function_r)
        self.a_matrix_r, self.g_matrix_r, self.h_cvx_r = vectorize_matrices(self.incidence_matrix_r, self.sync_net_r)
        self.cost_vec_r = matrix([x * 1.0 for x in self.cost_vec_r])

        self.closed_f = None
        self.open_set_f = None
        self.scanned_markings_f = None

        self.closed_r = None
        self.open_set_r = None
        self.scanned_markings_r = None

        self.visited = None
        self.queued = None
        self.traversed = None
        self.lp_solved = None

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
                                                  simple_lp=self.lp_solved)

    def _init_search(self):
        # init forward search
        h_f, x_f = compute_exact_heuristic(self.sync_net, self.a_matrix_f, self.h_cvx_f, self.g_matrix_f,
                                           self.cost_vec_f, self.incidence_matrix_f, self.initial_marking,
                                           self.fin_vec_f)

        self.est_dist_s_t = h_f

        self.closed_f = set()
        ini_state_f = SearchTuple(0, h_f, self.initial_marking, None, None, x_f, True)
        self.open_set_f = [ini_state_f]
        heapq.heapify(self.open_set_f)
        self.scanned_markings_f = {str(self.initial_marking): ini_state_f}

        # init reverse search
        h_r, x_r = compute_exact_heuristic(self.sync_net_r, self.a_matrix_r, self.h_cvx_r, self.g_matrix_r,
                                           self.cost_vec_r, self.incidence_matrix_r, self.initial_marking_r,
                                           self.fin_vec_r)

        self.closed_r = set()
        ini_state_r = SearchTuple(0, h_r, self.initial_marking_r, None, None, x_r, True)
        self.open_set_r = [ini_state_r]
        heapq.heapify(self.open_set_r)
        self.scanned_markings_r = {str(self.initial_marking_r): ini_state_r}

        # init counter
        self.visited = 0
        self.queued = 0
        self.traversed = 0
        self.lp_solved = 2

        self.mu = math.inf
        self.mu_forward_state = None
        self.mu_reverse_state = None

    def _forward_search_step(self):
        curr = heapq.heappop(self.open_set_f)

        if curr.m in self.closed_f:
            return

        if not curr.feasible:
            h, x = compute_exact_heuristic(self.sync_net, self.a_matrix_f, self.h_cvx_f, self.g_matrix_f,
                                           self.cost_vec_f, self.incidence_matrix_f, curr.m, self.fin_vec_f)
            tp = SearchTuple(curr.g, h, curr.m, curr.p, curr.t, x, True)
            heapq.heappush(self.open_set_f, tp)

            self.lp_solved += 1
            return

        self.closed_f.add(curr.m)
        self.visited += 1
        self._expand_from_current_marking_forward(curr)

    def _reverse_search_step(self):
        curr = heapq.heappop(self.open_set_r)

        if curr.m in self.closed_r:
            return

        if not curr.feasible:
            h, x = compute_exact_heuristic(self.sync_net_r, self.a_matrix_r, self.h_cvx_r, self.g_matrix_r,
                                           self.cost_vec_r, self.incidence_matrix_r, curr.m, self.fin_vec_r)
            tp = SearchTuple(curr.g, h, curr.m, curr.p, curr.t, x, True)
            heapq.heappush(self.open_set_r, tp)

            self.lp_solved += 1
            return

        self.closed_r.add(curr.m)
        self.visited += 1
        self._expand_from_current_marking_reverse(curr)

    def _expand_from_current_marking_forward(self, current: SearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_f, self.cost_function_f):
            self._expand_transition_from_marking_forward(current, t, cost)

    def _expand_transition_from_marking_forward(self, current: SearchTuple, transition: PetriNet.Transition,
                                                 cost: int):
        self.traversed += 1

        new_marking = utils.add_markings(current.m, transition.add_marking)
        if new_marking in self.closed_f:
            return

        h, x = derive_heuristic(self.incidence_matrix_f, self.cost_vec_f, current.x, transition, current.h)
        feasible = is_solution_feasible(x)
        tp = SearchTuple(current.g + cost, h, new_marking, current, transition, x, feasible)
        heapq.heappush(self.open_set_f, tp)
        self.queued += 1

        new_marking_str = str(new_marking)
        if new_marking_str not in self.scanned_markings_f or self.scanned_markings_f[new_marking_str].f > tp.f:
            self.scanned_markings_f[new_marking_str] = tp

        if new_marking_str in self.scanned_markings_r:
            matching = self.scanned_markings_r[new_marking_str]
            new_mu = tp.g + matching.g
            if new_mu < self.mu:
                self.mu = new_mu
                self.mu_forward_state = tp
                self.mu_reverse_state = matching

    def _expand_from_current_marking_reverse(self, current: SearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_r, self.cost_function_r):
            self._expand_transition_from_marking_reverse(current, t, cost)

    def _expand_transition_from_marking_reverse(self, current: SearchTuple, transition: PetriNet.Transition,
                                                 cost: int):
        self.traversed += 1

        new_marking = utils.add_markings(current.m, transition.add_marking)
        if new_marking in self.closed_r:
            return

        h, x = derive_heuristic(self.incidence_matrix_r, self.cost_vec_r, current.x, transition, current.h)
        feasible = is_solution_feasible(x)
        tp = SearchTuple(current.g + cost, h, new_marking, current, transition, x, feasible)
        heapq.heappush(self.open_set_r, tp)
        self.queued += 1

        new_marking_str = str(new_marking)
        if new_marking_str not in self.scanned_markings_r or self.scanned_markings_r[new_marking_str].f > tp.f:
            self.scanned_markings_r[new_marking_str] = tp

        if new_marking_str in self.scanned_markings_f:
            matching = self.scanned_markings_f[new_marking_str]
            new_mu = tp.g + matching.g
            if new_mu < self.mu:
                self.mu = new_mu
                self.mu_forward_state = matching
                self.mu_reverse_state = tp
