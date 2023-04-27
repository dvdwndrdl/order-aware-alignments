import heapq
import math
import time

from cvxopt import matrix
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, decorate_transitions_prepostset

from algo.search_algorithm import BidirectionalSearchAlgorithm
from util.alignment_utils import get_transitions_to_visit, OrderAwareAlignmentResult
from util.constants import TerminationCriterion, DependencyTypes, AlternatingMethod
from util.dependency_utils import combine_dependency_matrices, compute_dependency_cost_from_matrix
from util.heuristic_utils import vectorize_initial_final_cost, compute_exact_heuristic, derive_heuristic, \
    is_solution_feasible, vectorize_matrices
from util.petri_net_utils import invert_petri_net
from util.search_tuples import OrderAwareSearchTuple, StateSpaceItem, HashableDict


class AStarOrderAwareBidirectional(BidirectionalSearchAlgorithm):
    def __init__(self, sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                 cost_function: dict[PetriNet.Transition, int],
                 dependency_cost_function: dict[DependencyTypes, int],
                 alternating_method: AlternatingMethod = AlternatingMethod.STRICTLY_ALTERNATE,
                 variant: TerminationCriterion = TerminationCriterion.SYMMETRIC_LOWER_BOUNDING):
        self.start_time = time.time()
        self.stopping_criterion = variant
        self.alternating_method = alternating_method

        self.sync_net = sync_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking

        # build reverse product net
        self.sync_net_r, self.final_marking_r, self.initial_marking_r = \
            invert_petri_net(self.sync_net, self.initial_marking, self.final_marking)

        # build reverse cost function
        self.dep_cost_function = dependency_cost_function
        self.cost_function_f = cost_function
        cost_function_str = {str(t): c for t, c in cost_function.items()}
        self.cost_function_r = {t: cost_function_str[str(t)] for t in self.sync_net_r.transitions}

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

        self.open_set_f: list[OrderAwareSearchTuple] = None
        self.scanned_f: dict[frozenset, dict[HashableDict, OrderAwareSearchTuple]] = None
        self.explored_state_space_f: list[StateSpaceItem] = None

        self.open_set_r: list[OrderAwareSearchTuple] = None
        self.scanned_r: dict[frozenset, dict[HashableDict, OrderAwareSearchTuple]] = None
        self.explored_state_space_r: list[StateSpaceItem] = None

        self.visited: int = None
        self.queued: int = None
        self.traversed: int = None
        self.lp_solved: int = None
        self.dep_matrix_duration: int = None

        self.mu = math.inf
        self.mu_forward_state: OrderAwareSearchTuple = None
        self.mu_reverse_state: OrderAwareSearchTuple = None

        self.est_dist_s_t: float = None

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
            time.time() - self.start_time, self.dep_matrix_duration, self.lp_solved,
            dep_cost_function=self.dep_cost_function)

    def _init_search(self):
        # init forward search
        h_f, x_f = compute_exact_heuristic(self.sync_net, self.a_matrix_f, self.h_cvx_f, self.g_matrix_f,
                                           self.cost_vec_f, self.incidence_matrix_f, self.initial_marking,
                                           self.fin_vec_f)

        self.est_dist_s_t = h_f

        ini_state_f = OrderAwareSearchTuple(0, h_f, self.initial_marking, None, None, x_f, True)
        self.open_set_f = [ini_state_f]
        heapq.heapify(self.open_set_f)
        self.scanned_f = {ini_state_f.simple_marking: {ini_state_f.simple_parikh_vector: ini_state_f}}
        self.explored_state_space_f = [ini_state_f.state_space_item]

        # init reverse search
        h_r, x_r = compute_exact_heuristic(self.sync_net_r, self.a_matrix_r, self.h_cvx_r, self.g_matrix_r,
                                           self.cost_vec_r, self.incidence_matrix_r, self.initial_marking_r,
                                           self.fin_vec_r)

        ini_state_r = OrderAwareSearchTuple(0, h_r, self.initial_marking_r, None, None, x_r, True)
        self.open_set_r = [ini_state_r]
        heapq.heapify(self.open_set_r)
        self.scanned_r = {ini_state_r.simple_marking: {ini_state_r.simple_parikh_vector: ini_state_r}}
        self.explored_state_space_r = [ini_state_r.state_space_item]

        # init counter
        self.traversed = 0
        self.visited = 0
        self.queued = 2
        self.lp_solved = 2
        self.dep_matrix_duration = 0

        self.mu = math.inf
        self.mu_forward_state = None
        self.mu_reverse_state = None

    def _forward_search_step(self):
        curr = heapq.heappop(self.open_set_f)

        # REQUEUE IF NON-REALIZABLE FIRING SEQUENCE
        if not curr.feasible:
            h, x = compute_exact_heuristic(self.sync_net, self.a_matrix_f, self.h_cvx_f, self.g_matrix_f,
                                           self.cost_vec_f, self.incidence_matrix_f, curr.m, self.fin_vec_f)
            self.lp_solved += 1
            curr.h = h
            curr.x = x
            curr.feasible = True
            heapq.heappush(self.open_set_f, curr)

            return

        self.visited += 1
        self._expand_from_current_marking_forward(curr)

    def _reverse_search_step(self):
        curr = heapq.heappop(self.open_set_r)

        # REQUEUE IF NON-REALIZABLE FIRING SEQUENCE
        if not curr.feasible:
            h, x = compute_exact_heuristic(self.sync_net_r, self.a_matrix_r, self.h_cvx_r, self.g_matrix_r,
                                           self.cost_vec_r, self.incidence_matrix_r, curr.m, self.fin_vec_r)
            self.lp_solved += 1
            curr.h = h
            curr.x = x
            curr.feasible = True
            heapq.heappush(self.open_set_r, curr)

            return

        self.visited += 1
        self._expand_from_current_marking_reverse(curr)

    def _expand_from_current_marking_forward(self, current: OrderAwareSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_f, self.cost_function_f):
            self._expand_transition_from_marking_forward(current, t, cost)

    def _expand_transition_from_marking_forward(self, current: OrderAwareSearchTuple, transition: PetriNet.Transition,
                                                cost: int):
        h, x = derive_heuristic(self.incidence_matrix_f, self.cost_vec_f, current.x, transition, current.h)
        feasible = is_solution_feasible(x)
        tp = OrderAwareSearchTuple.from_current_search_tuple(
            current, transition, cost, h, x, feasible, self.dep_cost_function)
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

    def _expand_from_current_marking_reverse(self, current: OrderAwareSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_r, self.cost_function_r):
            self._expand_transition_from_marking_reverse(current, t, cost)

    def _expand_transition_from_marking_reverse(self, current: OrderAwareSearchTuple, transition: PetriNet.Transition,
                                                cost: int):
        h, x = derive_heuristic(self.incidence_matrix_r, self.cost_vec_r, current.x, transition, current.h)
        feasible = is_solution_feasible(x)
        tp = OrderAwareSearchTuple.from_current_search_tuple(
            current, transition, cost, h, x, feasible, self.dep_cost_function)
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

    def _match_forward_and_reverse_search(self, forward_tuple: OrderAwareSearchTuple,
                                          reverse_tuple: OrderAwareSearchTuple):
        # more detailed test including the connecting dependencies only if total costs so far are lower than current mu value
        if forward_tuple.total_cost + reverse_tuple.total_cost >= self.mu:
            return

        dep_matrix_start = time.time()
        combined_dep_matrix = combine_dependency_matrices(forward_tuple, reverse_tuple)
        self.dep_matrix_duration += time.time() - dep_matrix_start
        new_mu = forward_tuple.g + reverse_tuple.g + compute_dependency_cost_from_matrix(combined_dep_matrix,
                                                                                         self.dep_cost_function)

        if new_mu < self.mu:
            self.mu = new_mu
            self.mu_forward_state = forward_tuple
            self.mu_reverse_state = reverse_tuple
