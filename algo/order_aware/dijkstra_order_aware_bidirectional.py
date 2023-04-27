import heapq
import math
import time

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, decorate_transitions_prepostset

from algo.search_algorithm import BidirectionalSearchAlgorithm
from util.alignment_utils import get_transitions_to_visit, OrderAwareAlignmentResult
from util.constants import DependencyTypes, AlternatingMethod
from util.dependency_utils import combine_dependency_matrices, compute_dependency_cost_from_matrix
from util.petri_net_utils import invert_petri_net
from util.search_tuples import OrderAwareDijkstraSearchTuple, StateSpaceItem, HashableDict


class DijkstraOrderAwareBidirectional(BidirectionalSearchAlgorithm):
    def __init__(self, sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                 cost_function: dict[PetriNet.Transition, int], dependency_cost_function: dict[DependencyTypes, int],
                 alternating_method: AlternatingMethod = AlternatingMethod.STRICTLY_ALTERNATE
                 ):
        self.start_time = time.time()
        self.alternating_method = alternating_method

        self.sync_net = sync_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking

        self.cost_function = {trans.name: costs for trans, costs in cost_function.items()}
        self.dep_cost_function = dependency_cost_function

        # build reverse product net
        self.reverse_sync_net, self.reverse_final_marking, self.reverse_initial_marking = \
            invert_petri_net(self.sync_net, self.initial_marking, self.final_marking)

        decorate_transitions_prepostset(self.sync_net)
        decorate_places_preset_trans(self.sync_net)
        self.trans_empty_preset_forward = set(t for t in self.sync_net.transitions if len(t.in_arcs) == 0)

        decorate_transitions_prepostset(self.reverse_sync_net)
        decorate_places_preset_trans(self.reverse_sync_net)
        self.trans_empty_preset_reverse = set(t for t in self.reverse_sync_net.transitions if len(t.in_arcs) == 0)

        # init forward search
        self.open_set_f: list[OrderAwareDijkstraSearchTuple] = None
        self.scanned_f: dict[frozenset, dict[HashableDict, OrderAwareDijkstraSearchTuple]] = None
        self.explored_state_space_f: list[StateSpaceItem] = None

        # init reverse search
        self.open_set_r: list[OrderAwareDijkstraSearchTuple] = None
        self.scanned_r: dict[frozenset, dict[HashableDict, OrderAwareDijkstraSearchTuple]] = None
        self.explored_state_space_r: list[StateSpaceItem] = None

        # init counter
        self.visited: int = None
        self.traversed: int = None
        self.queued: int = None
        self.dep_matrix_duration: int = None

        self.mu: float = None
        self.mu_forward_state: OrderAwareDijkstraSearchTuple = None
        self.mu_reverse_state: OrderAwareDijkstraSearchTuple = None

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

    def _search_lowest_g_first(self) -> OrderAwareAlignmentResult:
        while self.open_set_f and self.open_set_r:
            if self._is_optimal():
                return self._get_alignment()

            if self.open_set_f[0].g <= self.open_set_r[0].g:
                self._forward_search_step()
            else:
                self._reverse_search_step()

    def _search_smaller_open_set_first(self) -> OrderAwareAlignmentResult:
        while self.open_set_f and self.open_set_r:
            if self._is_optimal():
                return self._get_alignment()

            if len(self.open_set_f) <= len(self.open_set_r):
                self._forward_search_step()
            else:
                self._reverse_search_step()

    def _is_optimal(self) -> bool:
        top_f = self.open_set_f[0].total_cost
        top_r = self.open_set_r[0].total_cost
        return top_f + top_r >= self.mu

    def _get_alignment(self) -> OrderAwareAlignmentResult:
        return OrderAwareAlignmentResult.from_bidirectional(
            self.mu_forward_state, self.mu_reverse_state, self.visited, self.queued, self.traversed,
            time.time() - self.start_time, self.dep_matrix_duration, dep_cost_function=self.dep_cost_function)

    def _init_search(self):
        # init forward search
        ini_state_f = OrderAwareDijkstraSearchTuple(0, self.initial_marking, None, None)
        self.open_set_f = [ini_state_f]
        heapq.heapify(self.open_set_f)
        self.scanned_f = {ini_state_f.simple_marking: {ini_state_f.simple_parikh_vector: ini_state_f}}
        self.explored_state_space_f = [ini_state_f.state_space_item]

        # init reverse search
        ini_state_r = OrderAwareDijkstraSearchTuple(0, self.reverse_initial_marking, None, None)
        self.open_set_r = [ini_state_r]
        heapq.heapify(self.open_set_r)
        self.scanned_r = {ini_state_r.simple_marking: {ini_state_r.simple_parikh_vector: ini_state_r}}
        self.explored_state_space_r = [ini_state_r.state_space_item]

        self.mu = math.inf
        self.visited = 0
        self.traversed = 0
        self.queued = 2
        self.dep_matrix_duration = 0

    def _forward_search_step(self):
        curr = heapq.heappop(self.open_set_f)
        self.visited += 1
        self._expand_from_current_marking_forward(curr)

    def _reverse_search_step(self):
        curr = heapq.heappop(self.open_set_r)
        self.visited += 1
        self._expand_from_current_marking_reverse(curr)

    def _expand_from_current_marking_forward(self, current: OrderAwareDijkstraSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_forward, self.cost_function):
            self._expand_transition_from_marking_forward(current, t, cost)

    def _expand_transition_from_marking_forward(self, current: OrderAwareDijkstraSearchTuple,
                                                transition: PetriNet.Transition, cost: int):
        tp = OrderAwareDijkstraSearchTuple.from_current_search_tuple(current, transition, cost, self.dep_cost_function)
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

    def _expand_from_current_marking_reverse(self, current: OrderAwareDijkstraSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_reverse, self.cost_function):
            self._expand_transition_from_marking_reverse(current, t, cost)

    def _expand_transition_from_marking_reverse(self, current: OrderAwareDijkstraSearchTuple,
                                                transition: PetriNet.Transition, cost: int):
        tp = OrderAwareDijkstraSearchTuple.from_current_search_tuple(current, transition, cost, self.dep_cost_function)
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

    def _match_forward_and_reverse_search(self, forward_tuple: OrderAwareDijkstraSearchTuple,
                                          reverse_tuple: OrderAwareDijkstraSearchTuple):
        # more detailed test including the connecting dependencies only if total costs are lower than current mu value
        if forward_tuple.total_cost + reverse_tuple.total_cost >= self.mu:
            return

        dep_matrix_start = time.time()
        combined_dep_matrix = combine_dependency_matrices(forward_tuple, reverse_tuple)
        self.dep_matrix_duration += time.time() - dep_matrix_start
        new_mu = forward_tuple.g + reverse_tuple.g + compute_dependency_cost_from_matrix(combined_dep_matrix,
                                                                                         self.dep_cost_function)

        # update best result so far
        if new_mu < self.mu:
            self.mu = new_mu
            self.mu_forward_state = forward_tuple
            self.mu_reverse_state = reverse_tuple
