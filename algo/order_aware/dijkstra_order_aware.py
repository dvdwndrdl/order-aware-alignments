import heapq
import math
import time

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, decorate_transitions_prepostset

from algo.search_algorithm import SearchAlgorithm
from util.alignment_utils import get_transitions_to_visit, OrderAwareAlignmentResult
from util.constants import DependencyTypes
from util.search_tuples import OrderAwareDijkstraSearchTuple, StateSpaceItem


class DijkstraOrderAware(SearchAlgorithm):
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

        self.open_set: list[OrderAwareDijkstraSearchTuple] = None
        self.explored_state_space: list[StateSpaceItem] = None
        self.mu: float = None
        self.mu_state: OrderAwareDijkstraSearchTuple = None
        self.visited: int = None
        self.traversed: int = None
        self.queued: int = None

    def search(self) -> OrderAwareAlignmentResult:
        self._init_search()

        while self.open_set:
            curr = heapq.heappop(self.open_set)

            if curr.total_cost >= self.mu:
                # OPTIMAL ALIGNMENT FOUND
                return OrderAwareAlignmentResult.from_unidirectional(
                    self.mu_state, self.visited, self.queued, self.traversed, time.time() - self.start_time)

            # update best result so far
            if curr.m == self.final_marking and self.mu > curr.total_cost:
                self.mu = curr.total_cost
                self.mu_state = curr

            self.visited += 1
            self._expand_from_current_marking(curr)

    def _init_search(self):
        initial_search_tuple = OrderAwareDijkstraSearchTuple(0, self.initial_marking, None, None)
        self.open_set = [initial_search_tuple]
        heapq.heapify(self.open_set)
        self.explored_state_space = [initial_search_tuple.state_space_item]

        self.mu = math.inf
        self.queued = 1
        self.visited = 0
        self.traversed = 0

    def _expand_from_current_marking(self, current: OrderAwareDijkstraSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset, self.cost_function):
            self._expand_transition_from_marking(current, t, cost)

    def _expand_transition_from_marking(self, current: OrderAwareDijkstraSearchTuple,
                                        transition: PetriNet.Transition, cost: int):
        tp = OrderAwareDijkstraSearchTuple.from_current_search_tuple(current, transition, cost, self.dep_cost_function)
        self.traversed += 1

        if tp.total_cost >= self.mu or tp.state_space_item in self.explored_state_space:
            return

        self.explored_state_space.append(tp.state_space_item)
        heapq.heappush(self.open_set, tp)
        self.queued += 1
