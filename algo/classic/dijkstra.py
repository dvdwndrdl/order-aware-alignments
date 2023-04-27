import heapq
import time

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, decorate_transitions_prepostset

from algo.search_algorithm import SearchAlgorithm
from util.alignment_utils import get_transitions_to_visit, AlignmentResult
from util.search_tuples import DijkstraSearchTuple


class Dijkstra(SearchAlgorithm):
    def __init__(self, sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                 cost_function: dict[PetriNet.Transition, int]):
        self.start_time = time.time()

        self.cost_function = cost_function
        self.sync_net = sync_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking

        decorate_transitions_prepostset(self.sync_net)
        decorate_places_preset_trans(self.sync_net)
        self.trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

        self.closed = None
        self.open_set = None
        self.visited = None
        self.queued = None
        self.traversed = None

    def search(self) -> AlignmentResult:
        self._init_search()

        while self.open_set:
            curr = heapq.heappop(self.open_set)

            # FINAL MARKING FOUND
            if curr.m == self.final_marking:
                return AlignmentResult.from_unidirectional(curr, self.visited, self.queued, self.traversed,
                                                           time.time() - self.start_time)

            # CLOSED ALREADY
            if curr.m in self.closed:
                continue

            self.closed.add(curr.m)
            self.visited += 1
            self._expand_from_current_marking(curr)

    def _init_search(self):
        self.closed = set()
        self.open_set = [DijkstraSearchTuple(0, self.initial_marking, None, None)]
        heapq.heapify(self.open_set)

        self.visited = 0
        self.queued = 0
        self.traversed = 0

    def _expand_from_current_marking(self, current: DijkstraSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset, self.cost_function):
            self._expand_transition_from_marking(current, t, cost)

    def _expand_transition_from_marking(self, current: DijkstraSearchTuple, transition: PetriNet.Transition,
                                        cost: int):
        self.traversed += 1
        new_marking = utils.add_markings(current.m, transition.add_marking)

        if new_marking in self.closed:
            return

        tp = DijkstraSearchTuple(current.g + cost, new_marking, current, transition)
        heapq.heappush(self.open_set, tp)
        self.queued += 1
