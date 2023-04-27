import heapq
import math
import time

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, decorate_transitions_prepostset

from algo.search_algorithm import SearchAlgorithm
from util.alignment_utils import get_transitions_to_visit, AlignmentResult
from util.constants import AlternatingMethod
from util.petri_net_utils import invert_petri_net
from util.search_tuples import DijkstraSearchTuple


class DijkstraBidirectional(SearchAlgorithm):
    def __init__(self, sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                 cost_function: dict[PetriNet.Transition, int],
                 alternating_method: AlternatingMethod = AlternatingMethod.STRICTLY_ALTERNATE
                 ):
        self.start_time = time.time()
        self.alternating_method = alternating_method

        self.sync_net = sync_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking

        self.cost_function = {trans.name: costs for trans, costs in cost_function.items()}

        # build reverse product net
        self.reverse_sync_net, self.reverse_final_marking, self.reverse_initial_marking = invert_petri_net(
            self.sync_net, self.initial_marking, self.final_marking)

        decorate_transitions_prepostset(self.sync_net)
        decorate_places_preset_trans(self.sync_net)
        self.trans_empty_preset_forward = set(t for t in self.sync_net.transitions if len(t.in_arcs) == 0)

        decorate_transitions_prepostset(self.reverse_sync_net)
        decorate_places_preset_trans(self.reverse_sync_net)
        self.trans_empty_preset_reverse = set(t for t in self.reverse_sync_net.transitions if len(t.in_arcs) == 0)

        # init forward search
        self.closed_forward = None
        self.open_set_forward = None
        self.scanned_forward = None

        # init reverse search
        self.closed_reverse = None
        self.open_set_reverse = None
        self.scanned_reverse = None

        # init counter
        self.visited = None
        self.queued = None
        self.traversed = None

        self.mu = math.inf
        self.mu_forward_state = None
        self.mu_reverse_state = None

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
        while self.open_set_forward and self.open_set_reverse:
            if is_forward:
                self._forward_search_step()
            else:
                self._reverse_search_step()

            top_f = self.open_set_forward[0].g
            top_r = self.open_set_reverse[0].g
            if top_f + top_r >= self.mu:
                # FINAL MARKING FOUND
                return AlignmentResult.from_bidirectional(self.mu_forward_state, self.mu_reverse_state, self.visited,
                                                          self.queued, self.traversed, time.time() - self.start_time)

            is_forward = not is_forward

    def _search_lowest_g_first(self) -> AlignmentResult:
        while self.open_set_forward and self.open_set_reverse:
            if self.open_set_forward[0].g <= self.open_set_reverse[0].g:
                self._forward_search_step()
            else:
                self._reverse_search_step()

            top_f = self.open_set_forward[0].g
            top_r = self.open_set_reverse[0].g
            if top_f + top_r >= self.mu:
                # FINAL MARKING FOUND
                return AlignmentResult.from_bidirectional(self.mu_forward_state, self.mu_reverse_state, self.visited,
                                                          self.queued, self.traversed, time.time() - self.start_time)

    def _search_smaller_open_set_first(self) -> AlignmentResult:
        while self.open_set_forward and self.open_set_reverse:
            if len(self.open_set_forward) <= len(self.open_set_reverse):
                self._forward_search_step()
            else:
                self._reverse_search_step()

            top_f = self.open_set_forward[0].g
            top_r = self.open_set_reverse[0].g
            if top_f + top_r >= self.mu:
                # FINAL MARKING FOUND
                return AlignmentResult.from_bidirectional(self.mu_forward_state, self.mu_reverse_state, self.visited,
                                                          self.queued, self.traversed, time.time() - self.start_time)

    def _init_search(self):
        # init forward search
        self.closed_forward = set()
        initial_forward_tuple = DijkstraSearchTuple(0, self.initial_marking, None, None)
        self.open_set_forward = [initial_forward_tuple]
        heapq.heapify(self.open_set_forward)
        self.scanned_forward = {str(self.initial_marking): initial_forward_tuple}

        # init reverse search
        self.closed_reverse = set()
        initial_reverse_tuple = DijkstraSearchTuple(0, self.reverse_initial_marking, None, None)
        self.open_set_reverse = [initial_reverse_tuple]
        heapq.heapify(self.open_set_reverse)
        self.scanned_reverse = {str(self.reverse_initial_marking): initial_reverse_tuple}

        self.visited = 0
        self.queued = 0
        self.traversed = 0

        self.mu = math.inf
        self.mu_forward_state = None
        self.mu_reverse_state = None

    def _forward_search_step(self):
        curr = heapq.heappop(self.open_set_forward)

        if curr.m in self.closed_forward:
            return

        self.closed_forward.add(curr.m)
        self.visited += 1
        self._expand_from_current_marking_forward(curr)

    def _reverse_search_step(self):
        curr = heapq.heappop(self.open_set_reverse)

        if curr.m in self.closed_reverse:
            return

        self.closed_reverse.add(curr.m)
        self.visited += 1
        self._expand_from_current_marking_reverse(curr)

    def _expand_from_current_marking_forward(self, current: DijkstraSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_forward, self.cost_function):
            self._expand_transition_from_marking_forward(current, t, cost)

    def _expand_transition_from_marking_forward(self, current: DijkstraSearchTuple, transition: PetriNet.Transition,
                                                 cost: int):
        self.traversed += 1
        new_marking = utils.add_markings(current.m, transition.add_marking)

        if new_marking in self.closed_forward:
            return

        tp = DijkstraSearchTuple(current.g + cost, new_marking, current, transition)
        heapq.heappush(self.open_set_forward, tp)
        self.queued += 1

        new_marking_str = str(new_marking)
        if new_marking_str not in self.scanned_forward or self.scanned_forward[new_marking_str].g > tp.g:
            self.scanned_forward[new_marking_str] = tp

        if new_marking_str not in self.scanned_reverse:
            return

        matching = self.scanned_reverse[new_marking_str]
        new_mu = tp.g + matching.g
        if new_mu < self.mu:
            self.mu = new_mu
            self.mu_forward_state = tp
            self.mu_reverse_state = matching

    def _expand_from_current_marking_reverse(self, current: DijkstraSearchTuple):
        for t, cost in get_transitions_to_visit(current.m, self.trans_empty_preset_reverse, self.cost_function):
            self._expand_transition_from_marking_reverse(current, t, cost)

    def _expand_transition_from_marking_reverse(self, current: DijkstraSearchTuple, transition: PetriNet.Transition,
                                                 cost: int):
        self.traversed += 1
        new_marking = utils.add_markings(current.m, transition.add_marking)

        if new_marking in self.closed_reverse:
            return

        tp = DijkstraSearchTuple(current.g + cost, new_marking, current, transition)
        heapq.heappush(self.open_set_reverse, tp)
        self.queued += 1

        new_marking_str = str(new_marking)
        if new_marking_str not in self.scanned_reverse or self.scanned_reverse[new_marking_str].g > tp.g:
            self.scanned_reverse[new_marking_str] = tp

        if new_marking_str not in self.scanned_forward:
            return

        matching = self.scanned_forward[new_marking_str]
        new_mu = tp.g + matching.g
        if new_mu < self.mu:
            self.mu = new_mu
            self.mu_forward_state = matching
            self.mu_reverse_state = tp
