from __future__ import annotations

from pm4py.objects.petri_net.obj import Marking, PetriNet

from util import constants
from util.constants import DEFAULT_DEPENDENCY_COSTS
from util.dependency_utils import combine_dependency_matrices, compute_dependency_cost_from_matrix
from util.search_tuples import DijkstraSearchTuple, OrderAwareDijkstraSearchTuple


class AlignmentResult:
    def __init__(self, alignment: list, marking: list, alignment_costs: int, visited_states: int, queued_states: int,
                 traversed_arcs: int, total_duration: float = 0, complex_duration: float = 0, simple_lp: int = 0,
                 complex_lp: int = 0, k: int = 0):
        self.alignment = alignment
        self.markings = marking
        self.alignment_costs = alignment_costs
        self.visited_states = visited_states
        self.queued_states = queued_states
        self.traversed_arcs = traversed_arcs
        self.total_duration = total_duration
        self.complex_duration = complex_duration
        self.simple_lp = simple_lp
        self.complex_lp = complex_lp
        self.k = k

    @staticmethod
    def from_unidirectional(state: DijkstraSearchTuple, visited_states: int, queued_states: int, traversed_arcs: int,
                            total_duration: float = 0, complex_duration: float = 0, simple_lp: int = 0,
                            complex_lp: int = 0, k: int = 0) -> AlignmentResult:
        alignment = []
        markings = []

        # reconstruct alignment
        s = state
        while s.p is not None:
            alignment = [(s.unique_transition_id, s.t.label)] + alignment
            markings = [s.m] + markings
            s = s.p

        # append initial marking
        markings = [s.m] + markings

        return AlignmentResult(alignment, markings, state.g, visited_states, queued_states, traversed_arcs,
                               total_duration, complex_duration, simple_lp, complex_lp, k)

    @staticmethod
    def from_bidirectional(forward_state: DijkstraSearchTuple, reverse_state: DijkstraSearchTuple,
                           visited_states: int, queued_states: int, traversed_arcs: int, total_duration: float = 0,
                           complex_duration: float = 0, simple_lp: int = 0, complex_lp: int = 0, k: int = 0) \
            -> AlignmentResult:
        alignment = []
        markings = []

        # forward search
        s = forward_state
        while s.p is not None:
            markings = [s.m] + markings
            alignment = [(s.unique_transition_id, s.t.label)] + alignment
            s = s.p

        # append initial marking
        markings = [s.m] + markings
        trans_id = len(alignment)

        # reverse search
        s = reverse_state
        while s.p is not None:
            trans_id += 1
            markings.append(s.m)
            alignment.append((f't{trans_id}', s.t.label))
            s = s.p

        # append initial and final markings
        markings.append(s.m)
        markings = [s.m] + markings

        return AlignmentResult(alignment, markings, forward_state.g + reverse_state.g, visited_states, queued_states,
                               traversed_arcs, total_duration, complex_duration, simple_lp, complex_lp, k)


class OrderAwareAlignmentResult(AlignmentResult):
    def __init__(self, alignment: list, marking: list, dependency_matrix: dict, move_costs: int,
                 dependency_costs: int, visited_states: int, queued_states: int, traversed_arcs: int,
                 total_duration: float = 0, complex_duration: float = 0, simple_lp: int = 0,
                 complex_lp: int = 0, k: int = 0):
        super().__init__(alignment, marking, move_costs, visited_states, queued_states, traversed_arcs,
                         total_duration, complex_duration, simple_lp, complex_lp, k)
        self.dependency_costs = dependency_costs
        self.dependency_matrix = dependency_matrix

    @property
    def total_cost(self):
        return self.alignment_costs + self.dependency_costs

    @staticmethod
    def from_unidirectional(state: OrderAwareDijkstraSearchTuple,
                            visited_states: int, queued_states: int, traversed_arcs: int, total_duration: float = 0,
                            complex_duration: float = 0, simple_lp: int = 0, complex_lp: int = 0, k: int = 0) \
            -> OrderAwareAlignmentResult:
        alignment = []
        markings = []
        dependency_matrix = state.get_dependency_matrix()

        # reconstruct alignment
        s = state
        while s.p is not None:
            alignment = [(s.unique_transition_id, s.t.label)] + alignment
            markings = [s.m] + markings
            s = s.p

        # append initial marking
        markings = [s.m] + markings

        return OrderAwareAlignmentResult(alignment, markings, dependency_matrix, state.g, state.dep_cost,
                                         visited_states, queued_states, traversed_arcs, total_duration,
                                         complex_duration, simple_lp, complex_lp, k)

    @staticmethod
    def from_bidirectional(forward_state: OrderAwareDijkstraSearchTuple, reverse_state: OrderAwareDijkstraSearchTuple,
                           visited_states: int, queued_states: int, traversed_arcs: int, total_duration: float = 0,
                           complex_duration: float = 0, simple_lp: int = 0, complex_lp: int = 0, k: int = 0,
                           dep_cost_function=None) \
            -> OrderAwareAlignmentResult:
        if dep_cost_function is None:
            dep_cost_function = DEFAULT_DEPENDENCY_COSTS

        alignment = []
        markings = []

        # reconstruct alignment
        s = forward_state
        while s.p is not None:
            alignment = [(s.unique_transition_id, s.t.label)] + alignment
            markings = [s.m] + markings
            s = s.p

        # append initial marking
        markings = [s.m] + markings
        trans_id = len(alignment)

        # reverse search
        s = reverse_state
        while s.p is not None:
            trans_id += 1
            markings.append(s.m)
            alignment.append((f't{trans_id}', s.t.label))
            s = s.p

        dep_matrix_closure = combine_dependency_matrices(forward_state, reverse_state)
        dep_costs = compute_dependency_cost_from_matrix(dep_matrix_closure, dep_cost_function)

        return OrderAwareAlignmentResult(alignment, markings, dep_matrix_closure, forward_state.g + reverse_state.g,
                                         dep_costs, visited_states, queued_states, traversed_arcs, total_duration,
                                         complex_duration, simple_lp, complex_lp, k)

    def __repr__(self):
        return f"Total Costs: {self.total_cost} | Move Costs: {self.alignment_costs} | Dependency Costs: {self.dependency_costs}"


def is_model_move(t, skip):
    return t.label[0] == skip and t.label[1] != skip


def is_log_move(t, skip):
    return t.label[0] != skip and t.label[1] == skip


def get_transitions_to_visit(current_marking: Marking, trans_empty_preset: set[PetriNet.Transition],
                             cost_function: dict[PetriNet.Transition | str, int]) \
        -> list[tuple[PetriNet.Transition, int]]:
    if isinstance(list(cost_function.keys())[0], tuple):
        return __get_transitions_to_visit_str_cost(current_marking, trans_empty_preset, cost_function)
    return __get_transitions_to_visit(current_marking, trans_empty_preset, cost_function)


def __get_transitions_to_visit(current_marking: Marking, trans_empty_preset: set[PetriNet.Transition],
                               cost_function: dict[PetriNet.Transition, int]) -> list[tuple[PetriNet.Transition, int]]:
    enabled_trans = trans_empty_preset.copy()
    for p in current_marking:
        for t in p.ass_trans:
            if t.sub_marking <= current_marking:
                enabled_trans.add(t)

    return [(t, cost_function[t]) for t in enabled_trans
            if not (t is not None and is_log_move(t, constants.SKIP) and is_model_move(t, constants.SKIP))]


def __get_transitions_to_visit_str_cost(current_marking: Marking, trans_empty_preset: set[PetriNet.Transition],
                                        cost_function: dict[str, int]) -> list[tuple[PetriNet.Transition, int]]:
    enabled_trans = trans_empty_preset.copy()
    for p in current_marking:
        for t in p.ass_trans:
            if t.sub_marking <= current_marking:
                enabled_trans.add(t)

    return [(t, cost_function[t.name]) for t in enabled_trans
            if not (t is not None and is_log_move(t, constants.SKIP) and is_model_move(t, constants.SKIP))]
