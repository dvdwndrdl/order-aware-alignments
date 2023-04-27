from __future__ import annotations

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.align_utils import add_markings

from util.constants import DependencyTypes, DEFAULT_DEPENDENCY_COSTS, SKIP
from util.dependency_utils import get_dependency_type, compute_dependency_cost_from_matrix


class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def copy(self) -> HashableDict:
        return HashableDict(self)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class StateSpaceItem:
    def __init__(self, marking: frozenset[str], parikh_vector: HashableDict):
        self.marking = marking
        self.parikh_vector = parikh_vector

    def __hash__(self):
        return hash((self.marking, self.parikh_vector))

    def __eq__(self, other: StateSpaceItem):
        return (self.marking, self.parikh_vector) == (other.marking, other.parikh_vector)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)


class DijkstraSearchTuple:
    """This class defines a search tuple as required for Dijkstra's algorithm.

    Attributes:
        g (int): Cost so far component
        m (Marking): Marking of the search state
        p (DijkstraSearchTuple | None): Predecessor state
        t (Transition | None): Transition that led to this state
        l (int): Length of the trace so far
        unique_transition_id (str): Unique transition identifier for the process net
    """
    def __init__(self,
                 cost: int,
                 marking: Marking,
                 parent: DijkstraSearchTuple | None,
                 transition: PetriNet.Transition | None):
        self.g = cost
        self.m = marking
        self.p = parent
        self.t = transition
        self.l = self.p.l + 1 if self.p else 0
        self.unique_transition_id = f't{self.l}'

    def get_firing_sequence(self):
        ret = []
        if self.p is not None:
            ret = ret + self.p.get_firing_sequence()
        if self.t is not None:
            ret.append(self.t)
        return ret

    def __lt__(self, other):
        if self.g < other.g:
            return True
        elif other.g < self.g:
            return False
        else:
            return other.l < self.l

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " g=" + str(self.g),
                        " path=" + str(self.get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)


class OrderAwareDijkstraSearchTuple(DijkstraSearchTuple):
    """This class defines a search tuple as required for Dijkstra's algorithm used in Order-Aware Alignments.

    Attributes:
        g (int): Cost so far component (primary cost function)
        m (Marking): Marking of the search state
        p (DijkstraSearchTuple | None): Predecessor state
        t (Transition | None): Transition that led to this state
        l (int): Length of the trace so far
        unique_transition_id (str): Unique transition identifier for the process net
        available_tokens (HashableDict): Dictionary containing the currently available tokens including their origins
        simple_parikh_vector (HashableDict): Parikh vector of the firing sequence that led to this state
        dep_cost (int): Dependency costs (secondary cost function)
        dependencies (dict): Dependencies for this state
    """
    def __init__(self,
                 cost: int,
                 marking: Marking,
                 parent: OrderAwareDijkstraSearchTuple | None,
                 transition: PetriNet.Transition | None,
                 available_tokens: HashableDict = None,
                 dep_cost: int = 0):
        super().__init__(cost, marking, parent, transition)
        if available_tokens is None:
            available_tokens = HashableDict()

        if parent:
            self.simple_parikh_vector = parent.simple_parikh_vector.copy()
        else:
            self.simple_parikh_vector = HashableDict()

        if transition:
            if transition.name in self.simple_parikh_vector:
                self.simple_parikh_vector[transition.name] += 1
            else:
                self.simple_parikh_vector[transition.name] = 1

        self.available_tokens = available_tokens
        self.dep_cost = dep_cost
        self.dependencies = {}

    @property
    def state_space_item(self) -> StateSpaceItem:
        return StateSpaceItem(self.simple_marking, self.simple_parikh_vector)

    @property
    def simple_marking(self) -> frozenset[str]:
        return frozenset([m.name for m in self.m.keys()])

    @property
    def total_cost(self):
        return self.g + self.dep_cost

    @classmethod
    def from_current_search_tuple(cls, current: OrderAwareDijkstraSearchTuple, transition: PetriNet.Transition,
                                  transition_cost: int, dep_cost_function: dict[DependencyTypes, int] = None):
        if dep_cost_function is None:
            dep_cost_function = DEFAULT_DEPENDENCY_COSTS

        new_marking = add_markings(current.m, transition.add_marking)
        tp = cls(current.g + transition_cost, new_marking, current, transition, current.available_tokens.copy())
        tp = _update_search_tuple_dep_matrix(tp, dep_cost_function)

        return tp

    def get_dependency_matrix(self):
        dep_matrix = {}
        tp = self
        while tp.p:
            dep_matrix[tp.unique_transition_id] = tp.dependencies
            tp = tp.p
        return dep_matrix

    def __lt__(self, other: OrderAwareDijkstraSearchTuple):
        if self.total_cost < other.total_cost:
            return True
        elif other.total_cost < self.total_cost:
            return False
        else:
            return other.l < self.l


class SearchTuple(DijkstraSearchTuple):
    """This class defines a search tuple as required for A* algorithm.

    Attributes:
        g (int): Cost so far component
        h (int): Heuristic value
        x (list[float]): Solution vector of the marking equation
        feasible (bool): Defines whether a solution is feasible or not
        m (Marking): Marking of the search state
        p (DijkstraSearchTuple | None): Predecessor state
        t (Transition | None): Transition that led to this state
        l (int): Length of the trace so far
        unique_transition_id (str): Unique transition identifier for the process net
    """
    def __init__(self,
                 g: int,
                 h: int,
                 m: Marking,
                 p: SearchTuple | None,
                 t: PetriNet.Transition | None,
                 x: list[float],
                 feasible: bool):
        super().__init__(g, m, p, t)
        self.h = h
        self.x = x
        self.feasible = feasible

    @property
    def f(self):
        return self.g + self.h

    def __lt__(self, other: SearchTuple):
        if self.f < other.f:
            return True
        if other.f < self.f:
            return False
        if self.feasible and not other.feasible:
            return True
        if not self.feasible and other.feasible:
            return False
        if self.g < other.g:
            return True
        if other.g < self.g:
            return False
        return self.h < other.h

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " f=" + str(self.f), ' g=' + str(self.g), " h=" + str(self.h),
                        " path=" + str(self.get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)


class OrderAwareSearchTuple(OrderAwareDijkstraSearchTuple):
    """This class defines a search tuple as required for A* algorithm used in Order-Aware Alignments.

    Attributes:
        g (int): Cost so far component (primary cost function)
        h (int): Heuristic value
        x (list[float]): Solution vector of the marking equation
        feasible (bool): Defines whether a solution is feasible or not
        m (Marking): Marking of the search state
        p (DijkstraSearchTuple | None): Predecessor state
        t (Transition | None): Transition that led to this state
        l (int): Length of the trace so far
        unique_transition_id (str): Unique transition identifier for the process net
        available_tokens (HashableDict): Dictionary containing the currently available tokens including their origins
        simple_parikh_vector (HashableDict): Parikh vector of the firing sequence that led to this state
        dep_cost (int): Dependency costs (secondary cost function)
        dependencies (dict): Dependencies for this state
    """
    def __init__(self,
                 g: int,
                 h: int,
                 m: Marking,
                 p: OrderAwareSearchTuple | None,
                 t: PetriNet.Transition | None,
                 x: list[float],
                 feasible: bool,
                 dep_cost: int = 0,
                 available_tokens: HashableDict = None):
        super().__init__(g, m, p, t, available_tokens, dep_cost)
        self.h = h
        self.x = x
        self.feasible = feasible

    @property
    def f(self):
        return self.g + self.h

    @property
    def total_cost(self):
        return self.f + self.dep_cost

    @property
    def total_cost_so_far(self):
        return self.g + self.dep_cost

    @classmethod
    def from_current_search_tuple(cls, current: OrderAwareSearchTuple, transition: PetriNet.Transition,
                                  transition_cost: int, h: int, x: list[float], feasible: bool,
                                  dep_cost_function: dict[DependencyTypes, int] = None):
        if dep_cost_function is None:
            dep_cost_function = DEFAULT_DEPENDENCY_COSTS

        new_marking = add_markings(current.m, transition.add_marking)
        tp = cls(current.g + transition_cost, h, new_marking, current, transition, x, feasible,
                 available_tokens=current.available_tokens.copy())
        tp = _update_search_tuple_dep_matrix(tp, dep_cost_function)

        return tp

    def __lt__(self, other: OrderAwareSearchTuple):
        if self.total_cost < other.total_cost:
            return True
        if other.total_cost < self.total_cost:
            return False
        if self.f < other.f:
            return True
        if other.f < self.f:
            return False
        if self.feasible and not other.feasible:
            return True
        if not self.feasible and other.feasible:
            return False
        return self.h < other.h

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " f=" + str(self.f), ' g=' + str(self.g), " h=" + str(self.h),
                        " path=" + str(self.get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)


class SplitPointSearchTuple(SearchTuple):
    """This class defines a search tuple as required for Split Point-Based Search.

    Attributes:
        g (int): Cost so far component
        h (int): Heuristic value
        x (list[float]): Solution vector of the marking equation
        feasible (bool): Defines whether a solution is feasible or not
        m (Marking): Marking of the search state
        p (DijkstraSearchTuple | None): Predecessor state
        t (Transition | None): Transition that led to this state
        l (int): Length of the trace so far
        unique_transition_id (str): Unique transition identifier for the process net
        max_event_explained (int): Max event explained so far (possibly new split point)
    """
    def __init__(self,
                 g: int,
                 h: int,
                 m: Marking,
                 p: SplitPointSearchTuple | None,
                 t: PetriNet.Transition | None,
                 x: list[float],
                 feasible: bool,
                 max_event_explained: int = 0):
        super().__init__(g, h, m, p, t, x, feasible)
        self.max_event_explained = max_event_explained

    def __lt__(self, other: SplitPointSearchTuple):
        # even though the paper says that we prioritize feasible markings, we would get non-optimal results if we use
        # feasibility as first criterion here, because we might stop when we found a final marking, although we still
        # have a cheaper option
        if self.f < other.f:
            return True
        if other.f < self.f:
            return False
        if self.feasible and not other.feasible:
            return True
        if not self.feasible and other.feasible:
            return False
        if self.g < other.g:
            return True
        if other.g < self.g:
            return False
        if self.max_event_explained > other.max_event_explained:
            return True
        if other.max_event_explained > self.max_event_explained:
            return False
        return self.h < other.h


class OrderAwareSplitPointSearchTuple(OrderAwareSearchTuple):
    """This class defines a search tuple as required for Split Point-Based Search used in Order-Aware Alignments.

    Attributes:
        g (int): Cost so far component (primary cost function)
        h (int): Heuristic value
        x (list[float]): Solution vector of the marking equation
        feasible (bool): Defines whether a solution is feasible or not
        m (Marking): Marking of the search state
        p (DijkstraSearchTuple | None): Predecessor state
        t (Transition | None): Transition that led to this state
        l (int): Length of the trace so far
        unique_transition_id (str): Unique transition identifier for the process net
        available_tokens (HashableDict): Dictionary containing the currently available tokens including their origins
        simple_parikh_vector (HashableDict): Parikh vector of the firing sequence that led to this state
        dep_cost (int): Dependency costs (secondary cost function)
        dependencies (dict): Dependencies for this state
        events_explained (list[str]): Events explained so far (for split point construction)
    """
    def __init__(self,
                 g: int,
                 h: int,
                 m: Marking,
                 p: OrderAwareSplitPointSearchTuple | None,
                 t: PetriNet.Transition | None,
                 x: list[float],
                 feasible: bool,
                 dep_cost: int = 0,
                 available_tokens: HashableDict = None,
                 events_explained: list[str] = None):
        super().__init__(g, h, m, p, t, x, feasible, dep_cost, available_tokens)
        if events_explained is None:
            events_explained = []
        self.events_explained = events_explained

    @classmethod
    def from_current_search_tuple(cls, current: OrderAwareSplitPointSearchTuple, transition: PetriNet.Transition,
                                  transition_cost: int, h: int, x: list[float], feasible: bool,
                                  events_explained: list[str], dep_cost_function: dict[DependencyTypes, int] = None):
        if dep_cost_function is None:
            dep_cost_function = DEFAULT_DEPENDENCY_COSTS

        new_marking = add_markings(current.m, transition.add_marking)
        tp = cls(current.g + transition_cost, h, new_marking, current, transition, x, feasible,
                 events_explained=events_explained,
                 available_tokens=current.available_tokens.copy())
        tp = _update_search_tuple_dep_matrix(tp, dep_cost_function)

        return tp

    def __lt__(self, other: OrderAwareSplitPointSearchTuple):
        # even though the paper says that we prioritize feasible markings, we would get non-optimal results if we use
        # feasibility as first criterion here, because we might stop when we found a final marking, although we still
        # have a cheaper option
        if self.f < other.f:
            return True
        if other.f < self.f:
            return False
        if self.total_cost < other.total_cost:
            return True
        if other.total_cost < self.total_cost:
            return False
        if self.feasible and not other.feasible:
            return True
        if not self.feasible and other.feasible:
            return False
        if self.g < other.g:
            return True
        if other.g < self.g:
            return False
        return self.h < other.h


def _update_search_tuple_dep_matrix(st: OrderAwareDijkstraSearchTuple,
                                    dependency_cost_function: dict[DependencyTypes, int]) \
        -> OrderAwareDijkstraSearchTuple:
    input_places = [arc.source for arc in st.t.in_arcs]
    dep_matrix = st.get_dependency_matrix()
    for place in input_places:
        if len(place.in_arcs) == 0:  # initial marking
            continue

        # get token's origin and remove from available tokens
        token_origin = st.available_tokens[place.name]
        del st.available_tokens[place.name]

        # update dependency from current transition to transition where token originates from
        new_dep_type = DependencyTypes.MODEL.value if place.name[0] == SKIP else DependencyTypes.LOG.value
        dep_matrix[st.unique_transition_id][token_origin] = \
            get_dependency_type(dep_matrix[st.unique_transition_id].get(token_origin, None), new_dep_type)

        # update transitive dependencies from current transition
        for trans_dep_transition, trans_dep_type in dep_matrix[token_origin].items():
            if trans_dep_type == new_dep_type or trans_dep_type == DependencyTypes.SYNCHRONOUS.value:
                dep_matrix[st.unique_transition_id][trans_dep_transition] = \
                    get_dependency_type(dep_matrix[st.unique_transition_id].get(trans_dep_transition, None),
                                        new_dep_type)
            else:
                dep_matrix[st.unique_transition_id][trans_dep_transition] = \
                    get_dependency_type(dep_matrix[st.unique_transition_id].get(trans_dep_transition, None),
                                        DependencyTypes.INDECISIVE.value)

    # update available tokens
    for out_arc in st.t.out_arcs:
        st.available_tokens[out_arc.target.name] = st.unique_transition_id

    # update dependencies and costs
    st.dep_cost = compute_dependency_cost_from_matrix(dep_matrix, dependency_cost_function)
    st.dependencies = dep_matrix[st.unique_transition_id]

    return st
