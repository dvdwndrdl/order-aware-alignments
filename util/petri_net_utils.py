from copy import deepcopy
from typing import Optional, Tuple

import networkx as nx
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from util.constants import PartialOrderMode


def invert_petri_net(petri_net: PetriNet, initial_marking: Marking, final_marking: Marking) \
        -> Tuple[PetriNet, Marking, Marking]:
    """Builds the inverse Petri net for a given net."""
    reverse_sync_net = deepcopy(petri_net)

    for arc in list(reverse_sync_net.arcs):
        petri_utils.add_arc_from_to(arc.target, arc.source, reverse_sync_net)
        petri_utils.remove_arc(reverse_sync_net, arc)

    reverse_final_marking = Marking()
    for marked_place in initial_marking:
        place = _get_place_by_name(reverse_sync_net, marked_place.name)
        reverse_final_marking[place] = 1

    reverse_initial_marking = Marking()
    for marked_place in final_marking:
        place = _get_place_by_name(reverse_sync_net, marked_place.name)
        reverse_initial_marking[place] = 1

    return reverse_sync_net, reverse_final_marking, reverse_initial_marking


def _get_place_by_name(net: PetriNet, place_name: str) -> Optional[PetriNet.Place]:
    for t in net.places:
        if t.name == place_name:
            return t
    return None


def _get_transition_by_name(net: PetriNet, transition_name: str) -> Optional[PetriNet.Transition]:
    for t in net.transitions:
        if t.name == transition_name:
            return t
    return None


def get_partial_order_from_trace(trace: Trace,
                                 mode: PartialOrderMode = PartialOrderMode.NONE,
                                 key_start_timestamp: str = "start_timestamp",
                                 key_end_timestamp: str = "time:timestamp",
                                 activity_key: str = "concept:name") \
        -> tuple[list[tuple[str, str]], list[str], list[str]]:
    events = {e: (e[key_start_timestamp], e[key_end_timestamp]) for e in trace}
    inserted_events = {}

    partial_order = nx.DiGraph()
    for event_idx, (event, (start, end)) in enumerate(events.items()):
        event_id = f"t_{event[activity_key]}_{event_idx}"
        partial_order.add_node(event_id)
        inserted_events[event_id] = (start, end)

        # check for dependencies to already inserted nodes
        for event_id2, (start2, end2) in inserted_events.items():
            if start > end2:
                partial_order.add_edge(event_id, event_id2)

    if mode == PartialOrderMode.REDUCTION:
        partial_order = nx.transitive_reduction(partial_order)
    elif mode == PartialOrderMode.CLOSURE:
        partial_order = nx.transitive_closure(partial_order)

    # find start and end activities
    start_events = [node for node, deg in partial_order.out_degree() if deg == 0]
    end_events = [node for node, deg in partial_order.in_degree() if deg == 0]

    dependencies = [(u, v) for v, u in partial_order.edges]

    return dependencies, start_events, end_events


def get_partial_trace_net_from_trace(trace: Trace,
                                     mode: PartialOrderMode = PartialOrderMode.NONE,
                                     add_artificial_start_and_end: bool = True,
                                     key_start_timestamp: str = "start_timestamp",
                                     key_end_timestamp: str = "time:timestamp",
                                     activity_key: str = "concept:name") \
        -> tuple[PetriNet, Marking, Marking]:
    """Converts a given trace into a Petri net.

    Parameters:
        trace: Trace to be converted into a Petri net.
        mode: Possible values: CLOSURE, REDUCTION, NONE (default)
        add_artificial_start_and_end: Add an artificial start and end activity (if necessary)
        key_start_timestamp: Key of the start timestamp.
        key_end_timestamp: Key of the end timestamp.
        activity_key: Key of the activity.

    Returns:
        Petri net and its initial and final marking.
    """
    events = {f"t_{e[activity_key]}_{i}": e[activity_key] for i, e in enumerate(trace)}

    edges, start_events, end_events = get_partial_order_from_trace(trace, mode, key_start_timestamp, key_end_timestamp)

    if add_artificial_start_and_end:
        # add artificial start if more than one start event
        if len(start_events) > 1:
            events["S"] = None
            edges = edges + [("S", s) for s in start_events]
            start_events = ["S"]

        # add artificial start if more than one start event
        if len(end_events) > 1:
            events["E"] = None
            edges = edges + [(e, "E") for e in end_events]
            end_events = ["E"]

    # create new petri net
    net = PetriNet()
    initial_marking = Marking()
    final_marking = Marking()

    # add events as transitions
    transitions = {}
    for event_id, event in events.items():
        transition = PetriNet.Transition(event_id, event)
        transitions[event_id] = transition
        net.transitions.add(transition)

        # add initial marking if event is start event
        if event_id in start_events:
            p_i = PetriNet.Place(f"p_i{len(initial_marking)}")
            net.places.add(p_i)
            petri_utils.add_arc_from_to(p_i, transition, net)
            initial_marking[p_i] = 1

        # add final marking if event is end event
        if event_id in end_events:
            p_o = PetriNet.Place(f"p_o{len(final_marking)}")
            net.places.add(p_o)
            petri_utils.add_arc_from_to(transition, p_o, net)
            final_marking[p_o] = 1

    # add dependencies as places
    for edge_idx, (source, target) in enumerate(edges):
        place = PetriNet.Place(f"p_{edge_idx}")
        net.places.add(place)
        petri_utils.add_arc_from_to(transitions[source], place, net)
        petri_utils.add_arc_from_to(place, transitions[target], net)

    return net, initial_marking, final_marking


def get_partial_order_relations_from_trace(trace: Trace):
    deps, _, _ = get_partial_order_from_trace(trace, PartialOrderMode.CLOSURE)
    dep_matrix = {}
    for dep_source, dep_target in deps:
        if dep_target not in dep_matrix:
            dep_matrix[dep_target] = [dep_source]
        else:
            dep_matrix[dep_target].append(dep_source)
    return dep_matrix
