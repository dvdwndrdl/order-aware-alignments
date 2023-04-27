from util import constants
from util.constants import DependencyTypes


def get_dependency_type(current_type: str, additional_type: str):
    if current_type is None or current_type == DependencyTypes.INDECISIVE.value:
        return additional_type

    if additional_type is None or additional_type == DependencyTypes.INDECISIVE.value \
            or current_type == DependencyTypes.SYNCHRONOUS.value or current_type == additional_type:
        return current_type

    return DependencyTypes.SYNCHRONOUS.value


def compute_dependency_cost_from_matrix(dependency_matrix: dict[str, dict[str, str]],
                                        dependency_cost_function: dict[DependencyTypes, int]):
    flattened_deps = [dep_type for deps in dependency_matrix.values() for dep_type in deps.values()]

    number_of_log_deps = flattened_deps.count(DependencyTypes.LOG.value)
    number_of_model_deps = flattened_deps.count(DependencyTypes.MODEL.value)
    number_of_indecisive_deps = flattened_deps.count(DependencyTypes.INDECISIVE.value)
    number_of_sync_deps = flattened_deps.count(DependencyTypes.SYNCHRONOUS.value)

    return number_of_log_deps * dependency_cost_function[DependencyTypes.LOG] \
        + number_of_model_deps * dependency_cost_function[DependencyTypes.MODEL] \
        + number_of_indecisive_deps * dependency_cost_function[DependencyTypes.INDECISIVE] \
        + number_of_sync_deps * dependency_cost_function[DependencyTypes.SYNCHRONOUS]


def combine_dependency_matrices(forward_state, reverse_state):
    available_tokens = forward_state.available_tokens.copy()
    trans_id = forward_state.l + 1

    # use forward dependency matrix as basis
    dep_matrix = forward_state.get_dependency_matrix().copy()

    # incrementally extend dependency matrix with reverse matrix
    state = reverse_state
    while state and state.t:
        unique_transition_id = f't{trans_id}'

        # extend dependency matrix with new transition
        dep_matrix[unique_transition_id] = {}

        input_places = [arc.target for arc in state.t.out_arcs]
        for place in input_places:  # output arcs of reversed search are input arcs in forward direction
            if len(place.out_arcs) == 0:  # initial marking (final marking of reverse search)
                continue

            # get token's origin and remove from available tokens
            token_origin = available_tokens[place.name]
            del available_tokens[place.name]

            # update dependency from current transition to transition where token originates from
            new_dep_type = DependencyTypes.MODEL.value if place.name[0] == constants.SKIP else DependencyTypes.LOG.value
            dep_matrix[unique_transition_id][token_origin] = get_dependency_type(
                dep_matrix[unique_transition_id].get(token_origin, None), new_dep_type)

            # update transitive dependencies from current transition
            for trans_dep_transition, trans_dep_type in dep_matrix[token_origin].items():
                if trans_dep_type == new_dep_type or trans_dep_type == DependencyTypes.SYNCHRONOUS.value:
                    dep_matrix[unique_transition_id][trans_dep_transition] = \
                        get_dependency_type(dep_matrix[unique_transition_id].get(trans_dep_transition, None),
                                            new_dep_type)
                else:
                    dep_matrix[unique_transition_id][trans_dep_transition] = \
                        get_dependency_type(dep_matrix[unique_transition_id].get(trans_dep_transition, None),
                                            DependencyTypes.INDECISIVE.value)

        # update available tokens
        for output_place in state.t.in_arcs:
            available_tokens[output_place.source.name] = unique_transition_id

        # get next reverse state and increase unique transition id counter
        state = state.p
        trans_id += 1

    return dep_matrix
