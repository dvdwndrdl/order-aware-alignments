from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.align_utils import construct_standard_cost_function
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net
from pm4py.objects.petri_net.utils.synchronous_product import construct as construct_synchronous_product

from algo.classic import AStar, AStarBidirectional, Dijkstra, DijkstraBidirectional, SplitPoint, SplitPointBidirectional
from algo.order_aware import AStarOrderAware, AStarOrderAwareBidirectional, DijkstraOrderAware, \
    DijkstraOrderAwareBidirectional, SplitPointOrderAware, SplitPointOrderAwareBidirectional
from util.alignment_utils import AlignmentResult, OrderAwareAlignmentResult
from util.constants import SearchAlgorithms, AlternatingMethod, SplitPointSolvers, TerminationCriterion, SKIP, \
    DEFAULT_DEPENDENCY_COSTS


class ExecutionVariant:
    def __init__(self,
                 search_algorithm: SearchAlgorithms,
                 bidirectional_alternating: AlternatingMethod = None,
                 bidirectional_termination: TerminationCriterion = None,
                 split_point_solver: SplitPointSolvers = None):
        self.search_algorithm = search_algorithm
        self.bidirectional_alternating = bidirectional_alternating
        self.bidirectional_termination = bidirectional_termination
        self.split_point_solver = split_point_solver

    def __repr__(self):
        string_build = [self.search_algorithm.value]
        if self.bidirectional_alternating:
            string_build.append(self.bidirectional_alternating.value)
        if self.bidirectional_termination:
            string_build.append(self.bidirectional_termination.value)
        if self.split_point_solver:
            string_build.append(self.split_point_solver.value)
        return "_".join(string_build)


def compute_alignments_for_trace(trace: Trace, petri_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                                 algorithm: ExecutionVariant, cost_function: dict = None) -> AlignmentResult:
    trace_net, trace_im, trace_fm, = construct_trace_net(trace)

    return compute_alignments_for_trace_net(petri_net, initial_marking, final_marking,
                                            trace_net, trace_im, trace_fm, algorithm, cost_function)


def compute_alignments_for_trace_net(petri_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                                     trace_net: PetriNet, trace_im: Marking, trace_fm: Marking,
                                     algorithm: ExecutionVariant, cost_function: dict = None) -> AlignmentResult:
    sync_prod, sync_initial_marking, sync_final_marking = construct_synchronous_product(
        trace_net, trace_im, trace_fm, petri_net, initial_marking, final_marking, SKIP)

    return compute_alignments_for_sync_product_net(sync_prod, sync_initial_marking, sync_final_marking,
                                                   algorithm, cost_function)


def compute_alignments_for_sync_product_net(sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
                                            algorithm: ExecutionVariant, cost_function: dict = None) -> AlignmentResult:
    if cost_function is None:
        cost_function = construct_standard_cost_function(sync_net, SKIP)

    match algorithm.search_algorithm:
        case SearchAlgorithms.DIJKSTRA:
            search_algorithm = Dijkstra(sync_net, initial_marking, final_marking, cost_function)
        case SearchAlgorithms.DIJKSTRA_BIDIRECTIONAL:
            search_algorithm = DijkstraBidirectional(sync_net, initial_marking, final_marking, cost_function,
                                                     algorithm.bidirectional_alternating)
        case SearchAlgorithms.A_STAR:
            search_algorithm = AStar(sync_net, initial_marking, final_marking, cost_function)
        case SearchAlgorithms.A_STAR_BIDIRECTIONAL:
            search_algorithm = AStarBidirectional(sync_net, initial_marking, final_marking, cost_function,
                                                  algorithm.bidirectional_termination,
                                                  algorithm.bidirectional_alternating)
        case SearchAlgorithms.SPLIT_POINT:
            search_algorithm = SplitPoint(sync_net, initial_marking, final_marking, cost_function,
                                          algorithm.split_point_solver)
        case SearchAlgorithms.SPLIT_POINT_BIDIRECTIONAL:
            search_algorithm = SplitPointBidirectional(sync_net, initial_marking, final_marking, cost_function,
                                                       algorithm.bidirectional_termination,
                                                       algorithm.bidirectional_alternating,
                                                       algorithm.split_point_solver)
        case _:
            raise NotImplementedError(f"Search algorithm {algorithm.search_algorithm.value} not implemented.")

    return search_algorithm.search()


def compute_order_aware_alignments_for_sync_product_net(
        sync_net: PetriNet, initial_marking: Marking, final_marking: Marking,
        algorithm: ExecutionVariant, cost_function: dict = None, dependency_cost_function: dict = None,
        trace_dependencies: dict = None
) -> OrderAwareAlignmentResult:
    if cost_function is None:
        cost_function = construct_standard_cost_function(sync_net, SKIP)

    if dependency_cost_function is None:
        dependency_cost_function = DEFAULT_DEPENDENCY_COSTS

    match algorithm.search_algorithm:
        case SearchAlgorithms.DIJKSTRA:
            search_algorithm = DijkstraOrderAware(sync_net, initial_marking, final_marking, cost_function,
                                                  dependency_cost_function)
        case SearchAlgorithms.DIJKSTRA_BIDIRECTIONAL:
            search_algorithm = DijkstraOrderAwareBidirectional(sync_net, initial_marking, final_marking, cost_function,
                                                               dependency_cost_function,
                                                               algorithm.bidirectional_alternating)
        case SearchAlgorithms.A_STAR:
            search_algorithm = AStarOrderAware(sync_net, initial_marking, final_marking, cost_function,
                                               dependency_cost_function)
        case SearchAlgorithms.A_STAR_BIDIRECTIONAL:
            search_algorithm = AStarOrderAwareBidirectional(sync_net, initial_marking, final_marking, cost_function,
                                                            dependency_cost_function,
                                                            algorithm.bidirectional_alternating,
                                                            algorithm.bidirectional_termination)
        case SearchAlgorithms.SPLIT_POINT:
            search_algorithm = SplitPointOrderAware(sync_net, initial_marking, final_marking, cost_function,
                                                    dependency_cost_function, algorithm.split_point_solver,
                                                    trace_dependencies)
        case SearchAlgorithms.SPLIT_POINT_BIDIRECTIONAL:
            search_algorithm = SplitPointOrderAwareBidirectional(sync_net, initial_marking, final_marking,
                                                                 trace_dependencies, cost_function,
                                                                 dependency_cost_function,
                                                                 algorithm.bidirectional_alternating,
                                                                 algorithm.bidirectional_termination,
                                                                 algorithm.split_point_solver)
        case _:
            raise NotImplementedError(f"Search algorithm {algorithm.search_algorithm.value} not implemented.")

    return search_algorithm.search()
