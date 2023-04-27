from enum import Enum
from typing import Union

SKIP = ">>"


class DependencyTypes(Enum):
    LOG = 'l'
    MODEL = 'm'
    INDECISIVE = 'i'
    SYNCHRONOUS = 's'


class StandardDependencyCosts(Enum):
    LOG = 10
    MODEL = 10
    INDECISIVE = 0
    SYNCHRONOUS = 0


DEFAULT_DEPENDENCY_COSTS: dict[DependencyTypes, int] = {
    DependencyTypes.LOG: StandardDependencyCosts.LOG.value,
    DependencyTypes.MODEL: StandardDependencyCosts.MODEL.value,
    DependencyTypes.SYNCHRONOUS: StandardDependencyCosts.SYNCHRONOUS.value,
    DependencyTypes.INDECISIVE: StandardDependencyCosts.INDECISIVE.value
}


class PartialOrderMode(Enum):
    NONE = "none"
    CLOSURE = "closure"
    REDUCTION = "reduction"


class SearchAlgorithms(Enum):
    DIJKSTRA = "dijkstra"
    DIJKSTRA_BIDIRECTIONAL = "dijkstra_bidirectional"

    A_STAR = "a_star"
    A_STAR_BIDIRECTIONAL = "a_star_bidirectional"

    SPLIT_POINT = "split_point"
    SPLIT_POINT_BIDIRECTIONAL = "split_point_bidirectional"


class AlternatingMethod(Enum):
    STRICTLY_ALTERNATE = "strictly_alternate"
    SMALLER_OPEN_SET = "smaller_open_set"  # cardinality criterion (Pohl, 1971)
    LOWEST_G_COST = "lowest_g_cost"


class TerminationCriterion(Enum):
    SYMMETRIC_LOWER_BOUNDING = "symmetric_lower_bounding"
    AVERAGE_FUNCTION = "average_function"
    HALF_AVERAGE_FUNCTION = "half_average_function"


class SplitPointSolvers(Enum):
    GUROBI = "GUROBI"
    GLPK_MI = "GLPK_MI"
    CBC_MI = "CBC_MI"
    GLPK_LP = "GLPK_LP"
    CBC_LP = "CBC_LP"


SearchAlgorithmVariants = Union[AlternatingMethod, TerminationCriterion, SplitPointSolvers]
