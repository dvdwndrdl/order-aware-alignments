from abc import ABC, abstractmethod

from util.alignment_utils import AlignmentResult


class SearchAlgorithm(ABC):
    @abstractmethod
    def search(self) -> AlignmentResult:
        pass

    @abstractmethod
    def _init_search(self):
        pass


class BidirectionalSearchAlgorithm(SearchAlgorithm):
    @abstractmethod
    def _forward_search_step(self):
        pass

    @abstractmethod
    def _reverse_search_step(self):
        pass
