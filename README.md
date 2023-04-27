## Order-Aware Alignments

### Abstract
We present a novel conformance artefact, namely **Order-Aware Alignments**. 
These alignments are a further development of the classic (sequential) alignments and allow to define a cost function on the *move level* and additionally introduce a secondary cost function to *quantify the ordering relations* between moves. With this secondary cost function, it is possible to assign costs to certain dependencies, e.g., it is possible to penalize dependencies that are not synchronous, i.e., originate either from the log or from the model. Note that this approach by definition supports both totally and partially ordered event data.
To compute order-aware alignments efficiently, we implement several computational strategies including bidirectional search.
As a side result, we also implement bidirectional search for classic alignments. 

### Overview
This repository contains the prototype implementation of order-aware alignments as well as the integration of existing search algorithms (e.g., Dijkstra's algorithm, A* algorithm, Split Point-Based search) for computing such alignments.
The theoretical background is part of the MSc thesis **Efficient Computation of Order-Aware Alignments**.

Corresponding author: David Wenderdel ([Mail](mailto:david.wenderdel@rwth-aachen.de?subject=github-order-aware-alignments))

### Repository Structure
Directory `util` contains several helper classes and methods, e.g., for defining the search tuples, the alignment results or to visualize order-aware alignments.
Directory `algo` contains various algorithms for computing alignments. We implemented the following six algorithms:
1. Dijkstra's Algorithm
2. Bidirectional Dijkstra's Algorithm
3. A* Algorithm
4. Bidirectional A* Algorithm
5. Split Point-Based Search 
6. Bidirectional Split Point-Based Search

All algorithms are implemented for both classical alignments (directory `classic`) and order-aware alignments (directory `order-aware`).
In `main.py`, we show an example of how to compute alignments. The following cli command computes order-aware alignments for event log *Bpic12* and process model *Bpic12_90*.

```shell
python main.py compute-order-aware-alignments -p <file-path> -l Bpic12.xes -m Bpic12_90.pnml 
```

### Note
This repository contains code that is partly taken from [pm4py](https://pm4py.fit.fraunhofer.de/).

The algorithm implementations are based on the following publication: B. F. Van Dongen: Efficiently Computing Alignments: Using the Extended Marking Equation (2018).
