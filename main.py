from os.path import join

import click as click
from pm4py.objects.log.importer.xes.importer import apply as xes_import
from pm4py.objects.petri_net.importer import importer as petri_importer
from pm4py.objects.petri_net.utils.synchronous_product import construct as construct_synchronous_product

from util.constants import SearchAlgorithms, AlternatingMethod, TerminationCriterion, SplitPointSolvers, \
    PartialOrderMode, SKIP
from util.petri_net_utils import get_partial_trace_net_from_trace, get_partial_order_relations_from_trace
from util.tools import compute_order_aware_alignments_for_sync_product_net, ExecutionVariant
from util.visualization_tools import draw_order_aware_alignment


@click.command()
@click.option('--path', '-p', help='Path to the data directory.')
@click.option('--log', '-l', help='Name of the event log.')
@click.option('--model', '-m', help='Name of the process model.')
def computeorderawarealignments(path: str, log: str, model: str):
    model_net, model_im, model_fm = petri_importer.apply(join(path, model))
    event_log = xes_import(join(path, log))

    variants_to_execute = [
        ExecutionVariant(search_algorithm=SearchAlgorithms.DIJKSTRA),
        ExecutionVariant(search_algorithm=SearchAlgorithms.DIJKSTRA_BIDIRECTIONAL,
                         bidirectional_alternating=AlternatingMethod.STRICTLY_ALTERNATE),
        ExecutionVariant(search_algorithm=SearchAlgorithms.A_STAR),
        ExecutionVariant(search_algorithm=SearchAlgorithms.A_STAR_BIDIRECTIONAL,
                         bidirectional_alternating=AlternatingMethod.STRICTLY_ALTERNATE,
                         bidirectional_termination=TerminationCriterion.SYMMETRIC_LOWER_BOUNDING),
        ExecutionVariant(search_algorithm=SearchAlgorithms.SPLIT_POINT,
                         split_point_solver=SplitPointSolvers.GUROBI),
        ExecutionVariant(search_algorithm=SearchAlgorithms.SPLIT_POINT_BIDIRECTIONAL,
                         bidirectional_alternating=AlternatingMethod.STRICTLY_ALTERNATE,
                         bidirectional_termination=TerminationCriterion.SYMMETRIC_LOWER_BOUNDING,
                         split_point_solver=SplitPointSolvers.GUROBI),
    ]

    for trace_idx, trace in enumerate(event_log, 1):
        # build trace net
        net, im, fm = get_partial_trace_net_from_trace(trace, PartialOrderMode.REDUCTION, False)
        trace_order_relations = get_partial_order_relations_from_trace(trace)

        # build SPN
        sync_prod, sync_im, sync_fm = construct_synchronous_product(net, im, fm, model_net, model_im, model_fm, SKIP)

        # compute alignments
        for var in variants_to_execute:
            a = compute_order_aware_alignments_for_sync_product_net(sync_prod, sync_im, sync_fm, var,
                                                                    trace_dependencies=trace_order_relations)
            print(f"{var}: {a}")
            draw_order_aware_alignment(a, f"alignment-{trace_idx}-{var}", path)


@click.group()
def cli():
    pass


cli.add_command(computeorderawarealignments, "compute-order-aware-alignments")

if __name__ == '__main__':
    cli()
