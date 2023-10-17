import argparse

import experiments
import stag.graph


class PerfData(object):
    """
    An object storing performance data for a single run of a
    clustering algorithm.
    """

    def __init__(self, g: stag.graph.Graph,
                 ari: float, nmi: float, t: float,
                 ari_std=None, nmi_std=None,
                 t_std=None):
        """
        :param ari: The ARI of the returned clustering.
        :param t: The time taken by the clustering algorithm.
        """
        self.n = g.number_of_vertices()
        self.ari = ari
        self.nmi = nmi
        self.time = t
        self.ari_std = ari_std
        self.nmi_std = nmi_std
        self.t_std = t_std


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument('command', type=str, choices=['plot', 'run'])
    parser.add_argument('experiment', type=str,
                        choices=['fig2a', 'fig2b', 'mnist', 'pen', 'fashion',
                                 'har', 'letter'])
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == 'plot':
        if args.experiment not in ['fig2a', 'fig2b']:
            print("Can only plot SBM experiments. Specify 'fig2a' or 'fig2b'.")
        elif args.experiment == 'fig2a':
            experiments.sbm_plot("grow_k")
        else:
            experiments.sbm_plot("grow_n")
    else:
        if args.experiment == "fig2a":
            experiments.run_sbm_experiment_growing_k()
        elif args.experiment == "fig2b":
            experiments.run_sbm_experiment_growing_n()
        elif args.experiment == "mnist":
            experiments.openml_experiment("mnist_784", t_const=15)
        elif args.experiment == "fashion":
            experiments.openml_experiment("Fashion-MNIST", t_const=15)
        elif args.experiment == "har":
            experiments.openml_experiment("har", t_const=30)
        elif args.experiment == "letter":
            experiments.openml_experiment("letter", t_const=15)
        elif args.experiment == "pen":
            experiments.openml_experiment("pendigits", t_const=30)
        else:
            print("Invalid experiment specified.")


if __name__ == "__main__":
    main()
