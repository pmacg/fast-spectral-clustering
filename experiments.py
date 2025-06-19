from typing import List, Set
import pickle
import time
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.sparse
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import fetch_openml
from scipy.spatial.distance import pdist, squareform

import stag.random
import stag.graph
import stag.cluster
import stag.graphio

import clusteralgs
import main

ALGS_TO_COMPARE = {
    "k eigenvectors": clusteralgs.spectral_cluster,
    "log(k) eigenvectors": clusteralgs.spectral_cluster_logk,
    "log(k) PM": clusteralgs.fast_spectral_cluster,
    "k PM": clusteralgs.spectral_cluster_pm_k,
    "KASP": clusteralgs.KASP,
}


def print_performances(performances):
    for alg_name, perf in performances.items():
        print(f"{alg_name: >20}: "
              f"\ttime: {perf.time: .3f}s +/- {perf.t_std: .3f},"
              f"\tari: {perf.ari: .3f} +/- {perf.ari_std: .3f},"
              f"\tnmi: {perf.nmi: .3f} +/- {perf.nmi_std: .3f}")
    print()


def create_simple_prob_mat(k, p, q):
    prob_mat = []
    for i in range(k):
        new_row = []
        for j in range(k):
            if i == j:
                new_row.append(p)
            else:
                new_row.append(q)
        prob_mat.append(new_row)
    return np.asarray(prob_mat)


def evaluate_one_algorithm(g: stag.graph.Graph,
                           k: int,
                           gt_labels: List[int],
                           method,
                           t_const=None,
                           data=None):
    """
    Evaluate the performance of a single spectral clustering algorithm.

    :param g: the graph to be clusters
    :param k: the number of clusters to find
    :param gt_labels: the ground truth labels
    :param method: the spectral clustering method to be called
    :return: a PerfData object with the results of running the algorithm
    """
    start = time.time()
    if method in [clusteralgs.spectral_cluster_pm_k, clusteralgs.fast_spectral_cluster]:
        labels = method(g, k, t_const=t_const)
    elif method in [clusteralgs.KASP, clusteralgs.nystrom_spectral_clustering]:
        labels = method(data, k, t_const)
    else:
        labels = method(g, k)
    end = time.time()

    running_time = end - start

    ari = stag.cluster.adjusted_rand_index(gt_labels, labels)
    nmi = normalized_mutual_info_score(gt_labels, labels)
    return main.PerfData(g, ari, nmi, running_time)


def compare_algs(g: stag.graph.Graph, k: int, gt_labels: List[int],
                 algs_to_run=None, num_trials=1, t_const=None, data=None):
    """
    Compare the spectral clustering on the given graph.

    Optionally specify a dictionary with boolean values of which algorithms
    to be compared.
    """
    if algs_to_run is None:
        algs_to_run = {alg: True for alg in ALGS_TO_COMPARE.keys()}

    # If the data is not provided, we do not run the KASP algorithm.
    # This is because the KASP algorithm fundamentally requires the 'raw' data
    # and does not operate directly on a graph.
    if data is None and 'KASP' in algs_to_run:
        algs_to_run['KASP'] = False

    performances = {}

    # Initialise the necessary matrices on the stag graph object for fair
    # comparison
    mat = g.normalised_laplacian()
    mat = g.normalised_signless_laplacian()

    for alg_name, method in ALGS_TO_COMPARE.items():
        if alg_name in algs_to_run and algs_to_run[alg_name]:
            print(f"Running method: {alg_name}", end='..')
            aris = []
            nmis = []
            times = []
            for t in range(num_trials):
                print(f".", end='')
                this_perf = evaluate_one_algorithm(g, k, gt_labels, method, t_const=t_const, data=data)
                aris.append(this_perf.ari)
                nmis.append(this_perf.nmi)
                times.append(this_perf.time)
            print()
            performances[alg_name] = main.PerfData(g,
                                                   np.mean(aris),
                                                   np.mean(nmis),
                                                   np.mean(times),
                                                   ari_std=np.std(aris),
                                                   nmi_std=np.std(nmis),
                                                   t_std=np.std(times))

    return performances


def run_sbm_experiment_growing_k():
    # Track whether we should run a certain algorithm
    still_running = {alg: True for alg in ALGS_TO_COMPARE.keys()}

    # Specify a cut-off time after which we will not run
    # an algorithm
    running_time_cutoff_s = 120

    all_performances = {}
    for ka in np.logspace(1, 3, num=30):
        # Create the test graph
        n = 1000
        k = int(ka)
        print(f"**Starting experiment with k = {k}**")
        g = stag.random.sbm(n * k, k, 0.04, 0.001 / k)
        gt_labels = stag.random.sbm_gt_labels(n * k, k)

        performances = compare_algs(g, k, gt_labels, algs_to_run=still_running,
                                    t_const=1)

        print(f"\n Summary for n = {n * k}, k = {k}\n")
        for alg_name, perf in performances.items():
            if perf.time > running_time_cutoff_s:
                still_running[alg_name] = False
            print(f"{alg_name: >25}: \ttime: {perf.time: .5f}s, \tari: {perf.ari: .3f}, \tnmi: {perf.nmi: .3f}")
        print('\n')
        all_performances[k] = performances

    with open("results/sbm/results_grow_k.pickle", 'wb') as fout:
        pickle.dump(all_performances, fout)


def run_sbm_experiment_growing_n():
    # Track whether we should run a certain algorithm
    still_running = {alg: True for alg in ALGS_TO_COMPARE.keys()}

    # Specify a cut-off time after which we will not run
    # an algorithm
    running_time_cutoff_s = 120

    all_performances = {}
    for na in np.logspace(3, 6, num=30):
        # Create the test graph
        k = 20
        n = int(na / k)
        print(f"**Starting experiment with n = {n * k}**")
        g = stag.random.sbm(n * k, k, 40 / n, 1 / (k * n))
        gt_labels = stag.random.sbm_gt_labels(n * k, k)

        performances = compare_algs(g, k, gt_labels, algs_to_run=still_running,
                                    t_const=1)

        print(f"\n Summary for n = {n * k}, k = {k}\n")
        for alg_name, perf in performances.items():
            if perf.time > running_time_cutoff_s:
                still_running[alg_name] = False
            print(f"{alg_name: >25}: \ttime: {perf.time: .5f}s, \tari: {perf.ari: .3f}")
        print('\n')
        all_performances[n * k] = performances

    with open("results/sbm/results_grow_n.pickle", 'wb') as fout:
        pickle.dump(all_performances, fout)


def sbm_plot(which: str, save=False):
    algname_map = {
        'k eigenvectors': '\\textsc{$k$ Eigenvectors}',
        'log(k) eigenvectors': '\\textsc{$\\log(k)$ Eigenvectors}',
        'k PM': '\\textsc{PM $k$ Vectors}',
        'log(k) PM': '\\textsc{PM $\\log(k)$ Vectors}',
    }

    linestyle_map = {
        'k eigenvectors': 'dashed',
        'log(k) eigenvectors': 'dotted',
        'k PM': 'dashed',
        'log(k) PM': 'solid',
    }

    color_map = {
        'log(k) PM': 'red',
        'k eigenvectors': 'blue',
        'log(k) eigenvectors': 'blue',
        'k PM': 'green',
    }

    if which == "grow_k":
        with open("results/sbm/results_grow_k.pickle", 'rb') as fin:
            all_performances = pickle.load(fin)

        # Display the running time results
        fig = plt.figure(figsize=(3.25, 2.75))
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times"
        })
        ax = plt.axes([0.2, 0.16, 0.75, 0.82])
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        for alg_name in linestyle_map:
            data = [(k, v[alg_name]) for k, v in all_performances.items() if alg_name in v]

            plt.plot([k for k, _ in data],
                     [perf.time for _, perf in data],
                     label=algname_map[alg_name],
                     linewidth=3,
                     linestyle=linestyle_map[alg_name],
                     color=color_map[alg_name])

        # plt.legend(loc='best', fontsize=10)
        plt.xlabel('Number of clusters', fontsize=10)
        plt.ylabel('Running time (s)', fontsize=10)
        # plt.xticks([0, 50000, 100000])
        ax.set_ylim(0, 120)
        ax.set_xlim(0, 400)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)

        if save:
            plt.savefig("results/figures/sbm_grow_k_time.pdf")

        plt.show()

    if which == "grow_n":
        with open("results/sbm/results_grow_n.pickle", 'rb') as fin:
            all_performances = pickle.load(fin)

        # Display the running time results
        fig = plt.figure(figsize=(3.25, 2.75))
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times"
        })
        ax = plt.axes([0.2, 0.16, 0.75, 0.82])
        # ax.xaxis.set_major_formatter(
        #     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        for alg_name in linestyle_map:
            data = [(n, v[alg_name]) for n, v in all_performances.items() if alg_name in v]

            plt.plot([n for n, _ in data],
                     [perf.time for _, perf in data],
                     label=algname_map[alg_name],
                     linewidth=3,
                     linestyle=linestyle_map[alg_name],
                     color=color_map[alg_name])

        plt.legend(loc='best', fontsize=10)
        plt.xlabel("Number of vertices", fontsize=10)
        plt.ylabel('Running time (s)', fontsize=10)
        # plt.xticks([0, 50000, 100000])
        # ax.set_ylim(0, 120)
        ax.set_xlim(0, 1000000)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)

        if save:
            plt.savefig("results/figures/sbm_grow_n_time.pdf")

        plt.show()


def preprocess_openml_data(dataset_name: str):
    # Load the graph
    mnist = fetch_openml(dataset_name)
    replace_dict = {chr(i): i-96 for i in range(97, 107)}
    X = np.array(mnist.data.replace(replace_dict))
    target_to_label = {}
    gt_labels = []
    next_label = 0
    for l in list(mnist.target):
        if l not in target_to_label:
            target_to_label[l] = next_label
            next_label += 1
        gt_labels.append(target_to_label[l])
    knn_graph = kneighbors_graph(X, n_neighbors=10, mode='connectivity',
                                 include_self=False)
    new_adj = scipy.sparse.lil_matrix(knn_graph.shape)
    for i, j in zip(*knn_graph.nonzero()):
        new_adj[i, j] = 1
        new_adj[j, i] = 1
    with open(f"data/{dataset_name}.pickle", 'wb') as fout:
        pickle.dump((new_adj, gt_labels), fout)
    with open(f"data/{dataset_name}_data.pickle", 'wb') as fout:
        pickle.dump(X, fout)

def preprocess_data_if_needed(dataset_name):
    """
    Check whether the data file for the given data exists
    already. If not, then call the preprocessing function.
    """
    if not os.path.isfile(f"data/{dataset_name}.pickle"):
        preprocess_openml_data(dataset_name)

def gaussian_kernel_graph(X, sigma):
    # Calculate the pairwise squared Euclidean distances between data points
    pairwise_distances = pdist(X, 'sqeuclidean')
    pairwise_distances_matrix = squareform(pairwise_distances)

    # Calculate the Gaussian kernel similarity
    similarity_matrix = np.exp(-pairwise_distances_matrix / (2 * sigma**2))

    return stag.graph.Graph(scipy.sparse.csc_matrix(similarity_matrix))


def openml_experiment(dataset_name: str, t_const=15):
    preprocess_data_if_needed(dataset_name)

    with open(f"data/{dataset_name}.pickle", 'rb') as fin:
        adj, gt_labels = pickle.load(fin)
    with open(f"data/{dataset_name}_data.pickle", 'rb') as fin:
        X = pickle.load(fin)
    g = stag.graph.Graph(adj)

    # Compare the algorithms
    k = max(gt_labels) + 1
    num_trials = 10
    performances = compare_algs(g, k, gt_labels, num_trials=num_trials,
                                t_const=t_const,
                                data=X)

    print(f"\n Summary for {dataset_name} graph, n = {g.number_of_vertices()}, k = {k}\n")
    print_performances(performances)


def openml_experiment_kasp(dataset_name: str, gamma=None):
    with open(f"data/{dataset_name}_data.pickle", 'rb') as fin:
        X = pickle.load(fin)
    with open(f"data/{dataset_name}.pickle", 'rb') as fin:
        adj, gt_labels = pickle.load(fin)
    g = stag.graph.Graph(adj)

    # Run the KASP algorithm
    k = max(gt_labels) + 1
    num_trials = 10
    performances = compare_algs(g, k, gt_labels, num_trials=num_trials, t_const=gamma,
                                data=X, algs_to_run={'KASP': True})

    print(f"\n Summary for {dataset_name} graph, n = {g.number_of_vertices()}, k = {k}\n")
    print_performances(performances)
