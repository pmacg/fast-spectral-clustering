"""
Several variants of spectral clustering that we would like to compare.
"""
import numpy as np
import scipy.sparse.linalg
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
import stag.graph
import stag.stag_internal
import math
import time
import random


#########################################
# Helper functions
#########################################
def labels_to_clusters(labels, k=None):
    """Take a list of labels, and return a list of clusters, using the indices"""
    if k is None:
        k = max(labels) + 1

    clusters = [[] for i in range(k)]
    for i, c in enumerate(labels):
        clusters[c].append(i)

    return clusters


def clusters_to_labels(clusters):
    """Take a list of clusters, and return a list of labels"""
    n = sum([len(cluster) for cluster in clusters])
    labels = [0] * n
    for c_idx, cluster in enumerate(clusters):
        for j in cluster:
            labels[j] = c_idx
    return labels


def kmeans(data, k):
    """
    Apply the kmeans algorithm to the given data, and return the labels.
    """
    kmeans_obj = KMeans(n_clusters=k, n_init='auto')
    kmeans_obj.fit(data)
    return [int(x) for x in list(kmeans_obj.labels_)], kmeans_obj.cluster_centers_


#############################################
# Normal Spectral Clustering
#############################################
def spectral_cluster(g: stag.graph.Graph, k: int):
    lap_mat = g.normalised_laplacian()
    _, eigenvectors = scipy.sparse.linalg.eigsh(lap_mat, k, which='SM')
    labels, _ = kmeans(eigenvectors, k)
    return labels


def spectral_cluster_logk(g: stag.graph.Graph, k: int):
    """Normal spectral clustering with only log(k) eigenvectors"""
    logk = math.ceil(math.log(k, 2))
    lap_mat = g.normalised_laplacian()
    _, eigenvectors = scipy.sparse.linalg.eigsh(lap_mat, logk, which='SM')
    labels, _ = kmeans(eigenvectors, k)
    return labels


########################################
# Other fast spectral clustering
########################################
def KASP(X, k, gamma):
    """Fast Spectral Clustering from the paper 'Fast Approximate Spectral Clustering'."""
    n = X.shape[0]
    if gamma is None:
        gamma = n / 50
    l = int(n / gamma)

    start = time.time()
    coarse_labels, cluster_centers = kmeans(X, l)
    end = time.time()
    running_time = end - start

    # Run spectral clustering on cluster_centers
    knn_graph = kneighbors_graph(cluster_centers, n_neighbors=10, mode='connectivity', include_self=False)
    new_adj = scipy.sparse.lil_matrix(knn_graph.shape)
    for i, j in zip(*knn_graph.nonzero()):
        new_adj[i, j] = 1
        new_adj[j, i] = 1
    g = stag.graph.Graph(new_adj)

    start = time.time()
    labels = spectral_cluster(g, k)
    final_labels = [labels[lab] for lab in coarse_labels]
    end = time.time()
    running_time += end - start

    return final_labels


def gaussian_kernel(u, v, sig):
    return (1 / (sig * math.sqrt(2 * math.pi))) * math.exp(- np.linalg.norm(u - v)**2 / sig )


def nystrom_spectral_clustering(X, k, gamma):
    """Fast Spectral Clustering via the Nystrom Method."""
    # Select the parameters from the original paper
    n = X.shape[0]
    r = 50
    l = int(0.2 * n)

    # Sample l vertices from the graph - corresponds to L in the alg
    sampled_vertices = random.sample(range(n), l)

    # Construct A_hat - requires computing the kernel distance
    A_hat = rbf_kernel(X, X[sampled_vertices, :], gamma=gamma)
    col_sums = A_hat.sum(axis=0)
    Delta = scipy.sparse.diags((1 / np.sqrt(col_sums)).tolist()).tocsc()
    row_sums = A_hat.sum(axis=1)
    D = scipy.sparse.diags((1 / np.sqrt(row_sums)).tolist()).tocsc()

    I_hat = scipy.sparse.eye(n).tocsc()[:, sampled_vertices]
    C = I_hat - math.sqrt(l/n) * D @ A_hat @ Delta
    W = C[sampled_vertices, :]

    eigvals, eigvecs = scipy.sparse.linalg.eigsh(W, k=r)
    U_hat = math.sqrt(l / n) * C @ eigvecs @ scipy.sparse.diags(eigvals)

    u, _, _ = scipy.sparse.linalg.svds(U_hat, k=k, which='SM')
    labels, _ = kmeans(u, k)
    return labels


########################################
# Power Method spectral clustering
########################################
def fast_spectral_cluster(g: stag.graph.Graph, k: int, t_const=10):
    l = min(k, math.ceil(math.log(k, 2)))
    t = t_const * math.ceil(math.log(g.number_of_vertices() / k, 2))
    M = g.normalised_signless_laplacian()
    Y = np.random.normal(size=(g.number_of_vertices(), l))
    for _ in range(t):
        Y = M @ Y
    labels, _ = kmeans(Y, k)
    return labels


def spectral_cluster_pm_k(g: stag.graph.Graph, k: int, t_const=None):
    if t_const is None:
        t_const = 2
    logn = t_const * math.ceil(math.log(g.number_of_vertices(), 2))
    signlap = g.normalised_signless_laplacian()
    eigenvectors = np.random.normal(size=(g.number_of_vertices(), k))
    for _ in range(logn):
        eigenvectors = signlap @ eigenvectors

    # Orthogonalise
    singular_vectors, _, _ = np.linalg.svd(eigenvectors, full_matrices=False)

    labels, _ = kmeans(singular_vectors, k)
    return labels
