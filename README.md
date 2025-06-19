# Fast and Simple Spectral Clustering in Theory and Practice
Fast spectral clustering, described in the NeurIPS'23 paper "Fast and Simple Spectral Clustering in Theory and Practice".

## Background
This paper describes a simple variant of the spectral clustering algorithm based
on embedding the vertices of the graph in log(k) dimensions, rather than the
usual k dimensions.
Furthermore, this embedding can be computed by a simple application of the power
method with the Laplacian matrix of the graph.

## Algorithm 
The algorithm is very simple to implement in python. The following is a complete
implementation of the algorithm, and can be easily modified to work with any
specific application.

```python
import stag.graph
import math
from sklearn.cluster import KMeans
import numpy as np

def fast_spectral_cluster(g: stag.graph.Graph, k: int):
    l = max(2, math.ceil(math.log(k, 2)))
    t = 10 * math.ceil(math.log(g.number_of_vertices() / k, 2))
    M = g.normalised_signless_laplacian()
    Y = np.random.normal(size=(g.number_of_vertices(), l))

    for _ in range(t):
        Y = M @ Y

    Y, _, _ = np.linalg.svd(Y, full_matrices=False)

    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(Y)
    return kmeans.labels_
```

The [stag library](https://staglibrary.io) is a library for working with graph objects. If you have the adjacency matrix, you can either create a stag Graph object with `g = stag.graph.Graph(adjacency_matrix)`, or construct the normalised signless Laplacian with the following method.

```
def signless_laplacian(A):
    n = A.shape[0]
    D = scipy.sparse.diags(A.sum(axis=1).A1)
    D_inv_half = scipy.sparse.diags(1 / numpy.sqrt(D.diagonal()))
    L = D + A
    N = D_inv_half @ L @ D_inv_half
    return M
```

## Reproducing the experiments

### Installing Dependencies
Install the python package dependencies with pip.

```
pip install -r requirements.txt
```

### Running Experiments
Run the experiments with the following command.

```
python main.py run {experiment}
```

where `{experiment}` is one of `fig2a`, `fig2b`, `mnist`, `pen`, `fashion`, `har`, or `letter`.
These correspond to the experiments reported in the paper, where `fig2a` and `fig2b`
are the experiments on the stochastic block model.

### Plotting the results
The figures included in the paper can be generated with the following commands.

```
python main.py plot fig2a
python main.py plot fig2b
```

## Issues

If you have any issues when using this software, please don't hesitate to get in touch. You can contact me using the contact details on [my website](https://pmacg.io) or you can raise an issue on this repository in Github and I will be happy to help.

## Reference

If you use this software in your work, you can cite it as follows.

```
Macgregor, Peter. "Fast and simple spectral clustering in theory and practice." Advances in Neural Information Processing Systems 36 (2023): 34410-34425.
```

If you use bibtex, you can copy the following into your bibtex file.

```
@inproceedings{macgregorFastSpectralClustering2023,
  title = {Fast and Simple Spectral Clustering in Theory and Practice},
  booktitle = {36th {{Advances}} in {{Neural Information Processing Systems}} ({{NeurIPS}}'23)},
  author = {Macgregor, Peter},
  year = {2023}
}
```