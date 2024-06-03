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
    l = min(k, math.ceil(math.log(k, 2)))
    t = 10 * math.ceil(math.log(g.number_of_vertices() / k, 2))
    M = g.normalised_signless_laplacian()
    Y = np.random.normal(size=(g.number_of_vertices(), l))

    # We know the top eigenvector of the normalised laplacian.
    # It doesn't help with clustering, so we will project our power method to
    # be orthogonal to it.
    top_eigvec = np.sqrt(g.degree_matrix().to_scipy() @ np.full((g.number_of_vertices(),), 1))
    norm = np.linalg.norm(top_eigvec)
    if norm > 0:
        top_eigvec /= norm

    for _ in range(t):
        Y = M @ Y

        # Project Y to be orthogonal to the top eigenvector
        for i in range(l):
            Y[:, i] -= (top_eigvec.transpose() @ Y[:, i]) * top_eigvec

    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(Y)
    return kmeans.labels_
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

