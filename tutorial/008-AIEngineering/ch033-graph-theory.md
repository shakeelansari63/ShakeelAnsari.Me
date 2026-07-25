# Graph Theory

> Your data is not a grid of pixels. Your data is a web of relationships. Graph theory is how you navigate it.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 1, Lessons 01-04  
**Time:** ~120 minutes  

## Learning Objectives

- Implement adjacency matrices and adjacency lists for graph representation
- Implement BFS and DFS traversal for connectivity and pathfinding
- Compute PageRank via power iteration on a directed graph
- Explain how graph convolutional networks generalize convolution to graph-structured data

## The Concept

### What Is a Graph?

A graph `G = (V, E)` consists of:
- A set of vertices (nodes) `V`
- A set of edges `E`, where each edge connects two vertices

**Types of graphs:**
- **Undirected:** Edges have no direction. `(u, v)` = `(v, u)`. Friendship networks, collaboration graphs.
- **Directed (digraph):** Edges have a direction. `(u -> v)` is not `(v -> u)`. Web links, Twitter follows, citation graphs.
- **Weighted:** Each edge has a number (weight). Road networks (distance), communication networks (bandwidth).
- **Unweighted:** All edges are equal. Social networks (friendship is binary).

**Special graph structures:**
- **Tree:** Connected, acyclic, undirected graph with no cycles. N nodes, N-1 edges.
- **Bipartite graph:** Vertices can be split into two sets such that all edges go between sets. User-item interactions, matching problems.
- **Complete graph:** Every pair of vertices is connected by an edge. N(N-1)/2 edges for undirected.
- **Sparse graph:** E << V^2. Most real-world graphs are sparse (social networks, web graphs).
- **Dense graph:** E ≈ V^2. Complete or near-complete graphs.

### Graph Representations

**Adjacency Matrix:** An `|V| x |V|` matrix A where `A[i][j] = 1` if there is an edge from i to j, 0 otherwise.

- Undirected: symmetric matrix
- Weighted: store weight instead of 1
- Memory: O(V^2). For V=100,000, that is 10^10 entries. Impractical for sparse graphs.

**Adjacency List:** For each vertex, store a list of its neighbors.

- Memory: O(V + E). For sparse graphs, this is much smaller than adjacency matrix.
- Iterating over neighbors: O(degree). Fast for sparse graphs.
- Checking if edge exists: O(degree). Slower than O(1) of adjacency matrix.

**Edge List:** List of all edges, each as (u, v, weight). Simple, used in graph files.

| Operation | Adjacency Matrix | Adjacency List |
|-----------|-----------------|----------------|
| Memory | O(V^2) | O(V + E) |
| Edge existence check | O(1) | O(degree(v)) |
| Iterate neighbors | O(V) | O(degree(v)) |
| Add edge | O(1) | O(1) |
| Remove edge | O(1) | O(degree(v)) |

For ML on graphs, adjacency lists are the standard for sparse graphs (social networks, citation networks, molecular graphs). Adjacency matrices are used when dense linear algebra operations are needed (spectral methods, graph neural networks).

### Graph Traversal: BFS and DFS

**BFS (Breadth-First Search):** Explore all neighbors at the current depth before going deeper.

```
BFS(start):
  queue = [start]
  visited = {start}
  while queue:
    v = queue.pop(0)
    for neighbor in neighbors(v):
      if neighbor not in visited:
        visited.add(neighbor)
        queue.append(neighbor)
```

Properties:
- Finds shortest path in unweighted graphs
- Order by distance from start
- O(V + E) time
- Applications: shortest path, connected components, web crawling

**DFS (Depth-First Search):** Go as deep as possible before backtracking.

```
DFS(start):
  stack = [start]
  visited = {start}
  while stack:
    v = stack.pop()
    for neighbor in neighbors(v):
      if neighbor not in visited:
        visited.add(neighbor)
        stack.append(neighbor)
```

Properties:
- Does not find shortest path
- Uses stack (or recursion)
- O(V + E) time
- Applications: topological sorting, cycle detection, maze solving, connected components

### Connected Components and Shortest Paths

**Connected components (undirected):** The maximal sets of vertices that are reachable from each other. BFS/DFS from each unvisited vertex finds all components.

**Strongly connected components (directed):** Vertices where there is a path in both directions (v -> u and u -> v). Kosaraju's or Tarjan's algorithm finds them in O(V + E).

**Shortest path in unweighted graphs:** BFS gives the shortest path (minimum edges) in O(V+E).

**Dijkstra's algorithm (weighted, non-negative):** Finds shortest path from one source to all vertices.

```
Dijkstra(start):
  dist = [inf] * V
  dist[start] = 0
  pq = [(0, start)]
  while pq:
    d, v = pq.pop_min()
    if d > dist[v]: continue
    for neighbor, weight in edges[v]:
      new_d = d + weight
      if new_d < dist[neighbor]:
        dist[neighbor] = new_d
        pq.push((new_d, neighbor))
```

O((V + E) * log V) with a binary heap. This is the standard shortest-path algorithm for road networks, routing, and many graph problems.

### PageRank

The algorithm that launched Google. It measures the importance of nodes in a directed graph based on the idea that important nodes are linked to by other important nodes.

The PageRank of node v:

```
PR(v) = (1 - d) / N + d * sum over u in in_neighbors(v) of PR(u) / out_degree(u)
```

where `d` is the damping factor (typically 0.85), representing the probability that a random surfer follows a link rather than jumping to a random page.

**Power iteration:** Start with equal ranks `PR(v) = 1/N`, then iterate until convergence:

```
For each node v:
  new_PR(v) = (1 - d) / N + d * sum over u in in_neighbors(v) of PR(u) / out_degree(u)
```

Properties:
- Converges to a unique stationary distribution
- Guaranteed convergence for d < 1
- Each iteration is O(E) for sparse graphs
- Typically converges in 50-100 iterations for web-scale graphs

PageRank is a special case of eigenvector centrality. The PageRank vector is the dominant eigenvector of the Google matrix `G = (1-d)/N * J + d * A^T * D^(-1)`, where D is the diagonal out-degree matrix.

### Graph Laplacian

The Laplacian matrix `L` of an undirected graph:

```
L = D - A
```

where `D` is the diagonal degree matrix (`D[i][i] = degree(v_i)`) and `A` is the adjacency matrix.

Properties of the Laplacian:
- Symmetric and positive semidefinite
- Has eigenvalues `0 = lambda_1 <= lambda_2 <= ... <= lambda_n`
- The smallest eigenvalue is always 0, with eigenvector (1, 1, ..., 1)
- The number of zero eigenvalues equals the number of connected components
- `lambda_2` (the Fiedler value) measures graph connectivity: larger = better connected

**Normalized Laplacian:**

```
L_sym = D^(-1/2) * L * D^(-1/2) = I - D^(-1/2) * A * D^(-1/2)
```

Used in spectral clustering and graph neural networks because its eigenvalues are bounded between 0 and 2.

**Why the Laplacian matters for ML:**

The Laplacian's eigenvectors give a natural Fourier basis for the graph. Just as the Fourier transform decomposes a signal into frequencies (Fourier basis = eigenvectors of the 1D Laplacian), the graph Fourier transform decomposes a graph signal into the eigenvectors of the graph Laplacian.

This is the foundation of:
- Spectral clustering (use bottom k eigenvectors for clustering)
- Graph Fourier transform
- Spectral graph convolutional networks (ChebNet, GCN)
- Label propagation and semi-supervised learning

### Spectral Clustering

Algorithm:

```
1. Compute the Laplacian L = D - A
2. Compute the k smallest eigenvectors of L
3. Form matrix U where columns are these eigenvectors
4. Each row of U is a k-dimensional embedding of a vertex
5. Run k-means on these row embeddings
```

Why it works: the eigenvectors of the Laplacian provide a low-dimensional embedding of the graph that preserves local neighborhood structure. Points that are close in the graph are close in the embedding. K-means on this embedding finds clusters that correspond to graph partitions with minimal cut weight.

Spectral clustering can find non-convex clusters that k-means alone cannot. It is the standard method for graph clustering and image segmentation.

### Graph Neural Networks (GNNs)

GNNs extend neural networks to graph-structured data. The core idea: each node's representation is computed by aggregating information from its neighbors.

**Message passing framework:**

```
h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) for u in N(v)}))
```

where `h_v^(k)` is the feature vector of node v at layer k, and `N(v)` is the set of neighbors of v.

**Graph Convolutional Network (GCN):**

```
H^(k+1) = sigma(D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2) * H^(k) * W^(k))
```

where `A_tilde = A + I` (adds self-loops), `D_tilde = sum of A_tilde rows`, `H^(k)` is the node feature matrix, `W^(k)` is the weight matrix, and `sigma` is an activation function.

The GCN layer is a spectral filter: it applies a localized first-order approximation of a spectral graph convolution. Each layer aggregates information from immediate neighbors. Stacking k layers gives each node information from its k-hop neighborhood.

**Other GNN variants:**
- **GraphSAGE:** Sample a fixed number of neighbors (scalable to large graphs)
- **GAT (Graph Attention):** Learn attention weights for neighbors (weights depend on node features, not just graph structure)
- **GIN (Graph Isomorphism Network):** Maximally expressive GNN that can distinguish different graph structures

### Applications of Graph Theory in ML

**Social network analysis:** Community detection, influence propagation, recommendation (friends-of-friends).

**Recommendation systems:** User-item bipartite graphs. Collaborative filtering via graph embeddings (Node2Vec, GraphSAGE).

**Molecular property prediction:** Molecules are graphs (atoms = nodes, bonds = edges). GNNs predict properties (solubility, toxicity, drug-target affinity).

**Knowledge graphs:** Entities as nodes, relationships as edges. Link prediction, entity classification, question answering.

**Computer vision:** Scene graphs (objects and their relationships), point cloud processing (3D points as graph), image segmentation (pixels as graph).

**Natural language processing:** Syntactic parse trees, dependency graphs, document citation graphs.

**Physics simulation:** Mesh-based simulations (airflow, structural mechanics). GNNs learn physics simulators.

**Traffic prediction:** Road networks as graphs, traffic sensors as nodes. GNNs predict traffic flow.

## Build It

### Step 1: Graph representation

```python
class Graph:
    def __init__(self, n_vertices, directed=False):
        self.n = n_vertices
        self.directed = directed
        self.adj_list = [[] for _ in range(n_vertices)]

    def add_edge(self, u, v, weight=1):
        self.adj_list[u].append((v, weight))
        if not self.directed:
            self.adj_list[v].append((u, weight))

    def adjacency_matrix(self):
        A = [[0] * self.n for _ in range(self.n)]
        for u in range(self.n):
            for v, w in self.adj_list[u]:
                A[u][v] = w
        return A
```

### Step 2: BFS and DFS

```python
def bfs(graph, start):
    visited = [False] * graph.n
    queue = [start]
    visited[start] = True
    order = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for neighbor, _ in graph.adj_list[v]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    return order

def dfs(graph, start):
    visited = [False] * graph.n
    stack = [start]
    order = []
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            order.append(v)
            for neighbor, _ in graph.adj_list[v]:
                if not visited[neighbor]:
                    stack.append(neighbor)
    return order
```

### Step 3: PageRank

```python
def pagerank(graph, damping=0.85, max_iter=100, tol=1e-6):
    n = graph.n
    ranks = [1.0 / n for _ in range(n)]
    out_degrees = [len(graph.adj_list[i]) for i in range(n)]
    for _ in range(max_iter):
        new_ranks = [(1.0 - damping) / n for _ in range(n)]
        for u in range(n):
            for v, _ in graph.adj_list[u]:
                new_ranks[v] += damping * ranks[u] / out_degrees[u]
        diff = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
        ranks = new_ranks
        if diff < tol:
            break
    return ranks
```

## Use It

The all implementations from `code/graph_theory.py` include complete functions:

```python
import math

class Graph:
    def __init__(self, n_vertices, directed=False):
        self.n = n_vertices
        self.directed = directed
        self.adj_list = [[] for _ in range(n_vertices)]

    def add_edge(self, u, v, weight=1):
        self.adj_list[u].append((v, weight))
        if not self.directed:
            self.adj_list[v].append((u, weight))

    def adjacency_matrix(self):
        A = [[0] * self.n for _ in range(self.n)]
        for u in range(self.n):
            for v, w in self.adj_list[u]:
                A[u][v] = w
        return A

    def degree(self, v):
        return len(self.adj_list[v])

    def neighbors(self, v):
        return [n for n, _ in self.adj_list[v]]

def bfs(graph, start):
    visited = [False] * graph.n
    queue = [start]
    visited[start] = True
    order = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for n, _ in graph.adj_list[v]:
            if not visited[n]:
                visited[n] = True
                queue.append(n)
    return order

def dfs(graph, start):
    visited = [False] * graph.n
    stack = [start]
    order = []
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            order.append(v)
            for n, _ in graph.adj_list[v]:
                if not visited[n]:
                    stack.append(n)
    return order

def connected_components(graph):
    visited = [False] * graph.n
    components = []
    for v in range(graph.n):
        if not visited[v]:
            component = bfs(graph, v)
            components.append(component)
            for u in component:
                visited[u] = True
    return components

def shortest_path_bfs(graph, start, end):
    visited = [False] * graph.n
    parent = [-1] * graph.n
    queue = [start]
    visited[start] = True
    while queue:
        v = queue.pop(0)
        if v == end:
            return reconstruct_path(parent, start, end)
        for n, _ in graph.adj_list[v]:
            if not visited[n]:
                visited[n] = True
                parent[n] = v
                queue.append(n)
    return []

def reconstruct_path(parent, start, end):
    path = []
    v = end
    while v != -1:
        path.append(v)
        v = parent[v]
    path.reverse()
    return path if path[0] == start else []

def dijkstra(graph, start):
    INF = float('inf')
    dist = [INF] * graph.n
    dist[start] = 0
    visited = [False] * graph.n
    pq = [(0, start)]
    while pq:
        pq.sort(key=lambda x: x[0])
        d, v = pq.pop(0)
        if visited[v]:
            continue
        visited[v] = True
        for n, w in graph.adj_list[v]:
            if not visited[n] and d + w < dist[n]:
                dist[n] = d + w
                pq.append((dist[n], n))
    return dist

def pagerank(graph, damping=0.85, max_iter=100, tol=1e-6):
    n = graph.n
    ranks = [1.0 / n] * n
    out_degrees = [len(graph.adj_list[i]) for i in range(n)]
    for _ in range(max_iter):
        new_ranks = [(1.0 - damping) / n] * n
        for u in range(n):
            if out_degrees[u] == 0:
                continue
            for v, _ in graph.adj_list[u]:
                new_ranks[v] += damping * ranks[u] / out_degrees[u]
        diff = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
        ranks = new_ranks
        if diff < tol:
            break
    return ranks

def laplacian_matrix(graph):
    A = graph.adjacency_matrix()
    n = graph.n
    L = [[0] * n for _ in range(n)]
    for i in range(n):
        degree = sum(A[i])
        for j in range(n):
            L[i][j] = -A[i][j]
        L[i][i] = degree
    return L

def degree_matrix(graph):
    n = graph.n
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        D[i][i] = graph.degree(i)
    return D

def normalized_laplacian(graph):
    n = graph.n
    L = laplacian_matrix(graph)
    D = degree_matrix(graph)
    D_inv_sqrt = [[0] * n for _ in range(n)]
    for i in range(n):
        if D[i][i] > 0:
            D_inv_sqrt[i][i] = 1.0 / math.sqrt(D[i][i])
    L_norm = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += D_inv_sqrt[i][k] * L[k][j]
            L_norm[i][j] = s
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += L_norm[i][k] * D_inv_sqrt[k][j]
            result[i][j] = s
    return result
```

## Ship It

This lesson produces `code/graph_theory.py` with graph representation, traversals, shortest paths, PageRank, and Laplacian utilities. These reappear in Phase 3 for clustering, Phase 4 for graph neural networks, and Phase 5 for advanced GNN architectures.

## Exercises

1. **PageRank on a small graph.** Create a directed graph with 6 nodes and edges representing a mini web. Compute PageRank manually (power iteration). Which nodes have the highest rank? Verify the ranks sum to 1.

2. **Graph Laplacian properties.** Create a graph with two disconnected cliques of size 4 each. Compute the Laplacian and its eigenvalues. How many zero eigenvalues do you see? Connect the cliques with one edge and repeat. How does the second smallest eigenvalue change?

3. **BFS vs DFS order.** Create a graph that is a binary tree of depth 4. Compare the order of nodes visited by BFS and DFS. What differences do you observe?

4. **Spectral clustering.** Generate a graph of two interleaving half-moons (like sklearn's make_moons). Use spectral clustering with the normalized Laplacian to separate them. Compare with k-means directly on the 2D coordinates.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vertex/Node | "A point in the graph" | The fundamental unit. Represents an entity (person, page, atom, sensor). |
| Edge | "A connection" | A relationship between two vertices. Can be directed, undirected, weighted, unweighted. |
| Adjacency list | "Neighbor list" | For each vertex, a list of neighbors. Memory O(V+E). Standard for sparse graphs. |
| Adjacency matrix | "Connection matrix" | V x V matrix where A[i][j] = 1 if edge exists. Memory O(V^2). Standard for dense graphs. |
| BFS | "Breadth-first search" | Explore in layers of increasing distance. Finds shortest paths in unweighted graphs. |
| DFS | "Depth-first search" | Explore as deep as possible before backtracking. Used for topological sorting, cycle detection. |
| PageRank | "Google's algorithm" | Eigenvector centrality for directed graphs. PR(v) = sum of PR(u)/outdeg(u) from in-neighbors, damped. |
| Graph Laplacian | "L = D - A" | Degree minus adjacency. Symmetric, positive semidefinite. Eigenvalues reveal graph structure. |
| Spectral clustering | "Laplacian eigenvectors" | Cluster by embedding nodes in Laplacian eigenvectors then running k-means. Handles non-convex clusters. |
| GNN | "Graph Neural Network" | Neural network for graph data. Each node aggregates features from its neighbors. Layer k gives k-hop information. |
| Message passing | "Neighbor aggregation" | The core GNN operation: update each node's features by combining its features with aggregated neighbor features. |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/21-graph-theory)
