// g++ -fopenmp example.cpp -o example
// ./example

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

// Define a structure for an undirected graph
class Graph {
    int V; // Number of vertices
    vector<vector<int>> adj; // Adjacency list

public:
    // Constructor to initialize the graph with a given number of vertices
    Graph(int V) : V(V) {
        adj.resize(V);
    }

    // Function to add an edge between two vertices
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // Since the graph is undirected
    }

    // Breadth First Search
    void parallelBFS(int start) {
        vector<bool> visited(V, false); // Keep track of visited vertices
        queue<int> q; // Queue for BFS traversal
        q.push(start); // Enqueue the starting vertex
        visited[start] = true; // Mark the starting vertex as visited

        // Perform BFS traversal
        while (!q.empty()) {
            int size = q.size(); // Get the current size of the queue
            #pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                int u;
                #pragma omp critical
                {
                    u = q.front(); // Get the front element of the queue
                    q.pop(); // Dequeue the front element
                }
                cout << u << " "; // Print the current vertex
                // Traverse adjacent vertices of the current vertex
                for (int v : adj[u]) {
                    if (!visited[v]) {
                        #pragma omp critical
                        {
                            visited[v] = true; // Mark the adjacent vertex as visited
                            q.push(v); // Enqueue the adjacent vertex
                        }
                    }
                }
            }
        }
    }

    // Depth First Search
    void parallelDFS(int start) {
        vector<bool> visited(V, false); // Keep track of visited vertices
        stack<int> s; // Stack for DFS traversal
        s.push(start); // Push the starting vertex onto the stack

        // Perform DFS traversal
        #pragma omp parallel
        {
            while (!s.empty()) {
                int u = -1; // Initialize u to an invalid value
                #pragma omp critical
                {
                    if (!s.empty()) {
                        u = s.top(); // Get the top element of the stack
                        s.pop(); // Pop the top element
                    }
                }
                if (u != -1 && !visited[u]) {
                    visited[u] = true; // Mark the current vertex as visited
                    cout << u << " "; // Print the current vertex
                    // Traverse adjacent vertices of the current vertex
                    for (int v : adj[u]) {
                        if (!visited[v]) {
                            #pragma omp critical
                            {
                                s.push(v); // Push the adjacent vertex onto the stack
                            }
                        }
                    }
                }
            }
        }
    }
};

int main() {
    // Create a graph
    int V, E;
    cout << "Enter the number of vertices and edges: ";
    cin >> V >> E;

    Graph graph(V); // Instantiate a graph with V vertices

    // Input edges of the graph
    cout << "Enter the edges (format: source destination):" << endl;
    for (int i = 0; i < E; ++i) {
        int u, v;
        cin >> u >> v;
        graph.addEdge(u, v); // Add the edge between vertices u and v
    }

    // Perform BFS and DFS
    int startVertex;
    cout << "Enter the starting vertex for BFS and DFS: ";
    cin >> startVertex;

    cout << "BFS Traversal: ";
    graph.parallelBFS(startVertex); // Perform parallel BFS traversal
    cout << endl;

    cout << "DFS Traversal: ";
    graph.parallelDFS(startVertex); // Perform parallel DFS traversal
    cout << endl;

    return 0;
}

/*
Depth-First Search (DFS) is a fundamental algorithm used for traversing or searching tree or graph data structures. It starts at a chosen node (often referred to as the "root" in trees) and explores as far as possible along each branch before backtracking. Let's delve into the details of DFS:

### Basic Idea:

1. **Start at a Node**: The algorithm begins at a chosen starting node.

2. **Explore as Far as Possible**: From the current node, DFS explores as far as possible along each branch before backtracking. It prioritizes going deep into the graph or tree rather than broad exploration.

3. **Backtrack when Necessary**: If a dead end is reached or all neighboring nodes have been visited, DFS backtracks to the most recent node with unexplored neighbors and continues.

### Pseudocode:

```
DFS(G, v):
    // G: Graph, v: Current vertex
    
    Mark v as visited
    
    For each neighbor w of v in G:
        If w is not visited:
            DFS(G, w)
```

### Detailed Explanation:

1. **Initialization**: Initialize a set to keep track of visited nodes and start DFS at a specified node.

2. **Explore Neighbors**: For each neighbor of the current node, if the neighbor has not been visited, recursively apply DFS to that neighbor.

3. **Mark Visited**: Mark the current node as visited to avoid revisiting it.

4. **Backtracking**: If all neighbors have been visited or there are no neighbors, backtrack to the previous node.

5. **Termination**: The process continues until all reachable nodes from the starting node have been visited.

### Applications:

1. **Graph Traversal**: DFS can be used to traverse a graph and visit all nodes reachable from a given starting node.

2. **Cycle Detection**: By keeping track of visited nodes, DFS can detect cycles in a graph.

3. **Pathfinding**: DFS can be adapted to find paths between nodes in a graph.

4. **Topological Sorting**: DFS can be used to perform topological sorting of a directed acyclic graph (DAG).

### Time Complexity:

The time complexity of DFS is \(O(V + E)\), where \(V\) is the number of vertices (nodes) and \(E\) is the number of edges in the graph. This is because DFS visits each vertex and edge at most once.

### Space Complexity:

The space complexity of DFS depends on the implementation. In the recursive implementation, the maximum space required on the call stack is proportional to the maximum depth of recursion, which is \(O(V)\) in the worst case for a graph with \(V\) vertices. Additionally, a set to keep track of visited nodes requires \(O(V)\) space.

### Summary:

Depth-First Search (DFS) is a versatile graph traversal algorithm that systematically explores the vertices and edges of a graph or tree. Its simplicity and effectiveness make it a widely used tool in various applications, including graph traversal, cycle detection, pathfinding, and topological sorting.


Certainly! Let's dive into the details of Breadth-First Search (BFS), another fundamental graph traversal algorithm:

### Basic Idea:

Breadth-First Search (BFS) explores a graph by systematically traversing all nodes at the current depth level before moving to the nodes at the next depth level. It starts at a chosen node (often referred to as the "root" in trees) and explores all its neighbors before moving to the next level of neighbors.

### Pseudocode:

```
BFS(G, s):
    // G: Graph, s: Starting node
    
    Initialize a queue Q
    Mark node s as visited and enqueue it into Q
    
    While Q is not empty:
        Dequeue a node v from Q
        For each neighbor w of v in G:
            If w is not visited:
                Mark w as visited and enqueue it into Q
```

### Detailed Explanation:

1. **Initialization**: Initialize a queue to keep track of nodes to be visited and mark the starting node as visited.

2. **Enqueue Starting Node**: Enqueue the starting node into the queue.

3. **Exploration Loop**: While the queue is not empty, dequeue a node from the front of the queue and explore its neighbors.

4. **Mark Visited**: Mark each unvisited neighbor as visited and enqueue it into the queue.

5. **Termination**: The process continues until the queue is empty, indicating that all reachable nodes from the starting node have been visited.

### Applications:

1. **Graph Traversal**: BFS can be used to traverse a graph and visit all nodes reachable from a given starting node.

2. **Shortest Path Finding**: BFS can find the shortest path between two nodes in an unweighted graph.

3. **Connected Components**: BFS can determine the connected components of a graph.

4. **Maze Solving**: BFS can be used to solve maze problems by finding the shortest path from the starting position to the goal.

### Time Complexity:

The time complexity of BFS is \(O(V + E)\), where \(V\) is the number of vertices (nodes) and \(E\) is the number of edges in the graph. This is because BFS visits each vertex and edge exactly once.

### Space Complexity:

The space complexity of BFS depends on the implementation. In the worst case, the space required is proportional to the maximum number of nodes at a single level in the graph, which can be \(O(V)\) in the case of a complete binary tree. Additionally, a queue is used to store nodes to be visited, requiring \(O(V)\) space.

### Summary:

Breadth-First Search (BFS) is a graph traversal algorithm that explores a graph level by level, visiting all nodes at each level before moving to the next level. Its simplicity and ability to find shortest paths in unweighted graphs make it a widely used algorithm in various applications, including graph traversal, shortest path finding, connected component determination, and maze solving.


Parallel Breadth-First Search (BFS) and Parallel Depth-First Search (DFS) are parallelized versions of the traditional BFS and DFS algorithms, respectively, designed to exploit parallelism on multi-core or distributed memory systems. Let's compare them:

### Parallel BFS:

1. **Exploration Strategy**:
   - BFS explores the graph level by level, visiting all nodes at each level before moving to the next level.
   - Parallel BFS maintains multiple frontier sets, each corresponding to a level in the graph. Multiple threads simultaneously explore nodes at different levels in parallel.

2. **Parallelism**:
   - BFS inherently exhibits fine-grained parallelism, as nodes at each level can be explored independently.
   - Parallel BFS can efficiently utilize multi-core processors, with each core responsible for processing a different level of the graph.

3. **Communication**:
   - In parallel BFS, communication between threads is minimal, as threads typically operate independently on different levels of the graph.
   - However, synchronization may be required to coordinate access to shared data structures, such as the frontier queue.

4. **Applications**:
   - Parallel BFS is commonly used in graph algorithms requiring level-wise traversal, such as shortest path finding, connected components, and network analysis.

### Parallel DFS:

1. **Exploration Strategy**:
   - DFS explores the graph depth-first, traversing as far as possible along each branch before backtracking.
   - Parallel DFS divides the search space into subproblems, with different threads exploring different branches of the search tree concurrently.

2. **Parallelism**:
   - DFS exhibits coarse-grained parallelism, as each thread typically explores a different subtree independently.
   - Parallel DFS may be less efficient than parallel BFS on multi-core processors due to potential load imbalances and thread contention.

3. **Communication**:
   - Parallel DFS may require more communication between threads, especially if load balancing techniques or work stealing are employed to distribute work among threads.

4. **Applications**:
   - Parallel DFS can be useful in applications where depth-first exploration is preferred, such as graph traversal with specific search criteria or constraint satisfaction problems.

### Comparison:

1. **Parallelism Type**:
   - BFS exhibits fine-grained parallelism, while DFS typically exhibits coarse-grained parallelism.
   
2. **Load Balancing**:
   - Load balancing is typically easier to achieve in parallel BFS, as nodes at each level can be evenly distributed among threads.
   - Load balancing may be more challenging in parallel DFS, especially if the search tree is unevenly partitioned among threads.

3. **Memory Usage**:
   - Parallel BFS may require more memory to store multiple frontier sets, one for each level.
   - Parallel DFS may have lower memory requirements as it typically operates on a single search path at a time.

4. **Performance**:
   - The performance of parallel BFS and parallel DFS depends on factors such as graph structure, workload distribution, and synchronization overhead.
   - In general, parallel BFS may exhibit better scalability on multi-core processors due to its finer-grained parallelism.

### Summary:

- Parallel BFS and parallel DFS are parallelized versions of traditional graph traversal algorithms designed to exploit parallelism on multi-core or distributed memory systems.
- Parallel BFS is well-suited for applications requiring level-wise traversal, while parallel DFS may be preferable for applications where depth-first exploration is more appropriate.
- The choice between parallel BFS and parallel DFS depends on factors such as algorithmic requirements, graph characteristics, and performance considerations.


OpenMP (Open Multi-Processing) is an API (Application Programming Interface) that supports multi-platform shared memory multiprocessing programming in C, C++, and Fortran. It enables developers to write parallel programs that can execute concurrently on multi-core processors and symmetric multiprocessing (SMP) systems.

### Key Features:

1. **Shared Memory Model**: OpenMP operates on a shared memory model, where multiple threads of execution share the same memory space. This simplifies parallel programming compared to distributed memory models like MPI (Message Passing Interface).

2. **Directive-Based Programming**: OpenMP uses compiler directives to parallelize code sections. These directives are pragmas (compiler hints) that instruct the compiler to generate parallel code. They are easy to add to existing serial code and can be selectively applied to parallelize specific regions of code.

3. **Thread-Based Parallelism**: OpenMP creates threads to achieve parallel execution. The number of threads is typically determined at runtime, based on the available hardware resources (e.g., number of CPU cores).

4. **Automatic Load Balancing**: OpenMP automatically handles load balancing by distributing work among threads. Each thread executes a portion of the parallel region, and the runtime system ensures that work is evenly distributed across threads.

5. **Task Parallelism**: OpenMP supports task-based parallelism, allowing developers to express parallelism at a higher level of abstraction. Tasks are units of work that can be executed independently and asynchronously.

6. **Data Sharing and Synchronization**: OpenMP provides mechanisms for sharing data among threads, including shared variables, private variables, and synchronization constructs like barriers, atomic operations, and critical sections.

### Basic Usage:

1. **Compiler Directives**: OpenMP directives are added to the code using pragma statements. For example:
    ```c
    #pragma omp parallel
    {
        // Parallel region
        printf("Hello, world!\n");
    }
    ```

2. **Parallel Regions**: The `omp parallel` directive creates a team of threads, each executing the enclosed parallel region.

3. **Work Sharing Constructs**: Directives like `omp for` and `omp sections` enable parallelization of loops and sections of code, respectively.

4. **Data Scoping**: OpenMP provides different data scoping options, including shared, private, firstprivate, and threadprivate, to control how variables are accessed and shared among threads.

5. **Synchronization**: Synchronization constructs like `omp barrier`, `omp critical`, `omp atomic`, and `omp flush` are used to coordinate access to shared resources and ensure correct program behavior.

### Example:

Here's a simple example of parallelizing a loop using OpenMP in C:
```c
#include <stdio.h>
#include <omp.h>

int main() {
    int n = 10;
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += i;
    }

    printf("Sum: %d\n", sum);

    return 0;
}
```

### Platforms and Implementations:

OpenMP is supported by most modern compilers, including GCC, Clang, Intel Compiler (icc), and Microsoft Visual C++. It is available on various operating systems, including Linux, Windows, and macOS. Additionally, OpenMP is supported by many parallel computing platforms, including supercomputers and cloud-based environments.

### Summary:

OpenMP is a powerful and widely used API for shared memory parallel programming. It provides a simple and portable way to write parallel programs in C, C++, and Fortran, enabling developers to leverage multi-core processors and SMP systems for improved performance and scalability. With its directive-based approach and support for task parallelism, OpenMP simplifies the development of parallel applications while offering high-level control over parallel execution.


### 1. `#pragma omp parallel`:

This directive creates a team of threads, each of which executes the enclosed parallel region. It is one of the fundamental directives in OpenMP for parallelism.

```c
#pragma omp parallel
{
    // Parallel region
    printf("Hello, world!\n");
}
```

In this example, multiple threads are created to execute the statements within the parallel region. Each thread executes the code independently, and the runtime system ensures proper thread management and synchronization.

### 2. `#pragma omp parallel for`:

This directive combines the functionality of both `#pragma omp parallel` and a loop parallelization directive (`#pragma omp for`). It parallelizes the execution of a loop across multiple threads.

```c
#include <omp.h>
#include <stdio.h>

int main() {
    int n = 100;
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += i;
    }

    printf("Sum: %d\n", sum);

    return 0;
}
```

In this example, the loop is parallelized across multiple threads, with each thread executing a portion of the loop iterations. The `reduction` clause is used to ensure that the `sum` variable is properly updated without race conditions.

### 3. `#pragma omp critical`:

This directive defines a critical section, which is a code segment that only one thread can execute at a time. It ensures that concurrent access to shared resources is properly synchronized to prevent data races and inconsistencies.

```c
#include <omp.h>
#include <stdio.h>

int main() {
    int sharedVariable = 0;

    #pragma omp parallel
    {
        #pragma omp critical
        {
            sharedVariable++;
            printf("Thread %d incremented sharedVariable to %d\n", omp_get_thread_num(), sharedVariable);
        }
    }

    return 0;
}
```

In this example, the `sharedVariable` is accessed and modified within a critical section to ensure that only one thread can modify it at a time. This prevents race conditions and ensures the correctness of the program's output.

### Summary:

- `#pragma omp parallel`: Creates a team of threads to execute a parallel region.
- `#pragma omp parallel for`: Parallelizes the execution of a loop across multiple threads.
- `#pragma omp critical`: Defines a critical section where only one thread can execute at a time, ensuring proper synchronization for shared resources.
*/
