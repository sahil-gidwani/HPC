// g++ -fopenmp example.cpp -o example
// ./example

#include <iostream>
#include <stdlib.h>
#include <omp.h>

using namespace std;

// Sequential Bubble Sort Algorithm
void sequentialBubble(int *a, int n) {
    for (int i = 0; i < n; i++) {
        // Last i elements are already sorted, so inner loop can be reduced
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]); // Swap elements if out of order
            }
        }
    }
}

// Parallel Bubble Sort Algorithm
// void parallelBubble(int *a, int n) {
//     #pragma omp parallel for shared(a, n)
//     for (int i = 0; i < n; i++) {
//         // Last i elements are already sorted, so inner loop can be reduced
//         for (int j = 0; j < n - 1; j++) {
//             if (a[j] > a[j + 1]) {
//                 swap(a[j], a[j + 1]); // Swap elements if out of order
//             }
//         }
//     }
// }

void parallelBubble(int *a, int n) {
    for (int i = 0; i < n; i++) {
        int first = i % 2;

        #pragma omp parallel for shared(a, first, n)
        for (int j = first; j < n - 1; j += 2) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]); // Swap elements if out of order
            }
        }
    }
}

// Merge Function for Merge Sort Algorithm
void merge(int *arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    // Copy data to temporary arrays L[] and R[]
    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Merge the temporary arrays back into arr[l..r]
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Sequential Merge Sort Algorithm
void sequentialMergeSort(int *arr, int l, int r) {
    if (l < r) {
        // Same as (l+r)/2, but avoids overflow for large l and r
        int m = l + (r - l) / 2;

        // Sort first and second halves
        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m + 1, r);

        // Merge the sorted halves
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort Algorithm
void parallelMergeSort(int *arr, int l, int r) {
    if (l < r) {
        // Same as (l+r)/2, but avoids overflow for large l and r
        int m = l + (r - l) / 2;

        // Parallelize the sorting of first and second halves
        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, l, m);
            #pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }

        // Merge the sorted halves
        merge(arr, l, m, r);
    }
}

// Function to print array elements
void printArray(int *a, int n) {
    for (int i = 0; i < n; i++) {
        cout << a[i] << " ";
    }
    cout << endl;
}

int main() {
    int n;
    cout << "Enter total number of elements: ";
    cin >> n;

    // Allocate memory for the array
    int *a = new int[n];
    
    // Initialize the array with random values
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 100; // Generate random numbers between 0 and 99
    }

    // Copy the array for later comparison
    int *original = new int[n];
    for (int i = 0; i < n; i++) {
        original[i] = a[i];
    }

    // Measure sequential bubble sort time
    double startSeq = omp_get_wtime();
    sequentialBubble(a, n);
    double endSeq = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << endSeq - startSeq << " seconds" << endl;

    // Verify correctness of sequential sorting
    cout << "Sequential Bubble Sorted Array: ";
    printArray(a, n);

    // Reset array for parallel sorting
    for (int i = 0; i < n; i++) {
        a[i] = original[i];
    }

    // Measure parallel bubble sort time
    double startPar = omp_get_wtime();
    parallelBubble(a, n);
    double endPar = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << endPar - startPar << " seconds" << endl;

    // Verify correctness of parallel sorting
    cout << "Parallel Bubble Sorted Array: ";
    printArray(a, n);

    // Reset array for merge sorting
    for (int i = 0; i < n; i++) {
        a[i] = original[i];
    }

    // Measure sequential merge sort time
    startSeq = omp_get_wtime();
    sequentialMergeSort(a, 0, n-1);
    endSeq = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << endSeq - startSeq << " seconds" << endl;

    // Verify correctness of sequential sorting
    cout << "Sequential Merge Sorted Array: ";
    printArray(a, n);

    // Reset array for parallel sorting
    for (int i = 0; i < n; i++) {
        a[i] = original[i];
    }

    // Measure parallel merge sort time
    startPar = omp_get_wtime();
    parallelMergeSort(a, 0, n-1);
    endPar = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << endPar - startPar << " seconds" << endl;

    // Verify correctness of parallel sorting
    cout << "Parallel Merge Sorted Array: ";
    printArray(a, n);
    
    // Free allocated memory
    delete[] a;
    delete[] original;

    return 0;
}

/*
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


Bubble Sort is a simple sorting algorithm that repeatedly steps through the list to be sorted, compares each pair of adjacent items, and swaps them if they are in the wrong order. The pass through the list is repeated until no swaps are needed, which indicates that the list is sorted. Let's dive into the details of Bubble Sort:

### Algorithm Steps:

1. **Start**: Begin with an unsorted list of elements.

2. **Pass through the List**: Iterate through the list from the beginning to the end.

3. **Compare Adjacent Elements**: Compare each pair of adjacent elements.

4. **Swap if Necessary**: If the elements are in the wrong order (e.g., the current element is greater than the next element in ascending order), swap them.

5. **Repeat**: Repeat steps 2-4 until no swaps are needed during a pass through the list.

6. **End**: When no swaps are performed during a pass, the list is considered sorted.

### Pseudocode:

```
procedure bubbleSort(A: list of sortable items)
    n := length(A)
    do
        swapped := false
        for i := 1 to n-1 inclusive do
            if A[i-1] > A[i] then
                swap(A[i-1], A[i])
                swapped := true
            end if
        end for
        n := n - 1
    while swapped
end procedure
```

### Example:

Consider the following array:
```
[5, 3, 8, 2, 7]
```

- Pass 1:
  - Compare 5 and 3: Swap (Array becomes [3, 5, 8, 2, 7])
  - Compare 5 and 8: No swap
  - Compare 8 and 2: Swap (Array becomes [3, 5, 2, 8, 7])
  - Compare 8 and 7: Swap (Array becomes [3, 5, 2, 7, 8])
- Pass 2:
  - Compare 3 and 5: No swap
  - Compare 5 and 2: Swap (Array becomes [3, 2, 5, 7, 8])
  - Compare 5 and 7: No swap
  - Compare 7 and 8: No swap
- Pass 3:
  - Compare 3 and 2: Swap (Array becomes [2, 3, 5, 7, 8])
  - Compare 3 and 5: No swap
  - Compare 5 and 7: No swap
  - Compare 7 and 8: No swap

No swaps are needed in Pass 3, so the array is sorted.

### Time Complexity:

The time complexity of Bubble Sort is \(O(n^2)\) in the worst and average case, where \(n\) is the number of elements in the array. This is because Bubble Sort requires \(n-1\) passes through the array, and each pass has a time complexity of \(O(n)\).

### Space Complexity:

The space complexity of Bubble Sort is \(O(1)\) because only a constant amount of additional space is required for storing temporary variables during the sorting process.

### Optimizations:

Although Bubble Sort is simple, it is not efficient for large datasets. Some optimizations include:
- **Early Exit**: If no swaps are made during a pass, the array is already sorted, and the algorithm can terminate early.
- **Adaptive Bubble Sort**: Keep track of the last swap index and only iterate up to that point on subsequent passes.

### Summary:

Bubble Sort is a straightforward sorting algorithm that repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. Despite its simplicity, Bubble Sort is inefficient for large datasets and is primarily used for educational purposes or on small datasets where simplicity is prioritized over efficiency.


Merge Sort is a popular sorting algorithm known for its efficiency and stability. It follows the Divide and Conquer approach to sorting, breaking down the input array into smaller subarrays, sorting them independently, and then merging the sorted subarrays to produce the final sorted output. Let's delve into the details of Merge Sort:

### Algorithm Steps:

1. **Divide**: Divide the unsorted array into two halves recursively until each subarray contains only one element.

2. **Conquer**: Sort the individual subarrays recursively.

3. **Merge**: Merge the sorted subarrays to produce the final sorted array.

### Pseudocode:

```
procedure mergeSort(A: list of sortable items)
    if length(A) ≤ 1 then
        return A
    
    mid := length(A) / 2
    left := mergeSort(first half of A)
    right := mergeSort(second half of A)
    
    return merge(left, right)
end procedure

procedure merge(left: list of sortable items, right: list of sortable items)
    result := empty list
    
    while left is not empty and right is not empty do
        if first element of left ≤ first element of right then
            append first element of left to result
            remove first element from left
        else
            append first element of right to result
            remove first element from right
    
    append remaining elements of left to result
    append remaining elements of right to result
    
    return result
end procedure
```

### Example:

Consider the following unsorted array:
```
[5, 3, 8, 2, 7]
```

- Divide:
  - Split the array into two halves: [5, 3] and [8, 2, 7].
  - Recursively divide the subarrays until each subarray contains only one element.

- Conquer:
  - Sort the individual subarrays: [3, 5] and [2, 7, 8].

- Merge:
  - Merge the sorted subarrays: [2, 3, 5, 7, 8].

### Time Complexity:

The time complexity of Merge Sort is \(O(n \log n)\) in all cases, where \(n\) is the number of elements in the array. This is because the array is recursively divided into halves until each subarray contains only one element, and then merged back together. The merge operation itself has a time complexity of \(O(n)\).

### Space Complexity:

The space complexity of Merge Sort is \(O(n)\) because it requires additional space for storing the temporary merged array during the merge operation. However, Merge Sort is a stable sorting algorithm, meaning it preserves the relative order of equal elements, which can be advantageous in certain situations.

### Stability:

Merge Sort is a stable sorting algorithm, meaning that it preserves the relative order of equal elements in the input array. This property is important in applications where the original order of equal elements needs to be maintained after sorting, such as sorting by multiple keys.

### Performance:

Merge Sort is highly efficient and performs well on large datasets. It is widely used in practice and is often preferred over other sorting algorithms like Quick Sort for its stability and predictable performance.

### Summary:

Merge Sort is a highly efficient and stable sorting algorithm based on the Divide and Conquer approach. It achieves \(O(n \log n)\) time complexity in all cases, making it suitable for sorting large datasets. Merge Sort's stability and predictability make it a popular choice for various sorting tasks in computer science and software development.


Parallel Bubble Sort and Parallel Merge Sort are parallelized versions of the traditional Bubble Sort and Merge Sort algorithms, respectively. They are designed to take advantage of multiple processing units, such as CPU cores or GPUs, to achieve faster sorting of large datasets. Let's explore each of them:

### Parallel Bubble Sort:

1. **Approach**:
   - Parallel Bubble Sort follows the same basic principle as the traditional Bubble Sort but leverages parallelism to perform multiple comparisons and swaps simultaneously.

2. **Implementation**:
   - In parallel Bubble Sort, multiple threads or processes are employed to divide the sorting workload among them. Each thread or process independently performs a portion of the sorting operations.
   
3. **Communication**:
   - Communication between threads or processes may be required to exchange information about elements that need to be swapped. Synchronization mechanisms such as barriers or message passing may be used.

4. **Load Balancing**:
   - Load balancing is crucial in parallel Bubble Sort to ensure that the workload is evenly distributed among processing units. Uneven workload distribution can lead to idle resources and suboptimal performance.

### Parallel Merge Sort:

1. **Approach**:
   - Parallel Merge Sort employs the Divide and Conquer approach to sorting, similar to the traditional Merge Sort. However, it parallelizes the sorting and merging operations to exploit parallelism.

2. **Implementation**:
   - In parallel Merge Sort, the input array is divided into smaller subarrays, which are independently sorted by different threads or processes. The sorted subarrays are then merged in parallel to produce the final sorted output.

3. **Communication**:
   - Communication between threads or processes is required during the merging phase to combine the sorted subarrays. Synchronization mechanisms such as barriers or shared memory may be used to coordinate merging.

4. **Load Balancing**:
   - Load balancing is crucial in parallel Merge Sort to ensure that the sorting and merging operations are evenly distributed among processing units. Uneven workload distribution can lead to idle resources and suboptimal performance.

### Comparison:

1. **Performance**:
   - Parallel Merge Sort generally outperforms parallel Bubble Sort, especially on large datasets, due to its superior time complexity (\(O(n \log n)\) vs. \(O(n^2)\)).
   
2. **Efficiency**:
   - Parallel Merge Sort exhibits better scalability and efficiency compared to parallel Bubble Sort, particularly on multi-core processors or distributed systems.

3. **Complexity**:
   - Parallel Merge Sort may have higher implementation complexity compared to parallel Bubble Sort due to the need for more sophisticated parallelization and merging strategies.

4. **Suitability**:
   - Parallel Merge Sort is well-suited for sorting large datasets efficiently, while parallel Bubble Sort may be more appropriate for educational purposes or small datasets where simplicity is prioritized over efficiency.

### Summary:

- Parallel Bubble Sort and Parallel Merge Sort are parallelized versions of the traditional sorting algorithms designed to exploit parallelism for faster sorting of large datasets.
- Parallel Merge Sort generally outperforms Parallel Bubble Sort in terms of efficiency and scalability, particularly on large datasets, due to its superior time complexity.
- The choice between Parallel Bubble Sort and Parallel Merge Sort depends on factors such as dataset size, computational resources, and performance requirements.


Odd-even transposition sort is a parallel sorting algorithm that is suitable for distributed-memory parallel architectures, such as clusters of computers or parallel computers with interconnected processors. It is based on the idea of repeatedly exchanging adjacent elements in the array until the entire array is sorted.

Here's how the algorithm works:

1. **Initialization**:
   - Each processor in the parallel system holds a portion of the array to be sorted.
   - The array is divided evenly among the processors, with each processor responsible for sorting its own portion of the array.

2. **Odd-even Transposition Sort Iterations**:
   - The algorithm consists of multiple iterations, with each iteration performing two phases: the odd phase and the even phase.
   - In each phase, neighboring elements are compared and exchanged if they are out of order.
   - The odd phase involves comparing and exchanging elements at odd indices (1, 3, 5, ...) with their neighboring elements.
   - The even phase involves comparing and exchanging elements at even indices (0, 2, 4, ...) with their neighboring elements.

3. **Communication and Exchange**:
   - During each phase, neighboring processors exchange elements that are out of order.
   - Processors communicate with their neighboring processors to exchange elements as necessary. This communication is typically performed through message passing in distributed-memory systems.

4. **Parallel Execution**:
   - The odd and even phases can be executed concurrently by different processors or threads.
   - Each processor performs the comparisons and exchanges locally on its portion of the array.
   - Synchronization may be required between processors to ensure that each phase completes before proceeding to the next phase.

5. **Iteration and Convergence**:
   - After completing both the odd and even phases, the algorithm repeats the process for a fixed number of iterations or until the array is sorted.
   - The number of iterations required depends on the size of the array and the degree of disorder.

6. **Completion and Sorted Array**:
   - Once the algorithm completes all iterations, each processor holds its portion of the array, which is now partially sorted.
   - Additional steps may be necessary to merge the partially sorted portions of the array into a single sorted array, depending on the specific parallel system and programming model used.

Overall, the odd-even transposition sort algorithm is efficient for parallel sorting on distributed-memory architectures, as it minimizes communication overhead by only exchanging neighboring elements and can exploit parallelism by executing multiple phases concurrently. However, its performance depends on factors such as the number of processors, communication latency, and the size and distribution of the input data.
*/
