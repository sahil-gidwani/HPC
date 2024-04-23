// g++ -fopenmp example.cpp -o example
// ./example

#include <iostream>
#include <omp.h>
#include <climits>

using namespace std;

// Function to find the minimum value in an array using OpenMP reduction
void min_reduction(int arr[], int n) {
    int min_value = INT_MAX; // Initialize min_value to the maximum possible integer value

    // OpenMP parallel for loop with reduction clause (min)
    #pragma omp parallel for reduction(min: min_value)
    for (int i = 0; i < n; i++) {
        if (arr[i] < min_value) {
            min_value = arr[i]; // Update min_value if the current element is smaller
        }
    }

    cout << "Minimum value: " << min_value << endl; // Output the minimum value
}

// Function to find the maximum value in an array using OpenMP reduction
void max_reduction(int arr[], int n) {
    int max_value = INT_MIN; // Initialize max_value to the minimum possible integer value

    // OpenMP parallel for loop with reduction clause (max)
    #pragma omp parallel for reduction(max: max_value)
    for (int i = 0; i < n; i++) {
        if (arr[i] > max_value) {
          max_value = arr[i]; // Update max_value if the current element is larger
        }
    }

    cout << "Maximum value: " << max_value << endl; // Output the maximum value
}

// Function to find the sum of all elements in an array using OpenMP reduction
void sum_reduction(int arr[], int n) {
    int sum = 0; // Initialize sum to zero

    // OpenMP parallel for loop with reduction clause (sum)
    #pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i]; // Add each element to sum
    }

    cout << "Sum: " << sum << endl; // Output the sum
}

// Function to find the average of all elements in an array using OpenMP reduction
void average_reduction(int arr[], int n) {
    int sum = 0; // Initialize sum to zero

    // OpenMP parallel for loop with reduction clause (sum)
    #pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < n; i++) {
      sum += arr[i]; // Add each element to sum
    }

    cout << "Average: " << (double)sum / n << endl; // Output the average (casting sum to double for accurate division)
}

int main() {
    int *arr, n;
    cout << "\nEnter the total number of elements: ";
    cin >> n;

    arr = new int[n]; // Dynamically allocate memory for the array
    cout << "\nEnter the elements: ";
    for (int i = 0; i < n; i++) {
        cin >> arr[i]; // Input the elements of the array
    }

    min_reduction(arr, n); // Find the minimum value in the array
    max_reduction(arr, n); // Find the maximum value in the array
    sum_reduction(arr, n); // Find the sum of all elements in the array
    average_reduction(arr, n); // Find the average of all elements in the array

    delete[] arr; // Deallocate memory for the array
    return 0;
}

// -----------------------------------------------------
// #include <iostream>
// #include <omp.h>
// #include <ctime>
// #include <cstdlib>

// using namespace std;

// void min(int *arr, int n)
// {
//    double min_val = 10000;
//    int i;
//    cout << endl;
// #pragma omp parallel for reduction(min : min_val)
//    for (i = 0; i < n; i++)
//    {
//       cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
//       if (arr[i] < min_val)
//       {
//          min_val = arr[i];
//       }
//    }
//    cout << "\n\nmin_val = " << min_val << endl;
// }

// void max(int *arr, int n)
// {
//    double max_val = 0.0;
//    int i;

// #pragma omp parallel for reduction(max : max_val)
//    for (i = 0; i < n; i++)
//    {
//       cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
//       if (arr[i] > max_val)
//       {
//          max_val = arr[i];
//       }
//    }
//    cout << "\n\nmax_val = " << max_val << endl;
// }

// void avg(int *arr, int n)
// {
//    int i;
//    float avg = 0, sum = 0;
// #pragma omp parallel reduction(+:sum)
//    {
//       // int id = omp_get_thread_num();
// #pragma omp for
//       for (i = 0; i < n; i++)
//       {
//          sum = sum + arr[i];
//          cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
//       }
//    }
//    cout << "\n\nSum = " << sum << endl;
//    avg = sum / n;
//    cout << "\nAverage = " << avg << endl;
// }

// int main()
// {
//    omp_set_num_threads(4);
//    int n, i;

//    cout << "Enter the number of elements in the array: ";
//    cin >> n;
//    int arr[n];

//    srand(time(0));
//    for (int i = 0; i < n; ++i)
//    {
//       arr[i] = rand() % 100;
//    }

//    cout << "\nArray elements are: ";
//    for (i = 0; i < n; i++)
//    {
//       cout << arr[i] << ",";
//    }

//    min(arr, n);
//    max(arr, n);
//    avg(arr, n);
//    return 0;
// }

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


Parallel reduction is a common parallel computing technique used to compute the sum (or other associative operations like maximum, minimum, etc.) of a large set of values in parallel. It aims to reduce the computation time by distributing the workload across multiple processing units, such as CPU cores or GPUs. Let's delve into the details of parallel reduction:

### Sequential Reduction:

In a sequential reduction, the sum of elements in an array is computed iteratively, typically using a loop. For example, to compute the sum of an array `arr`:

```c
int sum = 0;
for (int i = 0; i < n; i++) {
    sum += arr[i];
}
```

This sequential approach has a time complexity of \(O(n)\), where \(n\) is the number of elements in the array.

### Parallel Reduction:

In parallel reduction, the workload is divided among multiple processing units, each of which computes a partial sum of the array elements. These partial sums are then combined iteratively until a single result, the reduction, is obtained. Parallel reduction can be implemented using various parallel programming models, such as OpenMP, CUDA, or MPI.

### Example Implementation (OpenMP):

```c
#include <stdio.h>
#include <omp.h>

int main() {
    int n = 1000000; // Number of elements
    int arr[n];
    int sum = 0;

    // Initialize array
    for (int i = 0; i < n; i++) {
        arr[i] = i + 1; // Values from 1 to n
    }

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }

    printf("Sum: %d\n", sum);

    return 0;
}
```

In this OpenMP example, the `reduction(+:sum)` clause parallelizes the sum computation across multiple threads. Each thread computes a partial sum of the array elements, and the partial sums are combined using addition (`+`) at the end of the parallel region. The `reduction` clause ensures that each thread has its private copy of the `sum` variable, and the final reduction operation combines the partial sums into a single result.

### Performance Considerations:

- **Load Balancing**: It is essential to distribute the workload evenly among processing units to achieve optimal performance. Uneven workload distribution can lead to idle threads or cores, reducing overall efficiency.
  
- **Communication Overhead**: The overhead associated with combining partial results can impact performance, especially in distributed memory systems or GPUs. Minimizing communication overhead is crucial for achieving high parallel efficiency.

- **Algorithmic Complexity**: The time complexity of the reduction algorithm affects performance. Some reduction algorithms, such as tree-based reductions, offer better scalability and performance than simple linear reductions, especially for large datasets.

### Applications:

- **Parallel Reduction**: Used in various scientific computing applications, numerical simulations, data analytics, and machine learning algorithms, where aggregating data from multiple sources is required efficiently.
  
- **Reduction Operations**: Apart from sum, parallel reduction can also be applied to other associative operations like maximum, minimum, bitwise AND, bitwise OR, etc., depending on the specific application requirements.

### Summary:

Parallel reduction is a widely used parallel computing technique for efficiently computing the sum (or other associative operations) of a large set of values. By distributing the workload across multiple processing units and combining partial results, parallel reduction offers significant speedup over sequential approaches, making it invaluable for accelerating various computational tasks in parallel computing environments.
*/