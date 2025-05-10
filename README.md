# Parallel Linear Algebra Operations with OpenMP

## Vector Operations

### Vector Addition

Vector addition is a fundamental operation that adds corresponding elements of two vectors.

**Mathematical Definition:**
For two vectors $\vec{a}$ and $\vec{b}$ of size $n$, the result vector $\vec{c} = \vec{a} + \vec{b}$ is defined as:

$c_i = a_i + b_i \quad \text{for } i = 0, 1, 2, \ldots, n-1$

**Visual Representation:**

```
   a = [a₀, a₁, a₂, ..., aₙ₋₁]
   b = [b₀, b₁, b₂, ..., bₙ₋₁]
   +    +   +   +       +
   =    =   =   =       =
   c = [c₀, c₁, c₂, ..., cₙ₋₁]
Where: c₀ = a₀ + b₀, c₁ = a₁ + b₁, etc.
```

**Parallel Implementation:**

```cpp
std::vector<double> vectorAdd(const std::vector<double>& v1, const std::vector<double>& v2) {

    size_t size = v1.size();
    std::vector<double> result(size);

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        result[i] = v1[i] + v2[i];
    }

    return result;
}
```

**Parallelization Diagram:**

```
Thread 0: [a₀+b₀, a₁+b₁, ..., aₙ₋₁÷ₚ+bₙ₋₁÷ₚ]
Thread 1: [aₙ÷ₚ+bₙ÷ₚ, ..., a₂ₙ÷ₚ₋₁+b₂ₙ÷ₚ₋₁]
   ...
Thread p-1: [a₍ₚ₋₁₎ₙ÷ₚ+b₍ₚ₋₁₎ₙ÷ₚ, ..., aₙ₋₁+bₙ₋₁]
```

Where $p$ is the number of threads, and each thread processes $n/p$ elements.

### Vector-Scalar Multiplication

Vector-scalar multiplication multiplies each element of a vector by a scalar value.

**Mathematical Definition:**
For a vector $\vec{a}$ of size $n$ and a scalar $s$, the result vector $\vec{b} = s \cdot \vec{a}$ is defined as:

$b_i = s \cdot a_i \quad \text{for } i = 0, 1, 2, \ldots, n-1$

**Visual Representation:**

```
   a = [a₀, a₁, a₂, ..., aₙ₋₁]
   s = scalar
   ×    ×   ×   ×       ×
   =    =   =   =       =
   b = [b₀, b₁, b₂, ..., bₙ₋₁]
Where: b₀ = s × a₀, b₁ = s × a₁, etc.
```

**Parallel Implementation:**

```cpp
std::vector<double> vectorScalarMultiply(const std::vector<double>& v, double scalar) {
    size_t size = v.size();
    std::vector<double> result(size);

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        result[i] = v[i] * scalar;
    }

    return result;
}
```

**Parallelization Diagram:**

```
Thread 0: [s×a₀, s×a₁, ..., s×aₙ₋₁÷ₚ]
Thread 1: [s×aₙ÷ₚ, ..., s×a₂ₙ÷ₚ₋₁]
   ...
Thread p-1: [s×a₍ₚ₋₁₎ₙ÷ₚ, ..., s×aₙ₋₁]
```

## Matrix Operations

### Matrix Multiplication

Matrix multiplication is a binary operation that produces a matrix from two input matrices.

**Mathematical Definition:**
For matrices $A$ of size $m \times k$ and $B$ of size $k \times n$, the result matrix $C = A \times B$ of size $m \times n$ is defined as:

$C_{i,j} = \sum_{l=0}^{k-1} A_{i,l} \cdot B_{l,j} \quad \text{for } i = 0, 1, 2, \ldots, m-1 \text{ and } j = 0, 1, 2, \ldots, n-1$

## Prerequisites

1. **C++ Compiler** (with OpenMP support)
   - gcc/g++ (version 4.2 or later) or any other compiler that supports OpenMP.
   
2. **OpenMP** (usually included in modern compilers)

3. **CMake** (to manage the build process)

## Cloning the Repository

To get started, clone the repository:

```bash
git clone https://github.com/ppp16bit/parallel-linear-algebra.git
cd parallel-linear-algebra
```

## Output

```
maximum number of threads: 12

test vector add
v1 = [1, 2, 3, 4]
v2 = [5, 6, 7, 8]
v1 + v2 = [6, 8, 10, 12]
test case 1 passed
test with large vectors (size 1000000)
first few elements of result: 3, 3, 3, ...
test case 2 passed

vec scalar multiplication
v = [1, 2, 3, 4]
scalar = 2.5
v * scalar = [2.5, 5, 7.5, 10]
test case 1 passed
test with large vector (size 1000000)
first few elements of result: 4.5, 4.5, 4.5, ...
test case 2 passed

matrix multiplication
Matrix A = 
[    1.00,     2.00,     3.00]
[    4.00,     5.00,     6.00]
Matrix B = 
[    7.00,     8.00]
[    9.00,    10.00]
[   11.00,    12.00]
A * B = 
[   58.00,    64.00]
[  139.00,   154.00]
test case 1 passed
test with 100x100 matrices
first few elements of result:
100.00 100.00 100.00 ...
100.00 100.00 100.00 ...
100.00 100.00 100.00 ...
test case 2 passed

performance testing
multiplication with 1 threads: 0.7171 seconds
multiplication with 2 threads: 0.3643 seconds
multiplication with 4 threads: 0.1915 seconds
multiplication with 8 threads: 0.1770 seconds
```