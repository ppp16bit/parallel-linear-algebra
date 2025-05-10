#include <vector>
#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <cassert>
#include <iomanip>

std::vector<double> vecAdd(const std::vector<double> &v1, const std::vector<double> &v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("same size");
    }

    size_t size = v1.size();
    std::vector<double> result(size);

#pragma omp parallel for
    for (size_t row = 0; row < size; row++) {
        result[row] = v1[row] + v2[row];
    }

    return result;
}

std::vector<double> vecScalarMult(const std::vector<double> &v, double scalar) {
    size_t size = v.size();
    std::vector<double> result(size);

#pragma omp parallel for
    for (size_t row = 0; row < size; row++) {
        result[row] = v[row] * scalar;
    }

    return result;
}

std::vector<std::vector<double>> matrixMult(
    const std::vector<std::vector<double>> &A,
    const std::vector<std::vector<double>> &B) {

    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("not compatible for multiplication");
    }

    size_t aRows = A.size();
    size_t sharedDim = A[0].size();
    size_t bCols = B[0].size();

    std::vector<std::vector<double>> result(aRows, std::vector<double>(bCols, 0.0));
    const size_t BLOCK_SIZE = 32;

#pragma omp parallel for
    for (size_t row = 0; row < aRows; row += BLOCK_SIZE) {
        for (size_t col = 0; col < bCols; col += BLOCK_SIZE) {
            size_t row_end = std::min(row + BLOCK_SIZE, aRows);
            size_t col_end = std::min(col + BLOCK_SIZE, bCols);

            for (size_t outerRow = row; outerRow < row_end; outerRow++) {
                for (size_t outerCol = col; outerCol < col_end; outerCol++) {
                    double sum = 0.0;
                    for (size_t inner = 0; inner < sharedDim; inner++) {
                        sum += A[outerRow][inner] * B[inner][outerCol];
                    }
                    result[outerRow][outerCol] = sum;
                }
            }
        }
    }

    return result;
}

void printVec(const std::vector<double> &v, const std::string &name = "") {
    if (!name.empty()) {
        std::cout << name << " = ";
    }

    std::cout << "[";
    for (size_t row = 0; row < v.size(); row++) {
        std::cout << v[row];
        if (row < v.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void printMat(const std::vector<std::vector<double>> &m, const std::string &name = "") {
    if (!name.empty()) {
        std::cout << name << " = " << std::endl;
    }

    for (const auto &row : m) {
        std::cout << "[";
        for (size_t outerRow = 0; outerRow < row.size(); outerRow++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << row[outerRow];
            if (outerRow < row.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

void testVectorAddition() {
    std::cout << "test vector add" << std::endl;

    std::vector<double> v1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> v2 = {5.0, 6.0, 7.0, 8.0};
    std::vector<double> expected1 = {6.0, 8.0, 10.0, 12.0};

    std::vector<double> result1 = vecAdd(v1, v2);

    printVec(v1, "v1");
    printVec(v2, "v2");
    printVec(result1, "v1 + v2");

    assert(result1 == expected1 && "add test case 1 failed");
    std::cout << "test case 1 passed" << std::endl;

    size_t size = 1000000;
    std::vector<double> v3(size, 1.0);
    std::vector<double> v4(size, 2.0);
    std::vector<double> result2 = vecAdd(v3, v4);

    std::cout << "test with large vectors (size " << size << ")" << std::endl;
    std::cout << "first few elements of result: " << result2[0] << ", "
              << result2[1] << ", " << result2[2] << ", ..." << std::endl;

    assert(result2[0] == 3.0 && result2[1] == 3.0 && result2[2] == 3.0 && "Vector addition test case 2 failed");
    std::cout << "test case 2 passed" << std::endl;
    std::cout << std::endl;
}

void testVectorScalarMultiplication() {
    std::cout << "vec scalar multiplication" << std::endl;

    std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
    double scalar = 2.5;
    std::vector<double> expected = {2.5, 5.0, 7.5, 10.0};

    std::vector<double> result = vecScalarMult(v, scalar);
    printVec(v, "v");
    std::cout << "scalar = " << scalar << std::endl;
    printVec(result, "v * scalar");

    assert(result == expected && "multiplication test case 1 failed");
    std::cout << "test case 1 passed" << std::endl;

    size_t size = 1000000;
    std::vector<double> v2(size, 3.0);
    double scalar2 = 1.5;
    std::vector<double> result2 = vecScalarMult(v2, scalar2);

    std::cout << "test with large vector (size " << size << ")" << std::endl;
    std::cout << "first few elements of result: " << result2[0] << ", "
              << result2[1] << ", " << result2[2] << ", ..." << std::endl;

    assert(result2[0] == 4.5 && result2[1] == 4.5 && result2[2] == 4.5 && "multiplication test case 2 failed");
    std::cout << "test case 2 passed" << std::endl;
    std::cout << std::endl;
}

void testMatrixMultiplication() {
    std::cout << "matrix multiplication" << std::endl;

    std::vector<std::vector<double>> A = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}};

    std::vector<std::vector<double>> B = {
        {7.0, 8.0},
        {9.0, 10.0},
        {11.0, 12.0}};

    std::vector<std::vector<double>> expected = {
        {58.0, 64.0},
        {139.0, 154.0}};

    std::vector<std::vector<double>> result = matrixMult(A, B);
    printMat(A, "Matrix A");
    printMat(B, "Matrix B");
    printMat(result, "A * B");

    bool testPassed = true;
    for (size_t i = 0; i < expected.size(); i++) {
        for (size_t j = 0; j < expected[0].size(); j++) {
            if (std::abs(result[i][j] - expected[i][j]) > 1e-10) {
                testPassed = false;
                break;
            }
        }
    }

    assert(testPassed && "multiplication test case 1 failed");
    std::cout << "test case 1 passed" << std::endl;

    size_t size = 100;
    std::vector<std::vector<double>> C(size, std::vector<double>(size, 1.0));
    std::vector<std::vector<double>> D(size, std::vector<double>(size, 1.0));

    std::cout << "test with " << size << "x" << size << " matrices" << std::endl;

    std::vector<std::vector<double>> result2 = matrixMult(C, D);

    bool testPassed2 = true;
    for (size_t i = 0; i < 3 && i < size; i++) {
        for (size_t j = 0; j < 3 && j < size; j++) {
            if (std::abs(result2[i][j] - size) > 1e-10) {
                testPassed2 = false;
                break;
            }
        }
    }

    std::cout << "first few elements of result:" << std::endl;
    for (size_t i = 0; i < 3 && i < size; i++) {
        for (size_t j = 0; j < 3 && j < size; j++) {
            std::cout << result2[i][j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    assert(testPassed2 && "multiplication test case 2 failed");
    std::cout << "test case 2 passed" << std::endl;
    std::cout << std::endl;
}

void performanceTest() {
    std::cout << "performance testing" << std::endl;

    size_t size = 500;
    std::vector<std::vector<double>> A(size, std::vector<double>(size));
    std::vector<std::vector<double>> B(size, std::vector<double>(size));

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            A[i][j] = 1.0 + ((i * j) % 10) / 10.0;
            B[i][j] = 1.0 + ((i + j) % 10) / 10.0;
        }
    }

    for (int numThreads : {1, 2, 4, 8}) {
        omp_set_num_threads(numThreads);

        double startTime = omp_get_wtime();
        std::vector<std::vector<double>> result = matrixMult(A, B);
        double endTime = omp_get_wtime();

        std::cout << "multiplication with " << numThreads
                  << " threads: " << std::fixed << std::setprecision(4)
                  << (endTime - startTime) << " seconds" << std::endl;
    }

    std::cout << std::endl;
}

int main(void) {
    std::cout << "running with OpenMP version: " << _OPENMP << std::endl;
    std::cout << "maximum number of threads: " << omp_get_max_threads() << std::endl;
    std::cout << std::endl;

    testVectorAddition();
    testVectorScalarMultiplication();
    testMatrixMultiplication();
    performanceTest();

    std::cout << "success" << std::endl;
    return 0;
}