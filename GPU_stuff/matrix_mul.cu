#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device_functions.h"

__global__ void matrixMultiply(float* A, float* B, float* C, int rowsA, int colsA, int colsB) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}


std::vector<std::vector<float>> multiplyMatrices(const std::vector<std::vector<float>>& matrixA, const std::vector<std::vector<float>>& matrixB)
{
    int rowsA = matrixA.size();
    int colsA = matrixA[0].size();
    int rowsB = matrixB.size();
    int colsB = matrixB[0].size();
    std::vector<std::vector<float>> result(rowsA, std::vector<float>(colsB, 0));

    float* deviceMatrixA;
    float* deviceMatrixB;
    float* deviceResultMatrix;

    cudaMalloc((void**)&deviceMatrixA, rowsA * colsA * sizeof(float));
    cudaMalloc((void**)&deviceMatrixB, rowsB * colsB * sizeof(float));
    cudaMalloc((void**)&deviceResultMatrix, rowsA * colsB * sizeof(float));

    cudaMemcpy(deviceMatrixA, matrixA.data(), rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, matrixB.data(), rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x, (rowsA + blockSize.y - 1) / blockSize.y);

    matrixMultiply<<<gridSize, blockSize>>>(deviceMatrixA, deviceMatrixB, deviceResultMatrix,
                                                  rowsA, colsA, colsB);

    
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            result[i][j] = deviceResultMatrix[i * colsB + j];
        }
    }
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceResultMatrix);
    return result;
}