#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
        /**Умножение двух матриц.
        * @param float* A - указатель на массив значений матрицы A
        * @param float* B - указатель на массив значений матрицы B
        * @param float* C - указатель на массив значений матрицы C
        * @param int rowsA - количество строк в матрице A
        * @param int colsA - количество столбцов в матрице A
        * @param int colsB - количество столбцов в матрице B
        */
__global__ void matrixMultiply(float* A, float* B, float* C, int rowsA, int colsA, int colsB) //передается одномерный массив
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; //номер потока
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) { //текущий поток находится в пределах размеров матрицы C
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        } //индекс элемента матрица
        C[row * colsB + col] = sum;
    }
}

        /**Умножение двух матриц.
        * @param matrixA - двумерный массив A
        * @param matrixB - двумерный массив B
        * @param float* C - указатель на массив значений матрицы C
        * @return двумерный массив умноженных матриц
        */
std::vector<std::vector<float>> multiplyMatrices(const std::vector<std::vector<float>>& matrixA, const std::vector<std::vector<float>>& matrixB)
{
    int rowsA = matrixA.size();
    int colsA = matrixA[0].size();
    int rowsB = matrixB.size();
    int colsB = matrixB[0].size();
    std::vector<std::vector<float>> result(rowsA, std::vector<float>(colsB, 0));

    float* deviceMatrixA; //объявление массивов
    float* deviceMatrixB;
    float* deviceResultMatrix;

    cudaMalloc((void**)&deviceMatrixA, rowsA * colsA * sizeof(float)); //выделение памяти на куде
    cudaMalloc((void**)&deviceMatrixB, rowsB * colsB * sizeof(float));
    cudaMalloc((void**)&deviceResultMatrix, rowsA * colsB * sizeof(float));

    cudaMemcpy(deviceMatrixA, matrixA.data(), rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);//скопировать в память
    cudaMemcpy(deviceMatrixB, matrixB.data(), rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16); //в куде размер блока
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x, (rowsA + blockSize.y - 1) / blockSize.y);//размер сетки

    matrixMultiply<<<gridSize, blockSize>>>(deviceMatrixA, deviceMatrixB, deviceResultMatrix,
                                                  rowsA, colsA, colsB);

    
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            result[i][j] = deviceResultMatrix[i * colsB + j];//переведение из одномерного в двумерный
        }
    }
    cudaFree(deviceMatrixA); //почистить память
    cudaFree(deviceMatrixB);
    cudaFree(deviceResultMatrix);
    return result;
}