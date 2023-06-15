#include <cuda_runtime.h>


bool is_cuda() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}