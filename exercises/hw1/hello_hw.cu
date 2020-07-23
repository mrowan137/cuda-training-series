#include <stdio.h>

__global__ void hello(){
    // Each thread prints its block and thread
    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

int main(){
    // Launch 2 blocks, 2 threads per block
    hello<<<2,2>>>();
    cudaDeviceSynchronize();
}

