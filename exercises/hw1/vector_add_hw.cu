#include <stdio.h>

// Error check macro
#define cudaCheckErrors(msg) \
    do {\
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const int DSIZE = 4096;     // Size of the vector
const int block_size = 256; // CUDA maximum is 1024

// Add vectors A + B = C
__global__ void vadd(const float *A, const float *B, float *C, int ds) {
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < ds) { C[idx] = B[idx] + A[idx]; }
}

int main() {
    // 1) Initialize vectors host side
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE]; // allocate space for vectors in host memory
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];
    for (int i = 0; i < DSIZE; i++) {  // initialize vectors in host memory
        
        h_A[i] = rand()/float(RAND_MAX);
        h_B[i] = rand()/float(RAND_MAX);
        h_C[i] = 0.;
    }

    // 2) Initialize vectors device side
    cudaMalloc(&d_A, DSIZE*sizeof(float)); // Allocate device space for vector A
    cudaMalloc(&d_B, DSIZE*sizeof(float)); // Allocate device space for vector B
    cudaMalloc(&d_C, DSIZE*sizeof(float)); // Allocate device space for vector C
    // Commonly asked question: why is first argument of cudaMalloc a ptr to ptr?
    // Answer: &d_A is a ptr to ptr in device memory; one dereference (*&d_a)
    // is the pointer which points to data in device memory; second dereference
    // points to the data
    cudaCheckErrors("cudaMalloc failure"); // Error checking

    // 3) Copy host vectors to device
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    // Don't need to copy into C, we will do addition on d_C which is already
    // initialized in device, then copy back to host
    cudaCheckErrors("cudaMemcpy H2d failure");
    
    // 4) Do addition
    // Note: number of blocks is size of vector / block size, rounded up
    // so if e.g. 401 elements, block size 100, get 5 blocks
    vadd<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // 5) Copy result (vector C) from device to host
    //cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failures or cudaMemcpy D2H failure");

    // Sample 
    printf("A[0] = %f\n", h_A[0]);
    printf("B[0] = %f\n", h_B[0]);
    printf("C[0] = %f\n", h_C[0]);
    return 0;
}
