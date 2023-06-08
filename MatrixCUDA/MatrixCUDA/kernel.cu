﻿// This program computes a simple version of matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

__global__ void matrixMul(const int* a, const int* b, int* c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    
    c[row * N + col] = 0;
    for (int k = 0; k < N; k++) {
        
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}

// Проверка результатов на CPU
void verify_result(vector<int>& a, vector<int>& b, vector<int>& c, int N) {
    
    for (int i = 0; i < N; i++) {
        
        for (int j = 0; j < N; j++) {
            
            int tmp = 0;
            for (int k = 0; k < N; k++) {
                
                tmp += a[i * N + k] * b[k * N + j];
            }

            
            assert(tmp == c[i * N + j]);
            //cout << tmp << " ";
        }
    }
}

int main() {
    for (int t = 0; t < 5; t++) {
        
        int N = 1 << 10;

        
        size_t bytes = N * N * sizeof(int);

        
        vector<int> h_a(N * N);
        vector<int> h_b(N * N);
        vector<int> h_c(N * N);

        
        generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
        generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

        
        int* d_a, * d_b, * d_c;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        
        cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

        
        int THREADS = 32;

        
        int BLOCKS = N / THREADS;

        
        dim3 threads(THREADS, THREADS);
        dim3 blocks(BLOCKS, BLOCKS);

        
        matrixMul << <blocks, threads >> > (d_a, d_b, d_c, N);

        
        cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    return 0;
}