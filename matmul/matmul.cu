
#include <cuda_runtime.h>
#include <cassert>
#include <driver_types.h>
#include <iostream>
#include <iomanip>

using u64 = unsigned long long;
using f32 = float;
using i32 = int;

__host__ void matmul_cpu(u64 *output, u64 *a, u64 *b, u64 a_rows, u64 a_cols, u64 b_rows, u64 b_cols) {
  assert(a_cols == b_rows);
  for (u64 a_row = 0; a_row < a_rows; a_row++) {
    for (u64 b_col = 0; b_col < b_cols; b_col++) {
      u64 output_index = b_col + a_row * b_cols;
      output[output_index] = 0;
      for (u64 a_col = 0; a_col < a_cols; a_col++) {
        u64 a_index = a_col + a_row * a_cols;
        u64 b_index = b_col + a_col * b_cols;
        output[output_index] += a[a_index] * b[b_index];
      }
    }
  }
}

namespace pmpp {
__host__ __device__ i32 ceildiv(i32 x, i32 y) {
  return (x + y - 1) / y;
}

};

__global__ void naive_matmul_gpu_kernel(u64 *output, u64 *a, u64 *b, u64 a_rows, u64 a_cols, u64 b_rows, u64 b_cols) {
  u64 row = threadIdx.y + blockIdx.y * blockDim.y;
  u64 col = threadIdx.x + blockIdx.x * blockDim.x;

  if (col < b_cols && row < a_rows) {
    u64 output_index = col + row * b_cols;
    for (u64 i = 0; i < a_cols; i++) {
      u64 a_index = i + row * a_cols;
      u64 b_index = col + i * b_cols;

      output[output_index] += a[a_index] * b[b_index];
    }
  }
}

__host__ void naive_matmul_gpu(u64 *output, u64 *a, u64 *b, u64 a_rows, u64 a_cols, u64 b_rows, u64 b_cols) {
  assert(a_cols == b_rows);
  u64 *device_output, *device_a, *device_b;
  u64 output_size_bytes = a_rows * b_cols * sizeof(u64);
  u64 a_size_bytes = a_rows * a_cols * sizeof(u64);
  u64 b_size_bytes = b_rows * b_cols * sizeof(u64);

  cudaMalloc(&device_output, output_size_bytes);
  cudaMalloc(&device_a, a_size_bytes);
  cudaMalloc(&device_b, b_size_bytes);

  cudaMemcpy(device_a, a, a_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b, b_size_bytes, cudaMemcpyHostToDevice);
  cudaMemset(device_output, 0, output_size_bytes);

  dim3 blockDim(32, 32, 1);
  dim3 gridDim(pmpp::ceildiv(a_rows, 32), pmpp::ceildiv(b_cols, 32), 1);
  naive_matmul_gpu_kernel<<<gridDim, blockDim>>>(device_output, device_a, device_b, a_rows, a_cols, b_rows, b_cols);

  cudaMemcpy(output, device_output, output_size_bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_output);
  cudaFree(device_a);
  cudaFree(device_b);
}

#define TILE_WIDTH 16
#define TILE_HEIGHT 16
__global__ void tiled_matmul_gpu_kernel(u64 *output, u64 *a, u64 *b, u64 a_rows, u64 a_cols, u64 b_rows, u64 b_cols) {

  // NOTE: use dynamic instead.
  __shared__ u64 a_tile[TILE_HEIGHT][TILE_WIDTH];
  __shared__ u64 b_tile[TILE_HEIGHT][TILE_WIDTH];

  u64 row = threadIdx.y + blockIdx.y * blockDim.y;
  u64 col = threadIdx.x + blockIdx.x * blockDim.x;


  if (row < a_rows && col < b_cols) {
    u64 output_value = 0;
    u64 output_index = col + row * b_cols;

    // for loop, write to tiles, dot product into output_value, iterate.

    output[output_index] = output_value;
  }
}

__host__ void tiled_matmul_gpu(u64 *output, u64 *a, u64 *b, u64 a_rows, u64 a_cols, u64 b_rows, u64 b_cols) {
  assert(a_cols == b_rows);
  u64 *device_output, *device_a, *device_b;
  u64 output_size_bytes = a_rows * b_cols * sizeof(u64);
  u64 a_size_bytes = a_rows * a_cols * sizeof(u64);
  u64 b_size_bytes = b_rows * b_cols * sizeof(u64);

  cudaMalloc(&device_output, output_size_bytes);
  cudaMalloc(&device_a, a_size_bytes);
  cudaMalloc(&device_b, b_size_bytes);

  cudaMemcpy(device_a, a, a_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b, b_size_bytes, cudaMemcpyHostToDevice);
  cudaMemset(device_output, 0, output_size_bytes);

  dim3 blockDim(32, 32, 1);
  dim3 gridDim(pmpp::ceildiv(a_rows, 32), pmpp::ceildiv(b_cols, 32), 1);
  tiled_matmul_gpu_kernel<<<gridDim, blockDim>>>(device_output, device_a, device_b, a_rows, a_cols, b_rows, b_cols);

  cudaMemcpy(output, device_output, output_size_bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_output);
  cudaFree(device_a);
  cudaFree(device_b);
}


void verify(u64 *a, u64 *b, u64 rows, u64 cols) {
  for (u64 i = 0; i < rows; i++) {
    for (u64 j = 0; j < cols; j++) {
      u64 index = j + i * cols;
      if (a[index] != b[index]) {
        std::cout << i << ", " << j << ": ";
        std::cout << a[index] << " != " << b[index] << "\n";
        exit(1);
      }
    }
  }
}

void print_matrix(u64 *a, u64 rows, u64 cols) {
  for (u64 i = 0; i < rows; i++) {
    for (u64 j = 0; j < cols; j++) {
      u64 index = j + i * cols;
      std::cout << a[index] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

void generate_matrix(u64 *a, u64 rows, u64 cols) {
  for (u64 i = 0; i < rows; i++) {
    for (u64 j = 0; j < cols; j++) {
      u64 index = j + i * cols;
      a[index] = index % 11;
    }
  }
}

int main() {
  // const u64 a_rows = 10000;
  // const u64 a_cols = 513;

  // const u64 b_rows = 513;
  // const u64 b_cols = 10000;
  const u64 a_rows = 10;
  const u64 a_cols = 5;

  const u64 b_rows = 5;
  const u64 b_cols = 3;

  const u64 a_size_bytes = a_rows * a_cols * sizeof(u64);
  const u64 b_size_bytes = b_rows * b_cols * sizeof(u64);
  const u64 output_size_bytes = a_rows * b_cols * sizeof(u64);

  u64 *a = (u64*)malloc(a_size_bytes);
  u64 *b = (u64*)malloc(b_size_bytes);

  u64 *cpu = (u64*)malloc(output_size_bytes);
  u64 *naive_gpu = (u64*)malloc(output_size_bytes);
  u64 *tiled_gpu = (u64*)malloc(output_size_bytes);

  generate_matrix(a, a_rows, a_cols);
  generate_matrix(b, b_rows, b_cols);

  matmul_cpu(cpu, a, b, a_rows, a_cols, b_rows, b_cols);
  std::cout << "done cpu matmul\n";
  naive_matmul_gpu(naive_gpu, a, b, a_rows, a_cols, b_rows, b_cols);
  std::cout << "done naive matmul\n";
  tiled_matmul_gpu(tiled_gpu, a, b, a_rows, a_cols, b_rows, b_cols);
  std::cout << "done tiled matmul\n";

  verify(cpu, naive_gpu, a_rows, b_cols);
  verify(cpu, tiled_gpu, a_rows, b_cols);
  std::cout << "Matmul is correct.\n";

  free(cpu);
  free(tiled_gpu);
  free(naive_gpu);
  return 0;
}
