#include <driver_types.h>
#include <iostream>

using u64 = unsigned long long;

inline __host__ __device__ u64 ceildiv(u64 x, u64 y) {
  return (x + y - 1) / y;
}

__global__ void transpose_read_coalesce_kernel(u64 *output, u64 *input, u64 rows, u64 cols) {

  u64 i = threadIdx.y + blockIdx.y * blockDim.y;
  u64 j = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < rows && j < cols) {
    u64 input_index = j + i * cols;
    u64 output_index = i + j * rows;

    // reads are coalesced because their consecutive loads.
    output[output_index] = input[input_index];
  }
}

__global__ void transpose_write_coalesce_kernel(u64 *output, u64 *input, u64 rows, u64 cols) {

  u64 row = threadIdx.y + blockIdx.y * blockDim.y;
  u64 col = threadIdx.x + blockIdx.x * blockDim.x;

  if (row < cols && col < rows) {
    u64 input_index = row + col * cols;
    u64 output_index = col + row * rows;

    // writes are coalesced into one store because their consecutive.
    output[output_index] = input[input_index];
  }
}

__global__ void transpose_read_write_coalesce_kernel(u64 *output, u64 *input, u64 rows, u64 cols) {
  __shared__ u64 buffer[32][32+1];

  u64 row = threadIdx.y + blockIdx.y * blockDim.y;
  u64 col = threadIdx.x + blockIdx.x * blockDim.x;

  if (row < rows && col < cols) {
    u64 input_index = col + row * cols;
    buffer[threadIdx.y][threadIdx.x] = input[input_index];
  }

  __syncthreads();

  if (col < rows && row < cols) {
    u64 output_index = col + row * rows;
    output[output_index] = buffer[threadIdx.x][threadIdx.y];
  }
}

void transpose_gpu(u64 *output, u64 *input, u64 rows, u64 cols) {
  u64 *device_output, *device_input;
  const u64 num_bytes = rows * cols * sizeof(u64);
  cudaMalloc(&device_output, num_bytes);
  cudaMalloc(&device_input, num_bytes);

  cudaMemcpy(device_input, input, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_output, output, num_bytes, cudaMemcpyHostToDevice);

  dim3 blockDim(32, 32, 1);
  dim3 gridDim(ceildiv(cols, 32), ceildiv(rows, 32), 1);

  // transpose_read_coalesce_kernel<<<gridDim, blockDim>>>(device_output, device_input, rows, cols);
  // transpose_write_coalesce_kernel<<<gridDim, blockDim>>>(device_output, device_input, rows, cols);
  transpose_read_write_coalesce_kernel<<<gridDim, blockDim>>>(device_output, device_input, rows, cols);

  cudaMemcpy(output, device_output, num_bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_output);
  cudaFree(device_input);
}

void transpose_cpu(u64 *output, u64 *input, u64 rows, u64 cols) {
  for (u64 i = 0; i < rows; i++) {
    for (u64 j = 0; j < cols; j++) {
      u64 input_index = j + i * cols;
      u64 output_index = i + j * rows;
      output[output_index] = input[input_index];
    }
  }
}

void generate_matrix(u64 *matrix, u64 rows, u64 cols) {
  for (u64 i = 0; i < rows; i++) {
    for (u64 j = 0; j < cols; j++) {
      u64 index = j + i * cols;
      matrix[index] = index % 11;
    }
  }
}

void print_matrix(u64 *matrix, u64 rows, u64 cols) {
  for (u64 i = 0; i < rows; i++) {
    for (u64 j = 0; j < cols; j++) {
      u64 index = j + i * cols;
      std::cout << matrix[index] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

void verify_matrix(u64 *output, u64 *real, u64 rows, u64 cols) {
  for (u64 i = 0; i < rows; i++) {
    for (u64 j = 0; j < cols; j++) {
      u64 index = j + i * cols;
      if (output[index] != real[index]) {
        std::cout << "Error: ( " << i << " , " << j << " )" << ": " << output[index] << " != " << real[index] << "\n";
          exit(1);
      }
    }
  }
  std::cout << "Correct matrix\n";
}

int main() {
  const u64 rows = 10;
  const u64 cols = 12;
  const u64 num_bytes = rows * cols * sizeof(u64);
  u64 *matrix = (u64 *)malloc(num_bytes);
  u64 *cpu = (u64 *)malloc(num_bytes);
  u64 *gpu = (u64 *)malloc(num_bytes);

  generate_matrix(matrix, rows, cols);
  transpose_cpu(cpu, matrix, rows, cols);
  transpose_gpu(gpu, matrix, rows, cols);

  verify_matrix(gpu, cpu, cols, rows);
  print_matrix(matrix, rows, cols);
  print_matrix(cpu, cols, rows);
  print_matrix(gpu, cols, rows);
  return 0;
}
