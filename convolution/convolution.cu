
#include <cuda_runtime.h>
#include <cassert>
#include <driver_types.h>
#include <iostream>

using u64 = unsigned long long;
using f32 = float;
using i32 = int;

__host__ void conv2d_cpu(u64 *output, u64 *input, u64 *kernels, u64 rows, u64 cols, u64 channels, u64 kernel_size, u64 num_kernels) {
  assert(kernel_size % 2 != 0);
  assert(rows >= kernel_size && cols >= kernel_size);
  u64 output_cols = cols - (kernel_size - 1);
  u64 output_rows = rows - (kernel_size - 1);

  for (u64 k = 0; k < num_kernels; k++) {
    for (u64 output_row = 0; output_row < output_rows; output_row++) {
      for (u64 output_col = 0; output_col < output_cols; output_col++) {
        u64 output_index = output_col + output_row * output_cols + k * output_rows * output_cols;
        u64 output_value = 0;
        float perc = (float) output_index / (output_cols * output_rows * num_kernels);
        std::cout << perc * 100.f << "%\n";
        for (u64 c = 0; c < channels; c++) {
          for (u64 i = 0; i < kernel_size; i++) {
            for (u64 j = 0; j < kernel_size; j++) {
              u64 input_index = (output_col + j) + (output_row + i) * cols + c * rows * cols;
              u64 kernel_index = j + i * kernel_size + c * kernel_size * kernel_size + k * channels * kernel_size * kernel_size;
              output_value += input[input_index] * kernels[kernel_index];
            }
          }
        }
        output[output_index] = output_value;
      }
    }
  }
}

namespace pmpp {
  template <typename T>
  __host__ __device__ T ceildiv(T x, T y) {
    return (x + y - 1) / y;
  }
};

// each thread wil operate on 1 patch.
__global__ void naive_conv2d_kernel(u64 *output, u64 *input, u64 *kernels, u64 rows, u64 cols, u64 channels, u64 kernel_size, u64 num_kernels) {
  u64 output_col = threadIdx.x + blockIdx.x * blockDim.x;
  u64 output_row = threadIdx.y + blockIdx.y * blockDim.y;

  u64 kernel = blockIdx.z;

  u64 output_rows = rows - (kernel_size - 1);
  u64 output_cols = cols - (kernel_size - 1);

  if (output_col < output_cols && output_row < output_rows && kernel < num_kernels) {
    u64 output_value = 0;
    u64 output_index = output_col + output_row * output_cols + kernel * output_rows * output_cols;
    for (u64 c = 0; c < channels; c++) {
      for (u64 i = 0; i < kernel_size; i++) {
        for (u64 j = 0; j < kernel_size; j++) {
          u64 input_index = (output_col + j) + (output_row + i) * cols + c * rows * cols;
          u64 kernel_index = j + i * kernel_size + c * kernel_size * kernel_size + kernel * channels * kernel_size * kernel_size;
          output_value += input[input_index] * kernels[kernel_index];
        }
      }
    }
    output[output_index] = output_value;
  }
}


__host__ void naive_conv2d_gpu(u64 *output, u64 *input, u64 *kernels, u64 rows, u64 cols, u64 channels, u64 kernel_size, u64 num_kernels) {
  u64 *device_output, *device_input, *device_kernels;

  u64 shrink = kernel_size - 1;
  u64 output_size_bytes = (rows - shrink) * (cols - shrink) * num_kernels * sizeof(u64);
  u64 input_size_bytes = rows * cols * channels * sizeof(u64);
  u64 kernel_size_bytes = kernel_size * kernel_size * channels * num_kernels * sizeof(u64);

  cudaMalloc(&device_output, output_size_bytes);
  cudaMalloc(&device_input, input_size_bytes);
  cudaMalloc(&device_kernels, kernel_size_bytes);

  cudaMemcpy(device_input, input, input_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_kernels, kernels, kernel_size_bytes, cudaMemcpyHostToDevice);
  // cudaMemset(device_output, 0, output_size_bytes);

  dim3 blockDim(32, 32, 1);
  dim3 gridDim(pmpp::ceildiv(cols, 32ull), pmpp::ceildiv(rows, 32ull), num_kernels);
  naive_conv2d_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_kernels, rows, cols, channels, kernel_size, num_kernels);

  cudaMemcpy(output, device_output, output_size_bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_output);
  cudaFree(device_input);
  cudaFree(device_kernels);
}

void verify(u64 *a, u64 *b, u64 rows, u64 cols, u64 channels) {
  for (int c = 0; c < channels; c++) {
    for (u64 i = 0; i < rows; i++) {
      for (u64 j = 0; j < cols; j++) {
        u64 index = j + i * cols + c * rows * cols;
        if (a[index] != b[index]) {
          std::cout << i << ", " << j << ", " << c << ": ";
          std::cout << a[index] << " != " << b[index] << "\n";
          exit(1);
        }
      }
    }
  }
}

void print_3d_tensor(u64 *a, u64 rows, u64 cols, u64 channels) {
  for (u64 d = 0; d < channels; d++) {
    for (u64 r = 0; r < rows; r++) {
      for (u64 c = 0; c < cols; c++) {
        u64 a_index = c + r * cols + d * rows * cols;
        std::cout << a[a_index] << ", ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
  std::cout << "----------------------\n";
}

void print_4d_tensor(u64 *a, u64 rows, u64 cols, u64 channels, u64 num) {
  for (u64 n = 0; n < num; n++) {

    for (u64 d = 0; d < channels; d++) {
      for (u64 r = 0; r < rows; r++) {
        for (u64 c = 0; c < cols; c++) {
          u64 a_index = c + r * cols + d * rows * cols + n * channels * cols * rows;
          std::cout << a[a_index] << ", ";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
  std::cout << "----------------\n";
}

void generate_3d_tensor(u64 *a, u64 rows, u64 cols, u64 channels) {
  for (u64 c = 0; c < channels; c++) {
    for (u64 i = 0; i < rows; i++) {
      for (u64 j = 0; j < cols; j++) {
        u64 index = j + i * cols + c * rows * cols;
        a[index] = 2;
        a[index] = index % 11;
      }
    }
  }
}

void generate_4d_tensor(u64 *a, u64 rows, u64 cols, u64 channels, u64 num) {
  for (u64 n = 0; n < num; n++) {
    for (u64 c = 0; c < channels; c++) {
      for (u64 i = 0; i < rows; i++) {
        for (u64 j = 0; j < cols; j++) {
          u64 index = j + i * cols + c * rows * cols + n * rows * cols * channels;
          // a[index] = 2;
          a[index] = index % 11;
        }
      }
    }
  }
}

int main() {
  const u64 rows = 256;
  const u64 cols = 256;
  const u64 channels = 256;

  const u64 num_kernels = 512;
  const u64 kernel_size = 7;

  const u64 output_rows = rows - (kernel_size - 1);
  const u64 output_cols = cols - (kernel_size - 1);

  u64 *input = (u64*)malloc(rows * cols * channels * sizeof(u64));
  u64 *kernels = (u64*)malloc(kernel_size * kernel_size * channels * num_kernels * sizeof(u64));

  u64 *cpu = (u64*)malloc(output_rows * output_cols * num_kernels * sizeof(u64));
  u64 *gpu = (u64*)malloc(output_rows * output_cols * num_kernels * sizeof(u64));

  generate_3d_tensor(input, rows, cols, channels);
  generate_4d_tensor(kernels, kernel_size, kernel_size, channels, num_kernels);

  std::cout << "starting conv\n";
  // conv2d_cpu(cpu, input, kernels, rows, cols, channels, kernel_size, num_kernels);
  std::cout << "done cpu conv\n" << std::flush;
  naive_conv2d_gpu(gpu, input, kernels, rows, cols, channels, kernel_size, num_kernels);
  std::cout << "done naive matmul\n" << std::flush;

  // verify(cpu, gpu, output_rows, output_cols, channels);
  // verify(cpu, tiled_gpu, a_rows, b_cols);
  // std::cout << "conv is correct.\n";

  // print_3d_tensor(input, rows, cols, channels);
  // print_4d_tensor(kernels, kernel_size, kernel_size, channels, num_kernels);
  // print_3d_tensor(cpu, output_rows, output_cols, num_kernels);
  // print_3d_tensor(gpu, output_rows, output_cols, num_kernels);


  free(cpu);
  return 0;
}
