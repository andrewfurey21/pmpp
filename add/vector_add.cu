
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

using u64 = unsigned long long;
using f32 = float;

__host__ void vector_add_cpu(f32 *a, f32 *b, f32 *c, u64 size) {
  for (u64 i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

__global__ void vector_add_kernel(f32 *a, f32 *b, f32 *c, u64 size) {
  const u64 i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    c[i] = b[i] + a[i];
  }
}

__host__ void vector_add_gpu(f32 *a, f32 *b, f32 *c, u64 size) {
  f32 *device_a, *device_b, *device_c;
  const u64 num_bytes = sizeof(f32) * size;
  cudaMalloc(&device_a, num_bytes);
  cudaMalloc(&device_b, num_bytes);
  cudaMalloc(&device_c, num_bytes);

  cudaMemcpy(device_a, a, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b, num_bytes, cudaMemcpyHostToDevice);

  vector_add_kernel<<<(size + 255) / 256, 256>>>(device_a, device_b, device_c, size);

  cudaMemcpy(c, device_c, num_bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
}

void verify(f32 *a, f32 *b, u64 size) {
  for (u64 i = 0; i < size; i++) {
    std::cerr << i << ": " << a[i] << ", " << b[i] << "\n";
    assert(a[i] == b[i]);
  }
}

int main() {
  const u64 size = 4;
  f32 a[size] = {1, 2, 3, 4};
  f32 b[size] = {1, 2, 3, 4};

  f32 result_cpu[size]{};
  f32 result_gpu[size]{};

  vector_add_cpu(a, b, result_cpu, size);
  vector_add_gpu(a, b, result_gpu, size);

  verify(result_cpu, result_gpu, size);

  std::cout << "Vector add is correct.\n";

  return 0;
}
