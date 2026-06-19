#include <cuda_runtime_api.h>
#include <driver_types.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image.h"
#include "../stb_image_write.h"

#include <cuda_runtime.h>
#include <iostream>

using u64 = unsigned long long;
using i32 = int;
using f32 = float;
using u8 = unsigned char;

namespace pmpp {
__host__ __device__ i32 ceil2(i32 x, i32 y) {
  assert(y % 2 == 0);
  return (x + (y - 1)) / y;
}
};


float *load_rgb_image(char const *name, i32 *width, i32 *height) {
  i32 channels;
  u8 *load_image = stbi_load(name, width, height, &channels, 3);
  f32 *image = (f32*)malloc((*width) * (*height) * 3 * sizeof(f32));

  for (u64 row = 0; row < *width; row++) {
    for (u64 col = 0; col < *height; col++) {
      u64 index = 3 * (col + row * (*height));
      image[index] = (f32)(load_image[index]) / 255.f;
      image[index + 1] = (f32)(load_image[index + 1]) / 255.f;
      image[index + 2] = (f32)(load_image[index + 2]) / 255.f;
    }
  }
  stbi_image_free(load_image);
  return image;
}

void save_rgb_image(char const *name, f32 *image, i32 width, i32 height) {
  u8 *save_image = (u8*)malloc(width * height * 3 * sizeof(u8));

  for (u64 row = 0; row < width; row++) {
    for (u64 col = 0; col < height; col++) {
      u64 index = (col + row * height) * 3;
      save_image[index    ] = (u8)(image[index    ] * 255.f);
      save_image[index + 1] = (u8)(image[index + 1] * 255.f);
      save_image[index + 2] = (u8)(image[index + 2] * 255.f);
    }
  }

  stbi_write_jpg(name, width, height, 3, save_image, 90);
  free(save_image);
}

void blur_cpu(f32 *input, f32 *output, i32 rows, i32 cols, i32 kernel_size) {
  f32 last = 0.0f;
  for (i32 row = 0; row < rows; row++) {
    for (i32 col = 0; col < cols; col++) {
      i32 index = (col + row * cols) * 3;
      output[index] = 0;
      output[index + 1] = 0;
      output[index + 2] = 0;

      f32 perc = ((f32)(index) / 3.0f) / (rows * cols) * 100;
      if (perc - last > 1) {
        last = perc;
        std::cout << perc << "%\n";
      }

      for (i32 i = -(kernel_size / 2); i < pmpp::ceil2(kernel_size, 2); i++) {
        if (row + i < 0 || row + i >= rows) continue;
        for (i32 j = -(kernel_size / 2); j < pmpp::ceil2(kernel_size, 2); j++) {
          if (col + j < 0 || col + j >= cols) continue;

          i32 input_index = ((col + j) + ((row + i) * cols)) * 3;
          output[index] += (input[input_index] / (f32)(kernel_size * kernel_size));
          output[index + 1] += input[input_index + 1] / (f32)(kernel_size * kernel_size);
          output[index + 2] += input[input_index + 2] / (f32)(kernel_size * kernel_size);
        }
      }
    }
  }
}

__global__ void blur_gpu_kernel(f32 *input, f32 *output, i32 rows, i32 cols,
                                i32 kernel_size) {
  i32 row = threadIdx.y + blockIdx.y * blockDim.y;
  i32 col = threadIdx.x + blockIdx.x * blockDim.x;

  if (row < rows && col < cols) {
    i32 channel = blockIdx.z;
    i32 index = (col + row * cols) * 3;
    for (i32 i = -(kernel_size / 2); i < pmpp::ceil2(kernel_size, 2); i++) {
      if (row + i < 0 || row + i >= rows) continue;
      for (i32 j = -(kernel_size / 2); j < pmpp::ceil2(kernel_size, 2); j++) {
        if (col + j < 0 || col + j >= cols) continue;

        i32 input_index = ((col + j) + ((row + i) * cols)) * 3;
        output[index + channel] += (input[input_index + channel] / (f32)(kernel_size * kernel_size));
      }
    }
  }
}

void blur_gpu(f32 *input, f32 *output, i32 rows, i32 cols, i32 kernel_size) {
  u64 image_size = rows * cols * 3 * sizeof(f32);
  f32 *device_input, *device_output;
  cudaMalloc(&device_input, image_size);
  cudaMalloc(&device_output, image_size);
  cudaMemcpy(device_input, input, image_size, cudaMemcpyHostToDevice);
  cudaMemset(device_output, 0, image_size);

  dim3 blockDim(32, 32, 1);
  dim3 gridDim(pmpp::ceil2(cols, 32), pmpp::ceil2(rows, 32), 3);

  blur_gpu_kernel<<<gridDim, blockDim>>>(device_input, device_output, rows, cols, kernel_size);
  cudaMemcpy(output, device_output, image_size, cudaMemcpyDeviceToHost);

  cudaFree(device_input);
  cudaFree(device_output);
}

int main() {
  i32 width, height;
  i32 kernel_size = 100;
  f32 *image = load_rgb_image("./images/dog.jpg", &width, &height);
  f32 *blur_image= (f32 *)malloc(width * height * 3 * sizeof(float));

  // blur_cpu(image, blur_image_cpu, height, width, kernel_size);
  blur_gpu(image, blur_image, height, width, kernel_size);

  save_rgb_image("./build/blur_cpu_dog.jpg", blur_image, width, height);
  free(image);
  free(blur_image);
}
