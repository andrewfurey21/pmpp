#include <driver_types.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image.h"
#include "../stb_image_write.h"

using u64 = unsigned long long;
using i32 = int;
using f32 = float;
using u8 = unsigned char;

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

void save_grayscale_image(char const *name, f32 *image, i32 width, i32 height) {
  u8 *save_image = (u8*)malloc(width * height * 3 * sizeof(u8));

  for (u64 row = 0; row < width; row++) {
    for (u64 col = 0; col < height; col++) {
      u64 index = col + row * height;
      save_image[3 * index] = (u8)(image[index] * 255.f);
      save_image[3 * index + 1] = (u8)(image[index] * 255.f);
      save_image[3 * index + 2] = (u8)(image[index] * 255.f);
    }
  }

  stbi_write_jpg(name, width, height, 3, save_image, 90);
  free(save_image);
}

void grayscale_cpu(f32 *input, f32 *output, u64 rows, u64 cols) {
  for (u64 row = 0; row < rows; row++) {
    for (u64 col = 0; col < cols; col++) {
      u64 index = col + row * cols;
      u64 red   = 3 * index;
      u64 blue  = 3 * index + 1;
      u64 green = 3 * index + 2;
      output[index] = input[red] * 0.21f + input[green] * 0.71f + input[blue] * 0.07f;
    }
  }
}

namespace pmpp {
inline i32 ceil2(i32 x, i32 y) {
  assert(y % 2 == 0);
  return (x + (y - 1)) / y;
}
};

__global__ void grayscale_gpu_kernel(f32 *input, f32 *output, u64 rows, u64 cols) {
  u64 x_index = threadIdx.x + blockIdx.x * blockDim.x;
  u64 y_index = threadIdx.y + blockIdx.y * blockDim.y;

  if (x_index < cols && y_index < rows) {
    u64 out_index = x_index + y_index * cols;
    u64 in_index = out_index * 3;
    output[out_index] = 0.21f * input[in_index] + 0.71f * input[in_index + 1] + 0.07f * input[in_index + 2];
  }
}

void grayscale_gpu(f32 *input, f32 *output, u64 rows, u64 cols) {
  f32 *device_input, *device_output;
  u64 input_size = rows * cols * 3 * sizeof(f32);
  u64 output_size = rows * cols * sizeof(f32);
  cudaMalloc(&device_input, input_size);
  cudaMalloc(&device_output, output_size);
  cudaMemcpy(device_input, input, input_size, cudaMemcpyHostToDevice);

  dim3 blockDim(32, 32, 1);
  dim3 gridDim(pmpp::ceil2(cols, blockDim.x), pmpp::ceil2(rows, blockDim.y), 1);
  grayscale_gpu_kernel<<<gridDim, blockDim>>>(device_input, device_output, rows, cols);

  cudaMemcpy(output, device_output, output_size, cudaMemcpyDeviceToHost);
  cudaFree(device_input);
  cudaFree(device_output);
}

int main() {
  i32 width, height;
  f32 *image = load_rgb_image("./images/dog.jpg", &width, &height);
  f32 *grayscale_image_cpu = (f32 *)malloc(width * height * sizeof(float));
  f32 *grayscale_image_gpu = (f32 *)malloc(width * height * sizeof(float));

  grayscale_cpu(image, grayscale_image_cpu, width, height);
  grayscale_gpu(image, grayscale_image_gpu, width, height);

  save_grayscale_image("./build/grayscale_dog_cpu.jpg", grayscale_image_cpu, width, height);
  save_grayscale_image("./build/grayscale_dog_gpu.jpg", grayscale_image_gpu, width, height);
  free(image);
  free(grayscale_image_cpu);
  free(grayscale_image_gpu);
}
