all: build ./build/vector_add ./build/grayscale ./build/blur ./build/matmul ./build/transpose ./build/conv

build:
	@mkdir -p build

./build/grayscale:
	nvcc ./grayscale/grayscale.cu -g -o ./build/grayscale

./build/vector_add:
	nvcc ./add/vector_add.cu -o ./build/vector_add

./build/blur:
	nvcc ./blur/blur.cu -o ./build/blur -g
	# ./build/blur
	# open ./build/blur_cpu_dog.jpg

./build/matmul: ./matmul/matmul.cu
	nvcc ./matmul/matmul.cu -o ./build/matmul -g -G -O0
	# ./build/matmul

./build/transpose: ./transpose/transpose.cu
	nvcc ./transpose/transpose.cu -o ./build/transpose

./build/conv: ./convolution/convolution.cu
	nvcc ./convolution/convolution.cu -o ./build/conv

clean:
	rm -rf build
.PHONY: clean
