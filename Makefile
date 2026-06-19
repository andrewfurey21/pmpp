all: build ./build/vector_add ./build/grayscale ./build/blur ./build/matmul

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

./build/matmul:
	nvcc ./matmul/matmul.cu -o ./build/matmul -g
	./build/matmul

clean:
	rm -rf build
.PHONY: clean
