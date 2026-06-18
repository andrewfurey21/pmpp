all: build vector_add grayscale

build:
	@mkdir -p build

vector_add:
	nvcc ./add/vector_add.cu -o ./build/vector_add
	./build/vector_add
	nvcc ./grayscale/grayscale.cu -g -o ./build/grayscale
	./build/grayscale
	open ./build/grayscale_dog_cpu.jpg
	open ./build/grayscale_dog_gpu.jpg


clean:
	rm -rf build
.PHONY: clean
