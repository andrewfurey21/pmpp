all: build vector_add

build:
	@mkdir -p build

vector_add:
	nvcc ./add/vector_add.cu -o ./build/vector_add
	./build/vector_add

clean:
	rm -rf build
.PHONY: clean
