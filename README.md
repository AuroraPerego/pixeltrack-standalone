# Compile and link CUDA code using clang++

##Quick recipe
```bash
make environment
source env.sh
make -j `nproc` cuda
./cuda 
```

## Differences in the code
- inline PTX compiled with nvcc `needs` one `%`, with `clang++` needs `%%`:
  ```cpp
    // nvcc
    asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    // clang
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(ret));
  ```
- if there is the possibility to call a function from the host, it must  have the attribute `__host__` in `clang++`. In CUDA it compiles even without it (it will probably breaks if the function is actually called from the host at some point).
- due to a bug in LLVM with shared variables, two `__syncthreads()` in `radixSort.h` have been replaced with `__threadfence()`

## How to compile and link
- Compile files `.cu` with `clang++`:
  ```bash
  clang++-14 \
    -x cuda \           
    -std=c++17 \
    -O3 \
    -g \
    -fPIC \
    -include noinline.h \   # bug in CUDA/clang++?
    -I<CUDA install path>/include \ 
    --cuda-gpu-arch=<GPU arch> \           
    --cuda-gpu-arch=sm_50 \
    --cuda-gpu-arch=sm_60 \
    --cuda-gpu-arch=sm_70 \
    -Wno-deprecated-declarations \
    -c file.cu \
    -o file.cu.o \
    -MMD
  ```
  where:
  - `<CUDA install path>`: the directory where you installed CUDA. Typically, `/usr/local/cuda`.
  - `<GPU arch>`: the compute capability of your GPU. For example, if you want to run your program on a GPU with compute capability of 3.5, specify `--cuda-gpu-arch=sm_35`.
  - The file noinline.h is the following:
  ```cpp
  #if defined(__clang__) && defined(__CUDA__)
  #undef __noinline__
  #endif
  ```
  and is a workaround for an error at compile time(see below).

- Compile file `.cc` with `g++`
- Link with `clang++`:
  ```bash
  clang++-14  \
    # the .o files
    -std=c++17 \
    -O3 \
    -g \
    -fPIC \
    -pthread \                #  fPIC stands for "force Position Independent Code": enables one to share built library which has dependencies on other shared libraries. 
    -Wl,-E \                  # Pass "-E" as an option to the linker
    -lstdc++fs \              # filesystem library
    -shared \                 # Produce a shared object which can then be linked with other objects to form an executable
    -Wl,-z,defs \             # Pass "-z,defs" as an option to the linker
                              # -z <arg>   Pass -z <arg> to the linker
    -lcudart \                # CUDA runtime library
    -ldl \                    # Dynamically loaded libraries
    -lrt \                    # Realtime Extensions library
    -include noinline.h \     # bug in CUDA/clang++?
    -Wno-deprecated-declarations \ 
    -L<CUDA install path>/lib64 \
    -o shared_object.so
  ```
- final link at the end with `g++`
  ```bash
  g++ \
   # file .o
   -O2 \
   -fPIC \
   -pthread \
   -Wl,-E \
   -lstdc++fs \
   -ldl \
   -Wl,-rpath,/path/to/.../lib/cuda \
   -o cuda \
   -L... \                  # externals
   -l... \                  # shared libraries fw
   -L<CUDA install path>/lib64 \
   -lcudart \
   -ldl 
   ```

## About the error at compile time with `__noinline__`
Consider this file, `test.cu`:
```cpp
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

__attribute__((__noinline__))
void f(){
   std::cout << "Hello World :)" << std::endl;
}

int main(){
  f();
  return 0;
}
```
This compiles and runs with `g++` and `nvcc`:
```bash
$ g++ -x c++ -O2 -std=c++17 noinline.cu -o noinline
$ ./noinline
Hello World :)
$ /usr/local/cuda/bin/nvcc -O2 -std=c++17 noinline.cu -o noinline
$ ./noinline
Hello World :)
```
However, with `clang++` this raises an error:
```bash
$ clang++-14 -x cuda -O2 -std=c++17 --cuda-path=/usr/local/cuda-11.5.0 noinline.cu -o noinline
noinline.cu:7:16: error: use of undeclared identifier 'noinline'; did you mean 'inline'?
__attribute__((__noinline__))
               ^
/usr/local/cuda-11.5.0/include/crt/host_defines.h:83:24: note: expanded from macro '__noinline__'
        __attribute__((noinline))
                       ^
noinline.cu:7:16: error: type name does not allow function specifier to be specified
/usr/local/cuda-11.5.0/include/crt/host_defines.h:83:24: note: expanded from macro '__noinline__'
        __attribute__((noinline))
                       ^
noinline.cu:7:16: error: expected expression
/usr/local/cuda-11.5.0/include/crt/host_defines.h:83:33: note: expanded from macro '__noinline__'
        __attribute__((noinline))
                                ^
3 errors generated when compiling for sm_35.
```

A workaround for this is to include a file like the following:
```cpp
#if defined(__clang__) && defined(__CUDA__)
#undef __noinline__
#endif
```
Compiling with the additional `-include file.h` will solve the error:
```
$ clang++-14 -x cuda -O2 -std=c++17 --cuda-path=/usr/local/cuda-11.5.0 -include ../alpaka_sycl/pixeltrack-standalone/noinline.h noinline.cu -o noinline
$ ./noinline
Hello World :)
```

clang++: clang version 14.0.6
CUDA: 11.5.0
(same behaviour with the nightly of 15/12 and CUDA 11.7)

## Links:
- [LLVM - CUDA with clang](https://llvm.org/docs/CompileCudaWithLLVM.html)
- [#57544 llvm issue](https://github.com/llvm/llvm-project/issues/57544)
- [Separate Compilation and Linking of CUDA C++ Device Code with nvcc, nvlink and clang](https://github.com/fwyzard/cuda-linking)
- [Makefile below](https://stackoverflow.com/questions/67070926/struggling-with-cuda-clang-and-llvm-ir-and-getting-cuda-failure-invalid-dev)

```bash
BIN_FILE=test
SRC_FILE=$(BIN_FILE).cu

test: $(BIN_FILE)

CUDA_BASE := /usr/local/cuda-11.5.0
CUDA_BIN  := $(CUDA_BASE)/bin
CUDA_LIB  := $(CUDA_BASE)/lib64
CUDA_ARCH := 61

# Host Side
$(BIN_FILE).ll: $(SRC_FILE) $(BIN_FILE).fatbin
        clang++ -Wall -Werror $(BIN_FILE).cu -march=x86-64 --cuda-host-only --cuda-path=$(CUDA_BASE) -relocatable-pch -Xclang -fcuda-include-gpubinary -Xclang $(BIN_FILE).fatbin -S -g -c -emit-llvm

$(BIN_FILE).o: $(BIN_FILE).ll
        llc -march=x86-64 $(BIN_FILE).ll -o $(BIN_FILE).s
        clang++ -c -Wall --cuda-path=$(CUDA_BASE) $(BIN_FILE).s -o $(BIN_FILE).o

# GPU Side
$(BIN_FILE)-cuda-nvptx64-nvidia-cuda-sm_$(CUDA_ARCH).ll: $(SRC_FILE)
        clang++ -x cuda --cuda-path=$(CUDA_BASE) -Wall -Werror $(BIN_FILE).cu --cuda-device-only --cuda-gpu-arch=sm_$(CUDA_ARCH) -S -g -emit-llvm

$(BIN_FILE).ptx: $(BIN_FILE)-cuda-nvptx64-nvidia-cuda-sm_$(CUDA_ARCH).ll
        llc -march=nvptx64 -mcpu=sm_$(CUDA_ARCH) -mattr=+ptx64 $(BIN_FILE)-cuda-nvptx64-nvidia-cuda-sm_$(CUDA_ARCH).ll -o $(BIN_FILE).ptx

$(BIN_FILE).ptx.o: $(BIN_FILE).ptx
        $(CUDA_BIN)/ptxas -m64 --gpu-name=sm_$(CUDA_ARCH) $(BIN_FILE).ptx -o $(BIN_FILE).ptx.o

$(BIN_FILE).fatbin: $(BIN_FILE).ptx.o
        $(CUDA_BIN)/fatbinary --64 --create $(BIN_FILE).fatbin --image=profile=sm_$(CUDA_ARCH),file=$(BIN_FILE).ptx.o --image=profile=compute_$(CUDA_ARCH),file=$(BIN_FILE).ptx -link

$(BIN_FILE)_dlink.o: $(BIN_FILE).fatbin
        $(CUDA_BIN)/nvcc $(BIN_FILE).fatbin -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -dlink -o $(BIN_FILE)_dlink.o -lcudart -lcudart_static -lcudadevrt

# Link both object files together (either nvcc or clang works here):
$(BIN_FILE): $(BIN_FILE).o $(BIN_FILE)_dlink.o
        #$(CUDA_BIN)/nvcc $(BIN_FILE).o $(BIN_FILE)_dlink.o -o $(BIN_FILE) -arch=sm_$(CUDA_ARCH)
        clang++ --cuda-path=$(CUDA_BASE) $(BIN_FILE).o $(BIN_FILE)_dlink.o -o $(BIN_FILE) -lcuda -lcudart -lcudadevrt -L$(CUDA_LIB)

clean:
        rm -fR $(BIN_FILE).fatbin $(BIN_FILE).ll $(BIN_FILE).o $(BIN_FILE).ptx $(BIN_FILE).ptx.o $(BIN_FILE).s $(BIN_FILE)_dlink.o $(BIN_FILE)-cuda-nvptx64-nvidia-cuda-sm_$(CUDA_ARCH).ll $(BIN_FILE)
```
