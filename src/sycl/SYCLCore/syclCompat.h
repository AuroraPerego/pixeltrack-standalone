#ifndef HeterogeneousCore_SYCLUtilities_interface_syclCompat_h
#define HeterogeneousCore_SYCLUtilities_interface_syclCompat_h

#include <CL/sycl.hpp>
/*
 * Everything you need to run sycl code in plain sequential c++ code
 */

#ifndef SYCL_LANGUAGE_VERSION

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace cms {
  namespace syclcompat {

#ifndef __SYCL_HPP__ //FIXME it was __CUDA_RUNTIME_H__
    struct sycl::range<3> {  //FIXME idk if a struct can have this name
      uint32_t x, y, z;
    };
#endif
    const sycl::range<3> threadIdx = {0, 0, 0};
    const sycl::range<3> blockDim = {1, 1, 1};

    extern sycl::range<3> blockIdx;
    extern sycl::range<3> gridDim;

    template <typename T1, typename T2>
    T1 atomicCAS(T1* address, T1 compare, T2 val) {
      T1 old = *address;
      *address = old == compare ? val : old;
      return old;
    }

    template <typename T1, typename T2>
    T1 atomicInc(T1* a, T2 b) {
      auto ret = *a;
      if ((*a) < T1(b))
        (*a)++;
      return ret;
    }

    template <typename T1, typename T2>
    T1 atomicAdd(T1* a, T2 b) {
      auto ret = *a;
      (*a) += b;
      return ret;
    }

    template <typename T1, typename T2>
    T1 atomicSub(T1* a, T2 b) {
      auto ret = *a;
      (*a) -= b;
      return ret;
    }

    template <typename T1, typename T2>
    T1 atomicMin(T1* a, T2 b) {
      auto ret = *a;
      *a = std::min(*a, T1(b));
      return ret;
    }
    template <typename T1, typename T2>
    T1 atomicMax(T1* a, T2 b) {
      auto ret = *a;
      *a = std::max(*a, T1(b));
      return ret;
    }

    //inline void __syncthreads() {}
    //inline void __threadfence() {}
    //inline bool __syncthreads_or(bool x) { return x; }
    //inline bool __syncthreads_and(bool x) { return x; }
    template <typename T>
    inline T __ldg(T const* x) {
      return *x;
    }

    inline void resetGrid() {
      blockIdx = {0, 0, 0};
      gridDim = {1, 1, 1};
    }

  }  // namespace syclcompat
}  // namespace cms

// some  not needed as done by cuda runtime...
//#ifndef __CUDA_RUNTIME_H__
//#define __host__
//#define __device__
//#define __global__
//#define __shared__
//#define __forceinline__
//#endif

// make sure function are inlined to avoid multiple definition
//#ifndef __CUDA_ARCH__
//#undef __global__
//#define __global__ inline __attribute__((always_inline))
//#undef __forceinline__
//#define __forceinline__ inline __attribute__((always_inline))
//#endif

#ifndef __CUDA_ARCH__
using namespace cms::syclcompat;
#endif

#endif

#endif  // HeterogeneousCore_SYCLUtilities_interface_syclCompat_h
