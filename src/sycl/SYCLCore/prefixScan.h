#ifndef HeterogeneousCore_SYCLUtilities_interface_prefixScan_h
#define HeterogeneousCore_SYCLUtilities_interface_prefixScan_h

#include <CL/sycl.hpp>
#include <cstdint>

//#include "SYCLCore/syclCompat.h"
#include "SYCLCore/sycl_assert.h"

#ifdef __CUDA_ARCH__

template <typename T>
void __forceinline warpPrefixScan(T const* __restrict__ ci, T* __restrict__ co, uint32_t i, uint32_t mask, sycl::nd_item<3> item) {
  // ci and co may be the same
  auto x = ci[i];
  auto laneId = item.get_local_id(2) & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
}

template <typename T>
void __forceinline warpPrefixScan(T* c, uint32_t i, uint32_t mask, sycl::nd_item<3> item) {
  auto x = c[i];
  auto laneId = item.get_local_id(2) & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

#endif

namespace cms {
  namespace sycltools {

    // 1) limited to 32*32 elements....
    template <typename VT, typename T>
    __forceinline void blockPrefixScan(VT const* ci,
                                       VT* co,
                                       uint32_t size,
                                       T* ws
#ifndef __CUDA_ARCH__
                                       = nullptr
#endif
                                       ,
                                       sycl::nd_item<3> item) {
#ifdef __CUDA_ARCH__ 
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDim.x % 32);
      auto first = item.get_local_id(2);
      auto mask = sycl::reduce_over_group(item.get_sub_group(),
                                          (0xffffffff & (0x1 << item.get_sub_group().get_local_linear_id())) && first < size
                                              ? (0x1 << item.get_sub_group().get_local_linear_id())
                                              : 0,
                                          sycl::ext::oneapi::plus<>());

      for (auto i = first; i < size; i += item.get_local_range().get(2)) {
        warpPrefixScan(ci, co, i, mask, item);
        auto laneId = item.get_local_id(2) & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = co[i];
        mask = sycl::reduce_over_group(item.get_sub_group(),
                                      (mask & (0x1 << item.get_sub_group().get_local_linear_id())) && i + item.get_local_range(2) < size
                                          ? (0x1 << item.get_sub_group().get_local_linear_id())
                                          : 0,
                                      sycl::ext::oneapi::plus<>());
      }
      /*
      2) Output from DPCT:
      Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
      It should improve the performance, can be evaluated in the future
      */
      item.barrier(sycl::access::fence_space::local_space);
      if (size <= 32)
        return;
      if (item.get_local_id(2) < 32)
        warpPrefixScan(ws, item.get_local_id(2), 0xffffffff, item);
      //Same as above (2)
      item.barrier(sycl::access::fence_space::local_space);
      for (auto i = first + 32; i < size; i += item.get_local_range().get(2)) {
        auto warpId = i / 32;
        co[i] += ws[warpId - 1];
      }
      //Same as above (2)
      item.barrier(sycl::access::fence_space::local_space);
#else
      co[0] = ci[0];
      for (uint32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
#endif
    }

    // same as above (1), may remove
    // limited to 32*32 elements....
    template <typename T>
    __forceinline void blockPrefixScan(T* c,
                                         uint32_t size,
                                         T* ws
#ifndef __CUDA_ARCH__
                                         = nullptr
#endif
                                         ,
                                         sycl::nd_item<3> item) {
#ifdef __CUDA_ARCH__
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDim.x % 32);
      auto first = item.get_local_id(2);
      auto mask = sycl::reduce_over_group(item.get_sub_group(),
                                        (0xffffffff & (0x1 << item.get_sub_group().get_local_linear_id())) && first < size
                                            ? (0x1 << item.get_sub_group().get_local_linear_id())
                                            : 0,
                                        sycl::ext::oneapi::plus<>());

      for (auto i = first; i < size; i += item.get_local_range().get(2)) {
        warpPrefixScan(c, i, mask, item);
        auto laneId = item.get_local_id(2) & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = c[i];
        mask = sycl::reduce_over_group(item.get_sub_group(),
                                      (mask & (0x1 << item.get_sub_group().get_local_linear_id())) && i + item.get_local_range(2) < size
                                          ? (0x1 << item.get_sub_group().get_local_linear_id())
                                          : 0,
                                      sycl::ext::oneapi::plus<>());
      }
      //Same as above (2)
      item.barrier(sycl::access::fence_space::local_space);
      if (size <= 32)
        return;
      if (item.get_local_id(2) < 32)
        warpPrefixScan(ws, item.get_local_id(2), 0xffffffff, item);
      //Same as above (2)
      item.barrier(sycl::access::fence_space::local_space);
      for (auto i = first + 32; i < size; i += item.get_local_range().get(2)) {
        auto warpId = i / 32;
        c[i] += ws[warpId - 1];
      }
      //Same as above (2)
      item.barrier(sycl::access::fence_space::local_space);
#else
      for (uint32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
#endif      
    }

#ifdef __CUDA_ARCH__
    // see https://stackoverflow.com/questions/40021086/can-i-obtain-the-amount-of-allocated-dynamic-shared-memory-from-within-a-kernel/40021087#40021087
    __forceinline unsigned dynamic_smem_size() {
      unsigned ret;
      /*
      DPCT1053:10: Migration of device assembly code is not supported.
      */
      asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
      return ret;
    }
#endif

    // in principle not limited....
    template <typename T>
    void multiBlockPrefixScan(T const* ici, T* ico, int32_t size, int32_t* pc, sycl::nd_item<3> item,
                              uint8_t *local_psum, T *ws, bool *isLastBlockDone) {
      volatile T const* ci = ici;
      volatile T* co = ico;

#ifdef __CUDA_ARCH__
      assert(sizeof(T) * gridDim.x <= dynamic_smem_size());  // size of psum below
#endif
      assert(blockDim.x * gridDim.x >= size);
      // first each block does a scan
      int off = item.get_local_range().get(2) * item.get_group(2);
      if (size - off > 0)
        blockPrefixScan(ci + off, co + off, std::min(int(item.get_local_range().get(2)), size - off), ws, item);

      // count blocks that finished

      if (0 == item.get_local_id(2)) {
        /*
        DPCT1078:13: Consider replacing memory_order::acq_rel with memory_order::seq_cst for correctness if strong memory order restrictions are needed.
        */
        sycl::ext::oneapi::atomic_fence(sycl::ext::oneapi::memory_order::acq_rel,
                                        sycl::ext::oneapi::memory_scope::device);
        /*
        DPCT1039:14: The generated code assumes that "pc" points to the global memory address space. If it points to a local memory address space, replace "sycl::global_ptr" with "sycl::local_ptr".
        */
        auto value = sycl::atomic<int32_t>(sycl::global_ptr<int32_t>(pc)).fetch_add(1);  // block counter
        *isLastBlockDone = (value == (int(item.get_group_range(2)) - 1));
      }

      //Same as above (2)
      item.barrier(sycl::access::fence_space::local_space);

      if (!(*isLastBlockDone))
        return;

      assert(int(gridDim.x) == *pc);

      // good each block has done its work and now we are left in last block

      // let's get the partial sums from each block
      auto psum = (T*)local_psum;
      for (int i = item.get_local_id(2), ni = item.get_group_range(2); i < ni;
           i += item.get_local_range().get(2)) {
        auto j = item.get_local_range().get(2) * i + item.get_local_range().get(2) - 1;
        psum[i] = (j < size) ? co[j] : T(0);
      }
      //Same as above (2)
      item.barrier(sycl::access::fence_space::local_space);
      blockPrefixScan(psum, psum, item.get_group_range(2), item, ws);

      // now it would have been handy to have the other blocks around...
      for (int i = item.get_local_id(2) + item.get_local_range().get(2), k = 0; i < size;
           i += item.get_local_range().get(2), ++k) {
        co[i] += psum[k];
      }
    }
  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_SYCLUtilities_interface_prefixScan_h
