#ifndef HeterogeneousCore_SYCLUtilities_interface_prefixScan_h
#define HeterogeneousCore_SYCLUtilities_interface_prefixScan_h

#include <CL/sycl.hpp>
#include <cstdint>

//#include "SYCLCore/syclCompat.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"

//#ifdef __CUDA_ARCH__

template <typename T>
void __forceinline warpPrefixScan(T const* __restrict__ ci, T* __restrict__ co, uint32_t i, uint32_t mask, sycl::nd_item<1> item) {
  // ci and co may be the same
  auto x = ci[i];
  int laneId = item.get_local_id(0) & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    /*
    DPCT1023:17: The DPC++ sub-group does not support mask options for sycl::shift_group_right.
    */
    auto y = sycl::shift_group_right(item.get_sub_group(), x, offset); //FIXME_ it was __shfl_up_sync
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
}

template <typename T>
void __forceinline warpPrefixScan(T* c, uint32_t i, uint32_t mask, sycl::nd_item<1> item) {
  auto x = c[i];
  int laneId = item.get_local_id(0) & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    /*
    DPCT1023:17: The DPC++ sub-group does not support mask options for sycl::shift_group_right.
    */
    auto y = sycl::shift_group_right(item.get_sub_group(), x, offset); //FIXME_ it was __shfl_up_sync
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

//#endif

namespace cms {
  namespace sycltools {

    // 1) limited to 32*32 elements....
    template <typename VT, typename T>
    __forceinline void blockPrefixScan(VT const* ci,
                                       VT* co,
                                       uint32_t size,
                                       sycl::nd_item<1> item,
				                                T* ws) {
//#ifdef __CUDA_ARCH__
      assert(ws);
      assert(size <= 1024);
      assert(0 == item.get_local_range(0) % 32);
      auto first = item.get_local_id(0);
      auto mask = sycl::reduce_over_group(item.get_sub_group(),
                                          (0xffffffff & (0x1 << item.get_sub_group().get_local_linear_id())) && first < size
                                              ? (0x1 << item.get_sub_group().get_local_linear_id())
                                              : 0,
                                          sycl::ext::oneapi::plus<>());

      for (auto i = first; i < size; i += item.get_local_range(0)) {
        warpPrefixScan(ci, co, i, mask, item);
        int laneId = item.get_local_id(0) & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = co[i];
        mask = sycl::reduce_over_group(item.get_sub_group(),
                                      (mask & (0x1 << item.get_sub_group().get_local_linear_id())) && i + item.get_local_range(0) < size
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
      if (item.get_local_id(0) < 32)
        warpPrefixScan(ws, item.get_local_id(0), 0xffffffff, item);
      //Same as above (0)
      item.barrier(sycl::access::fence_space::local_space);
      for (auto i = first + 32; i < size; i += item.get_local_range(0)) {
        auto warpId = i / 32;
        co[i] += ws[warpId - 1];
      }
      //Same as above (0)
      item.barrier(sycl::access::fence_space::local_space);
      //CUDA_ARCH
      //co[0] = ci[0];
      //for (uint32_t i = 1; i < size; ++i)
      //  co[i] = ci[i] + co[i - 1];

    }

    // same as above (1), may remove
    // limited to 32*32 elements....
    template <typename T>
    __forceinline void blockPrefixScan(T* c,
                                         uint32_t size,
                                         sycl::nd_item<1> item,
                                         T* ws) {
      assert(ws);
      assert(size <= 1024);
      assert(0 == item.get_local_range(0) % 32);
      auto first = item.get_local_id(0);
      auto mask = sycl::reduce_over_group(item.get_sub_group(),
                                        (0xffffffff & (0x1 << item.get_sub_group().get_local_linear_id())) && first < size
                                            ? (0x1 << item.get_sub_group().get_local_linear_id())
                                            : 0,
                                        sycl::ext::oneapi::plus<>());

      for (auto i = first; i < size; i += item.get_local_range(0)) {
        warpPrefixScan(c, i, mask, item);
        int laneId = item.get_local_id(0) & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = c[i];
        mask = sycl::reduce_over_group(item.get_sub_group(),
                                      (mask & (0x1 << item.get_sub_group().get_local_linear_id())) && i + item.get_local_range(0) < size
                                          ? (0x1 << item.get_sub_group().get_local_linear_id())
                                          : 0,
                                      sycl::ext::oneapi::plus<>());
      }
      //Same as above (0)
      item.barrier(sycl::access::fence_space::local_space);
      if (size <= 32)
        return;
      if (item.get_local_id(0) < 32)
        warpPrefixScan(ws, item.get_local_id(0), 0xffffffff, item);
      //Same as above (0)
      item.barrier(sycl::access::fence_space::local_space);
      for (auto i = first + 32; i < size; i += item.get_local_range(0)) {
        auto warpId = i / 32;
        c[i] += ws[warpId - 1];
      }
      //Same as above (0)
      item.barrier(sycl::access::fence_space::local_space);

      //__CUDA_ARCH__
      //for (uint32_t i = 1; i < size; ++i)
      //  c[i] += c[i - 1];

    }

    // see https://stackoverflow.com/questions/40021086/can-i-obtain-the-amount-of-allocated-dynamic-shared-memory-from-within-a-kernel/40021087#40021087
    __forceinline unsigned dynamic_smem_size() {
      unsigned ret;
      //FIXME_ I added two %% instead of one % to make it work
      asm volatile("mov.u32 %%0, %%dynamic_smem_size;" : "=r"(ret));
      return ret;
    }


    // in principle not limited....
    template <typename T>
    void multiBlockPrefixScan(T const* ici, T* ico, int32_t size, int32_t* pc, sycl::nd_item<1> item,
                              uint8_t *local_psum, T *ws, bool *isLastBlockDone) {
      volatile T const* ci = ici;
      volatile T* co = ico;

      assert(sizeof(T) * item.get_group_range(0) <= dynamic_smem_size());  // size of psum below
      assert((int32_t)(item.get_local_range(0) * item.get_group_range(0)) >= size);
      // first each block does a scan
      int off = item.get_local_range(0) * item.get_group(0);
      if (size - off > 0)
        blockPrefixScan(ci + off, co + off, std::min(int(item.get_local_range(0)), size - off), item, ws);

      // count blocks that finished

      if (0 == item.get_local_id(0)) {
        /*
        DPCT1078:13: Consider replacing memory_order::acq_rel with memory_order::seq_cst for correctness if strong memory order restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
        /*
        DPCT1039:14: The generated code assumes that "pc" points to the global memory address space. If it points to a local memory address space, replace "sycl::global_ptr" with "sycl::local_ptr".
        */
        auto value =cms::sycltools::atomic_fetch_add<int32_t>(pc, static_cast<int32_t>(1));
        //auto value = cms::sycltools::AtomicAdd(pc, 1);  // block counter
        *isLastBlockDone = (value == (int(item.get_group_range(0)) - 1));
      }

      //Same as above (0)
      item.barrier(sycl::access::fence_space::local_space);

      if (!(*isLastBlockDone))
        return;

      assert(int(item.get_group_range(0)) == *pc);

      // good each block has done its work and now we are left in last block

      // let's get the partial sums from each block
      auto psum = (T*)local_psum;
      for (int i = item.get_local_id(0), ni = item.get_group_range(0); i < ni;
           i += item.get_local_range(0)) {
        int32_t j = item.get_local_range(0) * i + item.get_local_range(0) - 1;
        psum[i] = (j < size) ? co[j] : T(0);
      }
      //Same as above (0)
      item.barrier(sycl::access::fence_space::local_space);
      blockPrefixScan(psum, psum, item.get_group_range(0), item, ws);

      // now it would have been handy to have the other blocks around...
      for (int i = item.get_local_id(0) + item.get_local_range(0), k = 0; i < size;
           i += item.get_local_range(0), ++k) {
        co[i] += psum[k];
      }
    }
  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_SYCLUtilities_interface_prefixScan_h
