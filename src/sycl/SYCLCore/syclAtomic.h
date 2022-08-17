#ifndef HeterogeneousCore_SYCLUtilities_interface_syclAtomic_h
#define HeterogeneousCore_SYCLUtilities_interface_syclAtomic_h

// TODO add here also atomicCAS

#include <CL/sycl.hpp>
#include <cstdint>

namespace cms {
  namespace sycltools {

    //from the DPCT library
    template <typename T>
    T shift_sub_group_right(sycl::sub_group g, T x, unsigned int delta,
                            int logical_sub_group_size = 32) {
      unsigned int id = g.get_local_linear_id();
      unsigned int start_index =
          id / logical_sub_group_size * logical_sub_group_size;
      T result = sycl::shift_group_right(g, x, delta);
      if ((id - start_index) < delta) {
        result = x;
      }
      return result;
    }

    //analog of cuda atomicAdd
    template <typename A, typename B>
    inline A AtomicAdd(A* i, B j){
      sycl::atomic_ref<A, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(*i);
      return first.fetch_add(j);
    }

    template <typename A, typename B>
    inline A AtomicSub(A* i, B j){
      sycl::atomic_ref<A, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(*i);
      return first.fetch_add(-j);
    }

    template <typename A, typename B>
    inline A AtomicMin(A* i, B j){
      sycl::ext::oneapi::atomic_ref<A, sycl::ext::oneapi::memory_order::relaxed,
                                sycl::ext::oneapi::memory_scope::work_group,
                                sycl::access::address_space::local_space> first(*i);
      return first.fetch_min(j);
      //sycl::atomic<A>(sycl::global_ptr<A>(i)).fetch_min(j);
    }

    template <typename A, typename B>
    inline A AtomicMax(A* i, B j){
      sycl::ext::oneapi::atomic_ref<A, sycl::ext::oneapi::memory_order::relaxed,
                                sycl::ext::oneapi::memory_scope::work_group,
                                sycl::access::address_space::local_space> first(*i);
      return first.fetch_max(j); 
      //sycl::atomic<A>(sycl::global_ptr<A>(i)).fetch_max(j);
    }

    template <typename A, typename B>
    inline A AtomicInc(A* i, B j){
      auto ret = *i;
      sycl::atomic_ref<A, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(*i);
      first.fetch_add(-j);
      if (*i < 0){
          sycl::atomic_ref<A, sycl::memory_order::relaxed, sycl::memory_scope::work_group> second(*i);
          second.fetch_add(1);
      }
      first.fetch_add(j);
      return ret;
    }
  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_SYCLUtilities_interface_syclAtomic_h
