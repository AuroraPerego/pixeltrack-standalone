#ifndef HeterogeneousCore_SYCLUtilities_interface_syclAtomic_h
#define HeterogeneousCore_SYCLUtilities_interface_syclAtomic_h

// TODO add here also atomicCAS

#include <CL/sycl.hpp>
#include <cstdint>

namespace cms {
  namespace sycltools {

    //analog of cuda atomicAdd
    template <typename A, typename B>
    inline A AtomicAdd(A* i, B j){
      sycl::atomic_ref<A*, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(i);
      return *(first.fetch_add(j));
    }
    
    template <typename A, typename B>
    inline A AtomicAdd(A i, B j){
      sycl::atomic_ref<A*, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(&i);
      return *(first.fetch_add(j));
    }
  
    template <typename A, typename B>
    inline A AtomicSub(A* i, B j){
      sycl::atomic_ref<A*, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(i);
      return *(first.fetch_add(-j));
    }

    template <typename A, typename B>
    inline A AtomicSub(A i, B j){
      sycl::atomic_ref<A*, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(&i);
      return *(first.fetch_add(-j));
    }

    template <typename A, typename B>
    inline A AtomicMin(A* i, B j){
      sycl::atomic_ref<A*, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(i);
      return *(first.fetch_add(j));
    }

    template <typename A, typename B>
    inline A AtomicMin(A i, B j){
      sycl::atomic_ref<A*, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(&i);
      return *(first.fetch_add(j));
    }

    template <typename A, typename B>
    inline A AtomicMax(A* i, B j){
      sycl::atomic_ref<A*, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(i);
      return *(first.fetch_add(j));
    }

    template <typename A, typename B>
    inline A AtomicMax(A i, B j){
      sycl::atomic_ref<A*, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(&i);
      return *(first.fetch_add(j));
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
