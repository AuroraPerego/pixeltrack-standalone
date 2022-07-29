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
      return sycl::atomic<A>(sycl::global_ptr<A>(i)).fetch_add(j);
    }
    
    template <typename A, typename B>
    inline A AtomicAdd(A i, B j){
      return sycl::atomic<A>(sycl::global_ptr<A>(&i)).fetch_add(j);
    }
  
    template <typename A, typename B>
    inline A AtomicSub(A* i, B j){
      return sycl::atomic<A>(sycl::global_ptr<A>(i)).fetch_add(-j);
    }

    template <typename A, typename B>
    inline A AtomicSub(A i, B j){
      return sycl::atomic<A>(sycl::global_ptr<A>(&i)).fetch_add(-j);
    }

    template <typename A, typename B>
    inline A AtomicMin(A* i, B j){
      return sycl::atomic<A, sycl::access::address_space::local_space>(sycl::local_ptr<A>(i)).fetch_min(j);
    }

    template <typename A, typename B>
    inline A AtomicMin(A i, B j){
      return sycl::atomic<A, sycl::access::address_space::local_space>(sycl::local_ptr<A>(&i)).fetch_min(j);
    }

    template <typename A, typename B>
    inline A AtomicMax(A* i, B j){
      return sycl::atomic<A, sycl::access::address_space::local_space>(sycl::local_ptr<A>(i)).fetch_max(j);
    }

    template <typename A, typename B>
    inline A AtomicMax(A i, B j){
      return sycl::atomic<A, sycl::access::address_space::local_space>(sycl::local_ptr<A>(&i)).fetch_max(j);
    }

    template <typename A, typename B>
    inline A AtomicInc(A* i, B j){
      auto ret = *i;
      sycl::atomic<A>(sycl::global_ptr<A>(i)).fetch_add(-j);
      if (*i < 0)
        sycl::atomic<A>(sycl::global_ptr<A>(i)).fetch_add(1);
      sycl::atomic<A>(sycl::global_ptr<A>(i)).fetch_add(j);
      return ret;
    }
  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_SYCLUtilities_interface_syclAtomic_h
