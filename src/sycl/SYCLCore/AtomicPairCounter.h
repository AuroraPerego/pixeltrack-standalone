#ifndef HeterogeneousCore_SYCLUtilities_interface_AtomicPairCounter_h
#define HeterogeneousCore_SYCLUtilities_interface_AtomicPairCounter_h

#include <CL/sycl.hpp>
#include <cstdint>

//#include "SYCLCore/syclCompat.h"

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

    class AtomicPairCounter {
    public:
      using c_type = unsigned long long int;

      AtomicPairCounter() {}
      AtomicPairCounter(c_type i) { counter.ac = i; }

      AtomicPairCounter& operator=(c_type i) {
        counter.ac = i;
        return *this;
      }

      struct Counters {
        uint32_t n;  // in a "One to Many" association is the number of "One"
        uint32_t m;  // in a "One to Many" association is the total number of associations
      };

      union Atomic2 {
        Counters counters;
        c_type ac;
      };

      static constexpr c_type incr = 1UL << 32;

      Counters get() const { return counter.counters; }

      // increment n by 1 and m by i.  return previous value
      __forceinline Counters add(uint32_t i) {
        c_type c = i;
        c += incr;
        Atomic2 ret;
#ifdef __CUDA_ARCH__
        ret.ac = atomicAdd<cms::sycltools::AtomicPairCounter::c_type>(&counter.ac, c);
#else
        ret.ac = counter.ac;
        counter.ac += c;
#endif
        return ret.counters;
      }

    private:
      Atomic2 counter;
    };

  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_SYCLUtilities_interface_AtomicPairCounter_h
