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
      sycl::atomic_ref<A, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(*i);
      return first.fetch_add(j);
    }

    template <typename T>
    inline T atomic_fetch_add(T* addr, T operand){
      auto atm = 
            cl::sycl::atomic_ref<T, cl::sycl::memory_order::relaxed, sycl::memory_scope::device, 
                                  cl::sycl::access::address_space::global_space > (addr[0]);
      return atm.fetch_add(operand);
    }

    template <typename T>
    inline T atomic_fetch_add_shared(T* addr, T operand){
      auto atm = 
            cl::sycl::atomic_ref<T, cl::sycl::memory_order::relaxed, sycl::memory_scope::device, 
                                  cl::sycl::access::address_space::local_space>(addr[0]);
      return atm.fetch_add(operand);
    }

    template <typename T>
    inline T atomic_fetch_sub(T* addr, T operand) {
          auto atm =
              cl::sycl::atomic_ref<T, cl::sycl::memory_order::relaxed, sycl::memory_scope::device,
                                   cl::sycl::access::address_space::global_space>(addr[0]);
      return atm.fetch_sub(operand);
    }

    template <typename T>
    inline T atomic_fetch_sub_shared(T *addr, T operand) {
          auto atm =
              cl::sycl::atomic_ref<T, cl::sycl::memory_order::relaxed, sycl::memory_scope::device,
                                   cl::sycl::access::address_space::local_space>(addr[0]);
      return atm.fetch_sub(operand);
    }

    template <typename A, typename B>
    inline A AtomicSub(A* i, B j){
      sycl::atomic_ref<A, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(*i);
      return first.fetch_add(-j);
    }

  template <typename T>
  inline T atomic_fetch_compare_inc(T *addr,T operand) {
  auto atm = cl::sycl::atomic_ref<T, cl::sycl::memory_order::relaxed, 
                                                cl::sycl::memory_scope::device,
                                   cl::sycl::access::address_space::global_space>(addr[0]);
  T old;
  while (true) {
    old = atm.load();
    if (old >= operand) {
      if (atm.compare_exchange_strong(old, 0))
        break;
    } else if (atm.compare_exchange_strong(old, old + 1))
      break;
  }
  return old;
}

  template <typename T>
    inline T atomic_fetch_compare_inc_shared(T *addr,T operand) {
    auto atm = cl::sycl::atomic_ref<T, cl::sycl::memory_order::relaxed, 
                                                  cl::sycl::memory_scope::device,
                                     cl::sycl::access::address_space::local_space>(addr[0]);
    T old;
    while (true) {
      old = atm.load();
      if (old >= operand) {
        if (atm.compare_exchange_strong(old, 0))
          break;
      } else if (atm.compare_exchange_strong(old, old + 1))
        break;
    }
    return old;
  }

    template <typename T>
    inline T atomic_fetch_min_shared(T *addr, T operand) {
    auto atm =
      cl::sycl::atomic_ref<T, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device,
                              cl::sycl::access::address_space::local_space>
                              (addr[0]);            
    return atm.fetch_min(operand);
    }

    template <typename T>
    inline T atomic_fetch_max_shared(T *addr, T operand) {
    auto atm =
      cl::sycl::atomic_ref<T, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device,
                              cl::sycl::access::address_space::local_space>
                              (addr[0]);            
    return atm.fetch_max(operand);
    }

    template <typename T>
    inline T atomic_fetch_min(T *addr, T operand) {
    auto atm =
      cl::sycl::atomic_ref<T, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device,
                              cl::sycl::access::address_space::global_space>
                              (addr[0]);            
    return atm.fetch_min(operand);
    }

    template <typename A, typename B>
    inline A AtomicMin(A* i, B j){
      sycl::ext::oneapi::atomic_ref<A, sycl::ext::oneapi::memory_order::relaxed,
                                sycl::ext::oneapi::memory_scope::work_group,
                                sycl::access::address_space::local_space> first(*i);
      auto test = first.fetch_min(j);
      return test;
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

 template <typename A, typename B>
    inline A AtomicInc2(A* i, B j, sycl::stream out){
      auto ret = *i;
      //out << "The value of i before is :" << *i << "\n";
      sycl::atomic_ref<A, sycl::memory_order::relaxed, sycl::memory_scope::work_group> first(*i);
      first.fetch_add(-j);
      //out << "The value of fetchadd is :" <<  first.fetch_add(-j) << "\n";
      //out << "The value of i is :" << *i << "\n";
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
