#ifndef HeterogeneousCore_SYCLUtilities_SharedEventPtr_h
#define HeterogeneousCore_SYCLUtilities_SharedEventPtr_h

#include <memory>
#include <type_traits>

#include <CL/sycl.hpp>

namespace cms {
  namespace sycltools {
    class Event{
      sycl::event m_event; 
    };
    // cudaEvent_t itself is a typedef for a pointer, for the use with
    // edm::ReusableObjectHolder the pointed-to type is more interesting
    // to avoid extra layer of indirection
    using SharedEventPtr = std::shared_ptr<std::remove_pointer_t<sycl::event>>;
  }  // namespace sycltools
}  // namespace cms

#endif
