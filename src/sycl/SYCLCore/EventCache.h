//TODO

#ifndef HeterogeneousCore_SYCLUtilities_EventCache_h
#define HeterogeneousCore_SYCLUtilities_EventCache_h

#include <vector>

#include <CL/sycl.hpp>

#include "Framework/ReusableObjectHolder.h"
#include "SYCLCore/SharedEventPtr.h"

class SYCLService;

namespace cms {
  namespace sycltools {
    class EventCache {
    public:
      using BareEvent = SharedEventPtr::element_type;

      EventCache();

      // Gets a (cached) CUDA event for the current device. The event
      // will be returned to the cache by the shared_ptr destructor. The
      // returned event is guaranteed to be in the state where all
      // captured work has completed, i.e. cudaEventQuery() == cudaSuccess.
      //
      // This function is thread safe
      SharedEventPtr get(sycl::device device);

    private:
      friend class ::SYCLService;

      // thread safe
      SharedEventPtr makeOrGet(int dev);

      // not thread safe, intended to be called only from CUDAService destructor
      void clear();

      class Deleter {
      public:
        Deleter() = default;
        void operator()(sycl::event* event) const;
      };

      std::vector<edm::ReusableObjectHolder<BareEvent, Deleter>> cache_;
    };

    // Gets the global instance of a EventCache
    // This function is thread safe
    EventCache& getEventCache();
  }  // namespace sycltools
}  // namespace cms

#endif
