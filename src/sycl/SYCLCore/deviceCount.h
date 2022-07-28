#ifndef HeterogenousCore_SYCLUtilities_deviceCount_h
#define HeterogenousCore_SYCLUtilities_deviceCount_h

//#include "SYCLCore/syclCheck.h"

#include <CL/sycl.hpp>

namespace cms {
  namespace sycltools {
    inline int deviceCount() {
      return (sycl::device::get_devices(sycl::info::device_type::all)).size();
    }
  }  // namespace sycltools
}  // namespace cms

#endif
