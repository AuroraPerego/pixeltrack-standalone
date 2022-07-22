#ifndef HeterogenousCore_SYCLUtilities_currentDevice_h
#define HeterogenousCore_SYCLUtilities_currentDevice_h

//#include "SYCLCore/syclCheck.h"

#include <CL/sycl.hpp>

namespace cms {
  namespace sycltools {
    inline DeviceSelector currentDevice(sycl::queue stream) {
      DeviceSelector dev;
      dev = stream.get_device();
      //sycl::device::device(&dev);
      return dev;
    }
  }  // namespace sycltools
}  // namespace cms

#endif
