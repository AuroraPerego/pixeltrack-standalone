#ifndef AlpakaCore_getDeviceIndex_h
#define AlpakaCore_getDeviceIndex_h

#include <alpaka/alpaka.hpp>
#include "AlpakaCore/alpakaConfig.h"

namespace cms::alpakatools {

  inline std::optional<ALPAKA_ACCELERATOR_NAMESPACE::Platform> platform;

  // generic interface, for DevOacc and DevOmp5
  template <typename Device>
  inline int getDeviceIndex(Device const& device) {
    return device.iDevice();
  }

  // overload for DevCpu
  inline int getDeviceIndex(alpaka::DevCpu const& device) { return 0; }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  // overload for DevCudaRt
  inline int getDeviceIndex(alpaka::DevCudaRt const& device) { return alpaka::getNativeHandle(device); }
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  // overload for DevHipRt
  inline int getDeviceIndex(alpaka::DevHipRt const& device) { return alpaka::getNativeHandle(device); }
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#ifdef ALPAKA_ACC_SYCL_ENABLED
  // overload for DevGenericSycl
  inline int getDeviceIndex(alpaka::DevCpuSycl const& device) { return 0; }  //std::find(platform.syclDevices().begin(), platform.syclDevices().end(), device) - platform.syclDevices().begin(); }  // FIXME_
  inline int getDeviceIndex(alpaka::DevGpuSyclIntel const& device) { return 0; } //std::find(platform.syclDevices().begin(), platform.syclDevices().end(), device) - platform.syclDevices().begin(); }  // FIXME_
#endif  // ALPAKA_ACC_SYCL_ENABLED

}  // namespace cms::alpakatools

#endif  // AlpakaCore_getDeviceIndex_h
