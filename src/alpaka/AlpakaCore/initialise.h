#ifndef AlpakaCore_initialise_h
#define AlpakaCore_initialise_h

#include "AlpakaCore/alpakaConfig.h"

namespace cms::alpakatools {

  template <typename TPlatform>
  void initialise();

  template <typename TPlatform>
  void resetDevices();

  // explicit template instantiation declaration
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT
  extern template void initialise<alpaka_serial_sync::Platform>();
  extern template void resetDevices<alpaka_serial_sync::Platform>();
#endif
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_PRESENT
  extern template void initialise<alpaka_tbb_async::Platform>();
  extern template void resetDevices<alpaka_tbb_async::Platform>();
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_PRESENT
  extern template void initialise<alpaka_cuda_async::Platform>();
  extern template void resetDevices<alpaka_cuda_async::Platform>();
#endif
#ifdef ALPAKA_ACC_GPU_HIP_PRESENT
  extern template void initialise<alpaka_rocm_async::Platform>();
  extern template void resetDevices<alpaka_rocm_async::Platform>();
#endif
#ifdef ALPAKA_ACC_SYCL_PRESENT
  extern template void initialise<alpaka_cpu_sycl::Platform>();
  extern template void initialise<alpaka_gpu_sycl::Platform>();
  extern template void resetDevices<alpaka_cpu_sycl::Platform>();
  extern template void resetDevices<alpaka_gpu_sycl::Platform>();
#endif

}  // namespace cms::alpakatools

#endif  // AlpakaCore_initialise_h
