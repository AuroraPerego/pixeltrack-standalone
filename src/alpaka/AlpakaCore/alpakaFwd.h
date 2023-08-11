#ifndef AlpakaCore_alpakaFwd_h
#define AlpakaCore_alpakaFwd_h

#include <cstddef>
#include <cstdint>
#include <type_traits>

/**
 * This file forward declares specific types defined in Alpaka
 * (depending on the backend-enabling macros) so that these types
 * would be available throughout CMSSW without a direct dependence on
 * Alpaka in order to avoid the constraints that would impose
 * (primarily the device compiler)
 *
 * This is a little bit brittle, but let's see how it goes.
 */
namespace alpaka {

  // miscellanea
  template <std::size_t N>
  using DimInt = std::integral_constant<std::size_t, N>;

  template <typename TDim, typename TVal>
  class Vec;

  template <typename TDim, typename TIdx>
  class WorkDivMembers;

  // API
  struct ApiCudaRt;
  struct ApiHipRt;

  namespace detail {
    struct SyclCpuSelector;
    struct IntelGpuSelector;
  }  // namespace detail

  // Platforms
  class PlatformCpu;
  template <typename TApi>
  class PlatformUniformCudaHipRt;
  using PlatformCudaRt = PlatformUniformCudaHipRt<ApiCudaRt>;
  using PlatformHipRt = PlatformUniformCudaHipRt<ApiHipRt>;
  template <typename TSelector>
  class PlatformGenericSycl;
  using PlatformCpuSycl = PlatformGenericSycl<detail::SyclCpuSelector>;
  using PlatformGpuSyclIntel = PlatformGenericSycl<detail::IntelGpuSelector>;

  // Devices
  class DevCpu;
  template <typename TApi>
  class DevUniformCudaHipRt;
  using DevCudaRt = DevUniformCudaHipRt<ApiCudaRt>;
  using DevHipRt = DevUniformCudaHipRt<ApiHipRt>;
  template <typename TPlatform>
  class DevGenericSycl;
  using DevCpuSycl = DevGenericSycl<PlatformCpuSycl>;
  using DevGpuSyclIntel = DevGenericSycl<PlatformGpuSyclIntel>;

  // Queues
  template <typename TDev>
  class QueueGenericThreadsBlocking;
  using QueueCpuBlocking = QueueGenericThreadsBlocking<DevCpu>;

  template <typename TDev>
  class QueueGenericThreadsNonBlocking;
  using QueueCpuNonBlocking = QueueGenericThreadsNonBlocking<DevCpu>;

  namespace uniform_cuda_hip::detail {
    template <typename TApi, bool TBlocking>
    class QueueUniformCudaHipRt;
  }
  using QueueCudaRtBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<ApiCudaRt, true>;
  using QueueCudaRtNonBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<ApiCudaRt, false>;
  using QueueHipRtBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<ApiHipRt, true>;
  using QueueHipRtNonBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<ApiHipRt, false>;

  namespace detail {
    template <typename TDev, bool TBlocking>
    class QueueGenericSyclBase;
  }

  template <typename TDev>
  using QueueGenericSyclBlocking = detail::QueueGenericSyclBase<TDev, true>;
  template <typename TDev>
  using QueueGenericSyclNonBlocking = detail::QueueGenericSyclBase<TDev, false>;
  using QueueCpuSyclBlocking = QueueGenericSyclBlocking<DevCpuSycl>;
  using QueueCpuSyclNonBlocking = QueueGenericSyclNonBlocking<DevCpuSycl>;
  using QueueGpuSyclIntelBlocking = QueueGenericSyclBlocking<DevGpuSyclIntel>;
  using QueueGpuSyclIntelNonBlocking = QueueGenericSyclNonBlocking<DevGpuSyclIntel>;

  // Events
  template <typename TDev>
  class EventGenericThreads;
  using EventCpu = EventGenericThreads<DevCpu>;

  template <typename TApi>
  class EventUniformCudaHipRt;
  using EventCudaRt = EventUniformCudaHipRt<ApiCudaRt>;
  using EventHipRt = EventUniformCudaHipRt<ApiHipRt>;

  template <typename TDev>
  class EventGenericSycl;
  using EventCpuSycl = EventGenericSycl<DevCpuSycl>;
  using EventGpuSyclIntel = EventGenericSycl<DevGpuSyclIntel>;

  // Accelerators
  template <typename TApi, typename TDim, typename TIdx>
  class AccGpuUniformCudaHipRt;

  template <typename TDim, typename TIdx>
  using AccGpuCudaRt = AccGpuUniformCudaHipRt<ApiCudaRt, TDim, TIdx>;

  template <typename TDim, typename TIdx>
  using AccGpuHipRt = AccGpuUniformCudaHipRt<ApiHipRt, TDim, TIdx>;

  template <typename TDim, typename TIdx>
  class AccCpuSerial;

  template <typename TDim, typename TIdx>
  class AccCpuTbbBlocks;

  template <typename TDim, typename TIdx>
  class AccCpuOmp2Blocks;

  template <typename TDim, typename TIdx>
  class AccCpuSycl;

  template <typename TDim, typename TIdx>
  class AccGpuSyclIntel;

}  // namespace alpaka

#endif  // AlpakaCore_alpakaFwd_h
