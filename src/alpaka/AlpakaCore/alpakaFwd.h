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

  // Platforms
  class PltfCpu;
  template <typename TApi>
  class PltfUniformCudaHipRt;
  using PltfCudaRt = PltfUniformCudaHipRt<ApiCudaRt>;
  using PltfHipRt = PltfUniformCudaHipRt<ApiHipRt>;
  class PltfCpuSyclIntel;
  class PltfGpuSyclIntel;

  // Devices
  class DevCpu;
  template <typename TApi>
  class DevUniformCudaHipRt;
  using DevCudaRt = DevUniformCudaHipRt<ApiCudaRt>;
  using DevHipRt = DevUniformCudaHipRt<ApiHipRt>;
  template <typename TPltf>
  class DevGenericSycl;
  using DevCpuSyclIntel = DevGenericSycl<PltfCpuSyclIntel>;
  using DevGpuSyclIntel = DevGenericSycl<PltfGpuSyclIntel>;

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

  template <typename TDev, bool TBlocking>
  class QueueGenericSyclBase;
  using QueueCpuSyclIntelBlocking = QueueGenericSyclBase<DevCpuSyclIntel, true>;
  using QueueCpuSyclIntelNonBlocking = QueueGenericSyclBase<DevCpuSyclIntel, false>;
  using QueueGpuSyclIntelBlocking = QueueGenericSyclBase<DevGpuSyclIntel, true>;
  using QueueGpuSyclIntelNonBlocking = QueueGenericSyclBase<DevGpuSyclIntel, false>;


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
  using EventCpuSyclIntel = EventGenericSycl<DevCpuSyclIntel>;
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
  class AccCpuSyclIntel;

  template <typename TDim, typename TIdx>
  class AccGpuSyclIntel;

}  // namespace alpaka

#endif  // AlpakaCore_alpakaFwd_h
