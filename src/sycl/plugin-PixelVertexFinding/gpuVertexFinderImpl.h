#include "SYCLCore/syclAtomic.h"

#include "gpuClusterTracksIterative.h"

namespace gpuVertexFinder {

  using Hist = cms::sycltools::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;

  ZVertexHeterogeneous Producer::makeAsync(sycl::queue stream, TkSoA const* tksoa, float ptMin, bool isCpu) const {

    ZVertexHeterogeneous vertices(cms::sycltools::make_device_unique<ZVertexSoA>(stream));
    assert(tksoa);
    auto* soa = vertices.get();
    assert(soa);

    auto ws_d = cms::sycltools::make_device_unique<WorkSpace>(stream);
        auto numberOfBlocks = 1;
        auto blockSize = 1024 - 256;
        stream.submit([&](sycl::handler& cgh) {
          auto soa_kernel = soa;
          auto ws_kernel = ws_d.get();
          auto minT_kernel = minT;
          auto eps_kernel = eps;
          auto errmax_kernel = errmax;
          auto chi2max_kernel = chi2max;
          cgh.parallel_for(sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                           [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                             clusterTracksIterative(
                                 soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item);
                           });
        });
        return vertices;
  }

}  // namespace gpuVertexFinder

#undef FROM
