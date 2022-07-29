#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"


#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  __forceinline void fitVertices(ZVertices* pdata,
                                 WorkSpace* pws,
                                 float chi2Max,  // for outlier rejection
                                 sycl::nd_item<3> item,
                                 int* noise,
                                 sycl::stream out
  ) {
    constexpr bool verbose = false;  // in principle the compiler should optmize out if false

    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ zt = ws.zt;
    float const* __restrict__ ezt2 = ws.ezt2;
    float* __restrict__ zv = data.zv;
    float* __restrict__ wv = data.wv;
    float* __restrict__ chi2 = data.chi2;
    uint32_t& nvFinal = data.nvFinal;
    uint32_t& nvIntermediate = ws.nvIntermediate;

    int32_t* __restrict__ nn = data.ndof;
    int32_t* __restrict__ iv = ws.iv;

    assert(pdata);
    assert(zt);

    assert(nvFinal <= nvIntermediate);
    nvFinal = nvIntermediate;
    auto foundClusters = nvFinal;

    // zero
    for (auto i = item.get_local_id(2); i < foundClusters; i += item.get_local_range(2)) {
      zv[i] = 0;
      wv[i] = 0;
      chi2[i] = 0;
    }

    // only for test
    if (verbose && 0 == item.get_local_id(2))
      *noise = 0;

    item.barrier();

    // compute cluster location
    for (auto i = item.get_local_id(2); i < nt; i += item.get_local_range(2)) {
      if (iv[i] > 9990) {
        if (verbose)
          cms::sycltools::AtomicAdd(noise, 1);
        continue;
      }
      assert(iv[i] >= 0);
      assert(iv[i] < int(foundClusters));
      auto w = 1.f / ezt2[i];
      cms::sycltools::AtomicAdd(&zv[iv[i]], zt[i] * w);
      cms::sycltools::AtomicAdd(&wv[iv[i]], w);
    }

    item.barrier();
    // reuse nn
    for (auto i = item.get_local_id(2); i < foundClusters; i += item.get_local_range(2)) {
      assert(wv[i] > 0.f);
      zv[i] /= wv[i];
      nn[i] = -1;  // ndof
    }
    item.barrier();

    // compute chi2
    for (auto i = item.get_local_id(2); i < nt; i += item.get_local_range(2)) {
      if (iv[i] > 9990)
        continue;

      auto c2 = zv[iv[i]] - zt[i];
      c2 *= c2 / ezt2[i];
      if (c2 > chi2Max) {
        iv[i] = 9999;
        continue;
      }
      cms::sycltools::AtomicAdd(&chi2[iv[i]], c2);
      cms::sycltools::AtomicAdd(&nn[iv[i]], 1);
    }
    item.barrier();
    for (auto i = item.get_local_id(2); i < foundClusters; i += item.get_local_range(2))
      if (nn[i] > 0)
        wv[i] *= float(nn[i]) / chi2[i];

    if (verbose && 0 == item.get_local_id(2))
      out << "found " << foundClusters << " proto clusters ";
    if (verbose && 0 == item.get_local_id(2))
      out << "and " << *noise << " noise\n";
  }

  void fitVerticesKernel(ZVertices* pdata,
                         WorkSpace* pws,
                         float chi2Max,  // for outlier rejection
                         sycl::nd_item<3> item,
                         int* noise,
                         sycl::stream out
  ) {
    fitVertices(pdata, pws, chi2Max, item, noise, out);
  }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuFitVertices_h
