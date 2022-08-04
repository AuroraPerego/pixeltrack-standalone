#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"


#include "gpuVertexFinder.h"

namespace gpuVertexFinder {
  
  using Hist = cms::sycltools::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;

  // this algo does not really scale as it works in a single block...
  // enough for <10K tracks we have
  //
  // based on Rodrighez&Laio algo
  //
  __forceinline void clusterTracksByDensity(gpuVertexFinder::ZVertices* pdata,
                                            gpuVertexFinder::WorkSpace* pws,
                                            int minT,      // min number of neighbours to be "seed"
                                            float eps,     // max absolute distance to cluster
                                            float errmax,  // max error to be "seed"
                                            float chi2max,  // max normalized distance to cluster
                                            sycl::nd_item<1> item,
                                            Hist *hist,
                                            Hist::Counter* hws,
                                            unsigned int* foundClusters,
                                            const sycl::stream out
  ) {
    using namespace gpuVertexFinder;
    constexpr bool verbose = false;  // in principle the compiler should optmize out if false

    if (verbose && 0 == item.get_local_id(0))
      out << "params" << minT << " " << eps << " " << errmax << " " << chi2max << "\n";

    auto er2mx = errmax * errmax;

    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ zt = ws.zt;
    float const* __restrict__ ezt2 = ws.ezt2;

    uint32_t& nvFinal = data.nvFinal;
    uint32_t& nvIntermediate = ws.nvIntermediate;

    uint8_t* __restrict__ izt = ws.izt;
    int32_t* __restrict__ nn = data.ndof;
    int32_t* __restrict__ iv = ws.iv;

    assert(pdata);
    assert(zt);

    //__shared__ typename Hist::Counter hws[32];
    for (auto j = item.get_local_id(0); j < Hist::totbins(); j += item.get_local_range(0)) {
      hist->off[j] = 0;
    }
    item.barrier();

    if (verbose && 0 == item.get_local_id(0))
      out << "booked hist with " << hist->nbins() << " bins, size " << hist->capacity() << " for " << nt << " tracks\n";

    assert(nt <= hist->capacity());

    // fill hist  (bin shall be wider than "eps")
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      assert(i < ZVertices::MAXTRACKS);
      int iz = int(zt[i] * 10.);  // valid if eps<=0.1
      // iz = std::clamp(iz, INT8_MIN, INT8_MAX);  // sorry c++17 only
      iz = std::min(std::max(iz, INT8_MIN), INT8_MAX);
      izt[i] = iz - INT8_MIN;
      assert(iz - INT8_MIN >= 0);
      assert(iz - INT8_MIN < 256);
      hist->count(izt[i]);
      iv[i] = i;
      nn[i] = 0;
    }
    item.barrier();
    if (item.get_local_id(0) < 32)
      hws[item.get_local_id(0)] = 0;  // used by prefix scan...
    item.barrier();
    hist->finalize(item, hws);
    item.barrier();
    assert(hist->size() == nt);
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      hist->fill(izt[i], uint16_t(i));
    }
    item.barrier();

    // count neighbours
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (ezt2[i] > er2mx)
        continue;
      auto loop = [&](uint32_t j) {
        if (i == j)
          return;
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > eps)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;
        nn[i]++;
      };

      cms::sycltools::forEachInBins(*hist, izt[i], 1, loop);
    }

    item.barrier();

    // find closest above me .... (we ignore the possibility of two j at same distance from i)
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      float mdist = eps;
      auto loop = [&](uint32_t j) {
        if (nn[j] < nn[i])
          return;
        if (nn[j] == nn[i] && zt[j] >= zt[i])
          return;  // if equal use natural order...
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > mdist)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;  // (break natural order???)
        mdist = dist;
        iv[i] = j;  // assign to cluster (better be unique??)
      };
      cms::sycltools::forEachInBins(*hist, izt[i], 1, loop);
    }

    item.barrier();

#ifdef GPU_DEBUG
    //  mini verification
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] != int(i))
        assert(iv[iv[i]] != int(i));
    }
    item.barrier();
#endif

    // consolidate graph (percolate index of seed)
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      auto m = iv[i];
      while (m != iv[m])
        m = iv[m];
      iv[i] = m;
    }

#ifdef GPU_DEBUG
    item.barrier();
    //  mini verification
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] != int(i))
        assert(iv[iv[i]] != int(i));
    }
#endif

#ifdef GPU_DEBUG
    // and verify that we did not spit any cluster...
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      auto minJ = i;
      auto mdist = eps;
      auto loop = [&](uint32_t j) {
        if (nn[j] < nn[i])
          return;
        if (nn[j] == nn[i] && zt[j] >= zt[i])
          return;  // if equal use natural order...
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > mdist)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;
        mdist = dist;
        minJ = j;
      };
      cms::sycltools::forEachInBins(*hist, izt[i], 1, loop);
      // should belong to the same cluster...
      assert(iv[i] == iv[minJ]);
      assert(nn[i] <= nn[iv[i]]);
    }
    item.barrier();
#endif

    *foundClusters = 0;
    item.barrier();

    // find the number of different clusters, identified by a tracks with clus[i] == i and density larger than threshold;
    // mark these tracks with a negative id.
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] == int(i)) {
        if (nn[i] >= minT) {
          auto old = cms::sycltools::AtomicInc(foundClusters, 0xffffffff);
          iv[i] = -(old + 1);
        } else {  // noise
          iv[i] = -9998;
        }
      }
    }
    item.barrier();

    assert(*foundClusters < ZVertices::MAXVTX);

    // propagate the negative id to all the tracks in the cluster.
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] >= 0) {
        // mark each track in a cluster with the same id as the first one
        iv[i] = iv[iv[i]];
      }
    }
    item.barrier();

    // adjust the cluster id to be a positive value starting from 0
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      iv[i] = -iv[i] - 1;
    }

    nvIntermediate = nvFinal = *foundClusters;

    if (verbose && 0 == item.get_local_id(0))
      out << "found " << *foundClusters << " proto vertices\n";
  }

  void clusterTracksByDensityKernel(gpuVertexFinder::ZVertices* pdata,
                                    gpuVertexFinder::WorkSpace* pws,
                                    int minT,      // min number of neighbours to be "seed"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max,  // max normalized distance to cluster
                                    sycl::nd_item<1> item,
                                    cms::sycltools::HistoContainer<uint8_t, 256, 16000, 8, uint16_t> *hist,
                                    Hist::Counter* hws,
                                    unsigned int *foundClusters,
                                    const sycl::stream out
  ) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max, item, hist, hws, foundClusters, out);
  }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksByDensity_h
