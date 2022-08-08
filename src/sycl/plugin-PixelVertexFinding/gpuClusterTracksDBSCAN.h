#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksDBSCAN_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksDBSCAN_h

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
  void clusterTracksDBSCAN(ZVertices* pdata,
                           WorkSpace* pws,
                           int minT,      // min number of neighbours to be "core"
                           float eps,     // max absolute distance to cluster
                           float errmax,  // max error to be "seed"
                           float chi2max,  // max normalized distance to cluster
                           sycl::nd_item<1> item,
                           Hist *hist,
                           Hist::Counter* hws,
                           unsigned int* foundClusters,
                           const sycl::stream out
  ) {
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
        //        if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return;
        nn[i]++;
      };

      cms::sycltools::forEachInBins(*hist, izt[i], 1, loop);
    }

    item.barrier();

    // find NN with smaller z...
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (nn[i] < minT)
        continue;  // DBSCAN core rule
      float mz = zt[i];
      auto loop = [&](uint32_t j) {
        if (zt[j] >= mz)
          return;
        if (nn[j] < minT)
          return;  // DBSCAN core rule
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > eps)
          return;
        //        if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return;
        mz = zt[j];
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

    item.barrier();

#ifdef GPU_DEBUG
    //  mini verification
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] != int(i))
        assert(iv[iv[i]] != int(i));
    }
    item.barrier();
#endif

#ifdef GPU_DEBUG
    // and verify that we did not spit any cluster...
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (nn[i] < minT)
        continue;  // DBSCAN core rule
      assert(zt[iv[i]] <= zt[i]);
      auto loop = [&](uint32_t j) {
        if (nn[j] < minT)
          return;  // DBSCAN core rule
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > eps)
          return;
        //  if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return;
        // they should belong to the same cluster, isn't it?
        if (iv[i] != iv[j]) {
          out << "ERROR " << i << " " << iv[i] << " " << zt[i] << " " << zt[iv[i]] << " " << iv[iv[i]] << "\n";
          out << "      " << j << " " << iv[j] << " " << zt[j] << " " << zt[iv[j]] << " " << iv[iv[j]] << "\n";
          ;
        }
        assert(iv[i] == iv[j]);
      };
      cms::sycltools::forEachInBins(*hist, izt[i], 1, loop);
    }
    item.barrier();
#endif

    // collect edges (assign to closest cluster of closest point??? here to closest point)
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      //    if (nn[i]==0 || nn[i]>=minT) continue;    // DBSCAN edge rule
      if (nn[i] >= minT)
        continue;  // DBSCAN edge rule
      float mdist = eps;
      auto loop = [&](uint32_t j) {
        if (nn[j] < minT)
          return;  // DBSCAN core rule
        auto dist = std::abs(zt[i] - zt[j]);
        if (dist > mdist)
          return;
        if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
          return;  // needed?
        mdist = dist;
        iv[i] = iv[j];  // assign to cluster (better be unique??)
      };
      cms::sycltools::forEachInBins(*hist, izt[i], 1, loop);
    }

    *foundClusters = 0;
    item.barrier();

    // find the number of different clusters, identified by a tracks with clus[i] == i;
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

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuClusterTracksDBSCAN_h
