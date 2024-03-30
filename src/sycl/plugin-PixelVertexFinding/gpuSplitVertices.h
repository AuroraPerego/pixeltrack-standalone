#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"
#include "SYCLCore/printf.h"

#include "gpuVertexFinder.h"

// #define VERTEX_DEBUG

namespace gpuVertexFinder {
  using sycl::fabs;

  inline void splitVertices(ZVertices* pdata,
                                                    WorkSpace* pws,
                                                    float maxChi2,
                                                    sycl::nd_item<1> item,
                                   uint32_t *it,
                                   float *zz,
                                   uint8_t *newV,
                                   float *ww,
                                   uint32_t *nq,
                                   float *znew,
                                   float *wnew,
                                   uint32_t *igv) {
    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ zt = ws.zt;
    float const* __restrict__ ezt2 = ws.ezt2;
    float* __restrict__ zv = data.zv;
    [[maybe_unused]] float* __restrict__ wv = data.wv;
    float const* __restrict__ chi2 = data.chi2;
    uint32_t& nvFinal = data.nvFinal;

    int32_t const* __restrict__ nn = data.ndof;
    int32_t* __restrict__ iv = ws.iv;

    assert(pdata);
    assert(zt);

    // one vertex per block
    for (auto kv = item.get_group(0); kv < nvFinal; kv += item.get_group_range(0)) {
      if (nn[kv] < 4)
        continue;
      if (chi2[kv] < maxChi2 * float(nn[kv]))
        continue;

      constexpr int MAXTK = 512;
      assert(nn[kv] < MAXTK);
      if (nn[kv] >= MAXTK)
        continue;  // too bad FIXME

      *nq = 0;                                 // number of track for this vertex
      sycl::group_barrier(item.get_group());

      // copy to local
      for (auto k = item.get_local_id(0); k < nt; k += item.get_local_range(0)) {
        if (iv[k] == int(kv)) {
          int old = cms::sycltools::atomic_fetch_compare_inc<uint32_t, sycl::access::address_space::local_space>(
              nq, (uint32_t)MAXTK);
          zz[old] = zt[k] - zv[kv];
          newV[old] = zz[old] < 0 ? 0 : 1;
          ww[old] = 1.f / ezt2[k];
          it[old] = k;
        }
      }

      sycl::group_barrier(item.get_group());
      assert(int(*nq) == nn[kv] + 1);

      int maxiter = 20;
      // kt-min....
      bool more = true;
      while ((sycl::group_barrier(item.get_group()), sycl::any_of_group(item.get_group(), more))) {
        more = false;
        if (0 == item.get_local_id(0)) {
          znew[0] = 0;
          znew[1] = 0;
          wnew[0] = 0;
          wnew[1] = 0;
        }
        sycl::group_barrier(item.get_group());
        for (auto k = item.get_local_id(0); k < (unsigned long)*nq; k += item.get_local_range(0)) {
          auto i = newV[k];
          cms::sycltools::atomic_fetch_add<float, sycl::access::address_space::local_space>(&znew[i], zz[k] * ww[k]);
          cms::sycltools::atomic_fetch_add<float, sycl::access::address_space::local_space>(&wnew[i], ww[k]);
        }
        sycl::group_barrier(item.get_group());
        if (0 == item.get_local_id(0)) {
          znew[0] /= wnew[0];
          znew[1] /= wnew[1];
        }
        sycl::group_barrier(item.get_group());
        for (auto k = item.get_local_id(0); k < (unsigned long)*nq; k += item.get_local_range(0)) {
          auto d0 = fabs(zz[k] - znew[0]);
          auto d1 = fabs(zz[k] - znew[1]);
          auto newer = d0 < d1 ? 0 : 1;
          more |= newer != newV[k];
          newV[k] = newer;
        }
        --maxiter;
        if (maxiter <= 0)
          more = false;
      }

      // avoid empty vertices
      if (0 == wnew[0] || 0 == wnew[1])
        continue;

      // quality cut
      auto dist2 = (znew[0] - znew[1]) * (znew[0] - znew[1]);

      auto chi2Dist = dist2 / (1.f / wnew[0] + 1.f / wnew[1]);

#ifdef VERTEX_DEBUG
      if (0 == item.get_local_id(0))
        printf("inter %d %f %f\n", 20 - maxiter, chi2Dist, dist2 * wv[kv]);
#endif

      if (chi2Dist < 4)
        continue;

      // get a new global vertex
      if (0 == item.get_local_id(0))
        *igv = cms::sycltools::atomic_fetch_add<uint32_t>(&ws.nvIntermediate, (uint32_t)1);
      sycl::group_barrier(item.get_group());
      for (auto k = item.get_local_id(0); k < (unsigned long)*nq; k += item.get_local_range(0)) {
        if (1 == newV[k])
          iv[it[k]] = *igv;
      }

    }  // loop on vertices
  }

  void splitVerticesKernel(ZVertices* pdata, WorkSpace* pws, float maxChi2, sycl::nd_item<1> item,
                                    uint32_t *it,
                                    float *zz,
                                    uint8_t *newV,
                                    float *ww,
                                    uint32_t *nq,
                                    float *znew,
                                    float *wnew,
                                    uint32_t *igv) {
    splitVertices(pdata, pws, maxChi2, item, it, zz, newV, ww, nq, znew, wnew, igv);
  }
}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
