#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {
  using sycl::fabs;

  __forceinline void splitVertices(ZVertices* pdata, 
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
                                   uint32_t *igv,
                                   sycl::stream out) {
    constexpr bool verbose = false;  // in principle the compiler should optmize out if false

    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ zt = ws.zt;
    float const* __restrict__ ezt2 = ws.ezt2;
    float* __restrict__ zv = data.zv;
    float* __restrict__ wv = data.wv;
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
        continue;                      // too bad FIXME
      //uint32_t it[MAXTK];   // track index
      //float zz[MAXTK];      // z pos
      //uint8_t newV[MAXTK];  // 0 or 1
      //float ww[MAXTK];      // z weight

      //uint32_t nq;  // number of track for this vertex
      *nq = 0;
      item.barrier();

      // copy to local
      for (auto k = item.get_local_id(0); k < nt; k += item.get_local_range(0)) {
        if (iv[k] == int(kv)) {
          int old = cms::sycltools::AtomicInc(nq, MAXTK);
          zz[old] = zt[k] - zv[kv];
          newV[old] = zz[old] < 0 ? 0 : 1;
          ww[old] = 1.f / ezt2[k];
          it[old] = k;
        }
      }

      //float znew[2], wnew[2];  // the new vertices

      item.barrier();
      assert(int(*nq) == nn[kv] + 1);

      int maxiter = 20;
      // kt-min....
      bool more = true;
      while ((item.barrier(), sycl::any_of_group(item.get_group(), more))) {
        more = false;
        if (0 == item.get_local_id(0)) {
          znew[0] = 0;
          znew[1] = 0;
          wnew[0] = 0;
          wnew[1] = 0;
        }
        item.barrier();
        for (auto k = item.get_local_id(0); k < (unsigned long)*nq; k += item.get_local_range(0)) {
          auto i = newV[k];
          cms::sycltools::AtomicAdd(&znew[i], zz[k] * ww[k]);
          cms::sycltools::AtomicAdd(&wnew[i], ww[k]);
        }
        item.barrier();
        if (0 == item.get_local_id(0)) {
          znew[0] /= wnew[0];
          znew[1] /= wnew[1];
        }
        item.barrier();
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

      if (verbose && 0 == item.get_local_id(0))
        out << "inter " << 20 - maxiter << " " << chi2Dist << " " << dist2 * wv[kv] << " " << "\n";

      if (chi2Dist < 4)
        continue;

      // get a new global vertex
      //__shared__ uint32_t igv;
      if (0 == item.get_local_id(0))
        *igv = cms::sycltools::AtomicAdd(&ws.nvIntermediate, 1);
      item.barrier();
      for (auto k = item.get_local_id(0); k < (unsigned long)*nq; k += item.get_local_range(0)) {
        if (1 == newV[k])
          iv[it[k]] = *igv;
      }

    }  // loop on vertices
  }

  void splitVerticesKernel(ZVertices* pdata, 
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
                           uint32_t *igv,
                           const sycl::stream out
  ) {
    splitVertices(pdata, pws, maxChi2, item, it, zz, newV, ww, nq, znew, wnew, igv, out);
  }
}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSplitVertices_h
