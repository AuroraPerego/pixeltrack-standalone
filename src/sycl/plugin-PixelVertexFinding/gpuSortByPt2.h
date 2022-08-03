#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h

//#include <algorithm>
#include <cmath>
#include <cstdint>

#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"

#include "SYCLCore/radixSort.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  __forceinline void sortByPt2(ZVertices* pdata, WorkSpace* pws, sycl::nd_item<3> item, uint16_t* sws, 
                               int32_t *c, int32_t *ct, int32_t *cu, int *ibs, int *p, uint32_t *firstNeg) {
    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ ptt2 = ws.ptt2;
    uint32_t const& nvFinal = data.nvFinal;

    int32_t const* __restrict__ iv = ws.iv;
    float* __restrict__ ptv2 = data.ptv2;
    uint16_t* __restrict__ sortInd = data.sortInd;

    // if (item.get_local_id(2) == 0)
    //    printf("sorting %d vertices\n",nvFinal);

    if (nvFinal < 1)
      return;

    // fill indexing
    for (auto i = item.get_local_id(2); i < nt; i += item.get_local_range(2)) {
      data.idv[ws.itrk[i]] = iv[i];
    }

    // can be done asynchronoisly at the end of previous event
    for (auto i = item.get_local_id(2); i < nvFinal; i += item.get_local_range(2)) {
      ptv2[i] = 0;
    }
    item.barrier();

    for (auto i = item.get_local_id(2); i < nt; i += item.get_local_range(2)) {
      if (iv[i] > 9990)
        continue;
      cms::sycltools::AtomicAdd(&ptv2[iv[i]], ptt2[i]);
    }
    item.barrier();

    if (1 == nvFinal) {
      if (item.get_local_id(2) == 0)
        sortInd[0] = 0;
      return;
    }
    radixSort<float, 2>(ptv2, sortInd, sws, nvFinal, item, c, ct, cu, ibs, p, firstNeg);
  }

  void sortByPt2Kernel(ZVertices* pdata, WorkSpace* pws, sycl::nd_item<3> item, uint16_t* sws, int32_t *c, int32_t *ct, int32_t *cu, int *ibs, int *p, uint32_t *firstNeg) { 
    sortByPt2(pdata, pws, item, sws, c, ct, cu, ibs, p, firstNeg); }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
