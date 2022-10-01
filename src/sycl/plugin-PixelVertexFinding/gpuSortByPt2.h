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

  __attribute__((always_inline)) void sortByPt2(ZVertices* pdata, WorkSpace* pws, sycl::nd_item<1> item) {
    auto& __restrict__ data = *pdata;
    auto& __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const* __restrict__ ptt2 = ws.ptt2;
    uint32_t const& nvFinal = data.nvFinal;

    int32_t const* __restrict__ iv = ws.iv;
    float* __restrict__ ptv2 = data.ptv2;
    uint16_t* __restrict__ sortInd = data.sortInd;

    // if (item.get_local_id(0) == 0)
    //    printf("sorting %d vertices\n",nvFinal);

    if (nvFinal < 1)
      return;

    // fill indexing
    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      data.idv[ws.itrk[i]] = iv[i];
    }

    // can be done asynchronoisly at the end of previous event
    for (auto i = item.get_local_id(0); i < nvFinal; i += item.get_local_range(0)) {
      ptv2[i] = 0;
    }
    item.barrier();

    for (auto i = item.get_local_id(0); i < nt; i += item.get_local_range(0)) {
      if (iv[i] > 9990)
        continue;
      cms::sycltools::atomic_fetch_add<float>(&ptv2[iv[i]], ptt2[i]);
    }
    item.barrier();

    if (1 == nvFinal) {
      if (item.get_local_id(0) == 0)
        sortInd[0] = 0;
      return;
    }
    auto swsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint16_t[1024]>(item.get_group());
    uint16_t* sws = (uint16_t*)swsbuff.get();
    radixSort<float, 2>(ptv2, sortInd, sws, nvFinal, item);
  }

  void sortByPt2Kernel(ZVertices* pdata, WorkSpace* pws, sycl::nd_item<1> item) { 
    sortByPt2(pdata, pws, item); }

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuSortByPt2_h
