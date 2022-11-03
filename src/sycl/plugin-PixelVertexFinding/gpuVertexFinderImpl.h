#include "SYCLCore/syclAtomic.h"

#include "gpuClusterTracksByDensity.h"
#include "gpuClusterTracksDBSCAN.h"
#include "gpuClusterTracksIterative.h"
#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

// #define VERTEX_DEBUG
// #define GPU_DEBUG

namespace gpuVertexFinder {
  
  using Hist = cms::sycltools::HistoContainer<uint8_t, 256, 16000, 8, uint16_t>;

  void loadTracks(TkSoA const* ptracks, ZVertexSoA* soa, WorkSpace* pws, float ptMin, sycl::nd_item<1> item) {
    assert(ptracks);
    assert(soa);
    auto const& tracks = *ptracks;
    auto const& fit = tracks.stateAtBS;
    auto const* quality = tracks.qualityData();

    auto first = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    for (int idx = first, nt = TkSoA::stride(); idx < nt; idx += item.get_group_range(0) * item.get_local_range(0)) {
      auto nHits = tracks.nHits(idx);
      if (nHits == 0)
        break;  // this is a guard: maybe we need to move to nTracks...

      // initialize soa...
      soa->idv[idx] = -1;

      if (nHits < 4)
        continue;  // no triplets
      if (quality[idx] != trackQuality::loose)
        continue;

      auto pt = tracks.pt(idx);

      if (pt < ptMin)
        continue;

      auto& data = *pws;
      auto it = cms::sycltools::atomic_fetch_add<uint32_t>(&data.ntrks, (uint32_t)1);
      data.itrk[it] = idx;
      data.zt[it] = tracks.zip(idx);
      data.ezt2[it] = fit.covariance(idx)(14);
      data.ptt2[it] = pt * pt;
    }
  }

// #define THREE_KERNELS
#ifndef THREE_KERNELS
  void vertexFinderOneKernel(gpuVertexFinder::ZVertices* pdata,
                             gpuVertexFinder::WorkSpace* pws,
                             int minT,      // min number of neighbours to be "seed"
                             float eps,     // max absolute distance to cluster
                             float errmax,  // max error to be "seed"
                             float chi2max,  // max normalized distance to cluster
                             sycl::nd_item<1> item, 
                             int32_t *c_acc, 
                             int32_t *ct_acc, 
                             int32_t *cu_acc, 
                             int *ibs_acc, 
                             int *p_acc,
                             uint32_t *firstNeg_acc
  ) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max, item); 
    item.barrier();
    fitVertices(pdata, pws, 50., item);
    item.barrier();
    splitVertices(pdata, pws, 9.f, item);
    item.barrier();
    fitVertices(pdata, pws, 5000., item);
    item.barrier();
    sortByPt2(pdata, pws, item, c_acc, ct_acc, cu_acc, ibs_acc, p_acc, firstNeg_acc);
  }
#else
  void vertexFinderKernel1(gpuVertexFinder::ZVertices* pdata,
                           gpuVertexFinder::WorkSpace* pws,
                           int minT,      // min number of neighbours to be "seed"
                           float eps,     // max absolute distance to cluster
                           float errmax,  // max error to be "seed"
                           float chi2max  // max normalized distance to cluster,
                           sycl::nd_item<1> item) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max, item);
    item.barrier();
    fitVertices(pdata, pws, 50., item);
  }

  void vertexFinderKernel2(gpuVertexFinder::ZVertices* pdata, gpuVertexFinder::WorkSpace* pws,
                           sycl::nd_item<1> item, 
                             int32_t *c_acc, 
                             int32_t *ct_acc, 
                             int32_t *cu_acc, 
                             int *ibs_acc, 
                             int *p_acc,
                             uint32_t *firstNeg_acc
  ) {
    fitVertices(pdata, pws, 5000., item);
    item.barrier();
    sortByPt2(pdata, pws, item, c_acc, ct_acc, cu_acc, ibs_acc, p_acc, firstNeg_acc);
  }
#endif

ZVertexHeterogeneous Producer::makeAsync(sycl::queue stream, TkSoA const* tksoa, float ptMin) const {
    #ifdef VERTEX_DEBUG
        std::cout << "producing Vertices on GPU" << std::endl;
    #endif
        
    ZVertexHeterogeneous vertices(cms::sycltools::make_device_unique<ZVertexSoA>(stream));
    assert(tksoa);
    auto* soa = vertices.get();
    assert(soa);

    auto ws_d = cms::sycltools::make_device_unique<WorkSpace>(stream);

    stream.submit([&](sycl::handler &cgh) {
      auto soa_kernel = soa;
      auto ws_kernel   = ws_d.get();
      cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
          [=](sycl::nd_item<1> item){ 
                init(soa_kernel, ws_kernel);
      });
    });

    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif

    auto blockSize = 128;
    auto numberOfBlocks = (TkSoA::stride() + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      auto tksoa_kernel = tksoa;
      auto soa_kernel   = soa;
      auto ws_kernel    = ws_d.get();
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
              loadTracks(tksoa_kernel, soa_kernel, ws_kernel, ptMin, item);  
      });
    });

    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif

    if (oneKernel_) {
      // implemented only for density clustesrs
#ifndef THREE_KERNELS
    numberOfBlocks = 1;
    blockSize      = 32; //1024 - 256; SYCL_BUG_
    stream.submit([&](sycl::handler &cgh) {
      auto soa_kernel  = soa;
      auto ws_kernel   = ws_d.get();
      auto minT_kernel = minT;
      auto eps_kernel  = eps;
      auto errmax_kernel = errmax;
      auto chi2max_kernel = chi2max;
      constexpr int d = 8;
      constexpr int sb = 1 << d;
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              c_acc(sb, cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ct_acc(sb, cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              cu_acc(sb, cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ibs_acc(1, cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              p_acc(1, cgh); 
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              firstNeg_acc(1, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(32)]]{ 
	            vertexFinderOneKernel(soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item,
                             (int32_t *)c_acc.get_pointer(), 
                             (int32_t *)ct_acc.get_pointer(), 
                             (int32_t *)cu_acc.get_pointer(), 
                             (int *)ibs_acc.get_pointer(), 
                             (int *)p_acc.get_pointer(),
                             (uint32_t *)firstNeg_acc.get_pointer());
      });
    });
    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif

#else
    numberOfBlocks = 1;
    blockSize      = 1024 - 256;
    stream.submit([&](sycl::handler &cgh) {
      auto soa_kernel = soa;
      auto ws_kernel  = ws_d.get();
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { 
              vertexFinderKernel1(soa_kernel, 
                                  ws_kernel, 
                                  minT, 
                                  eps,
                                  errmax,
                                  chi2max,
                                  item);
      });
    });

    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif

    numberOfBlocks = 1;
    blockSize      = 32; //1024 - 256;
    stream.submit([&](sycl::handler &cgh) {
      auto soa_kernel = soa;
      auto ws_kernel  = ws_d.get();
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { 
              splitVerticesKernel(soa_kernel, ws_kernel, 9.f, item);
      });
    });

    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif

    numberOfBlocks = 1;
    blockSize      = 32; //1024 - 256; SYCL_BUG_
    stream.submit([&](sycl::handler &cgh) {
      auto soa_kernel = soa;
      auto ws_kernel  = ws_d.get();
      constexpr int d = 8;
      constexpr int sb = 1 << d;
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              c_acc(sb, cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ct_acc(sb, cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              cu_acc(sb, cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ibs_acc(1, cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              p_acc(1, cgh); 
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              firstNeg_acc(1, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { 
              vertexFinderKernel2(soa_kernel, ws_kernel, item,
                             (int32_t *)c_acc.get_pointer(), 
                             (int32_t *)ct_acc.get_pointer(), 
                             (int32_t *)cu_acc.get_pointer(), 
                             (int *)ibs_acc.get_pointer(), 
                             (int *)p_acc.get_pointer(),
                             (uint32_t *)firstNeg_acc.get_pointer());
      });
    });

    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif

#endif
    } else {  // five kernels
      if (useDensity_) {
      numberOfBlocks = 1;
      blockSize      = 1024 - 256;
      stream.submit([&](sycl::handler &cgh) {
        auto soa_kernel = soa;
        auto ws_kernel  = ws_d.get();
        auto minT_kernel = minT;
        auto eps_kernel  = eps;
        auto errmax_kernel = errmax;
        auto chi2max_kernel = chi2max;
        cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(32)]]{ 
              clusterTracksByDensityKernel(soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item);               
      });
    });

    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif

      } else if (useDBSCAN_) {
      numberOfBlocks = 1;
      blockSize      = 1024 - 256;
      stream.submit([&](sycl::handler &cgh) {
        auto soa_kernel = soa;
        auto ws_kernel  = ws_d.get();
        auto minT_kernel = minT;
        auto eps_kernel  = eps;
        auto errmax_kernel = errmax;
        auto chi2max_kernel = chi2max;
        cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(32)]]{ 
              clusterTracksDBSCAN(soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item);
      });
    });

    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif

      } else if (useIterative_) {
      numberOfBlocks = 1;
      blockSize      = 32; //1024 - 256; SYCL_BUG_
      stream.submit([&](sycl::handler &cgh) {
        auto soa_kernel = soa;
        auto ws_kernel  = ws_d.get();
        auto minT_kernel = minT;
        auto eps_kernel  = eps;
        auto errmax_kernel = errmax;
        auto chi2max_kernel = chi2max;
        cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(32)]]{ 
              clusterTracksIterative(soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item);
      });
    });
      }

    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif

      numberOfBlocks = 1;
      blockSize      = 1024 - 256;
      stream.submit([&](sycl::handler &cgh) {
        auto soa_kernel = soa;
        auto ws_kernel  = ws_d.get();
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { 
                fitVerticesKernel(soa_kernel, ws_kernel, 50., item);
      });
    });

    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif

      // one block per vertex...
      numberOfBlocks = 1024;
      blockSize      = 32; //128 SYCL_BUG_
      stream.submit([&](sycl::handler &cgh) {
      auto soa_kernel = soa;
      auto ws_kernel  = ws_d.get();
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { 
              splitVerticesKernel(soa_kernel, ws_kernel, 9.f, item);
      });
    });

    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif

      numberOfBlocks = 1;
      blockSize      = 1024 - 256;
      stream.submit([&](sycl::handler &cgh) {
        auto soa_kernel = soa;
        auto ws_kernel  = ws_d.get();
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { 
                fitVerticesKernel(soa_kernel, ws_kernel, 5000., item);

      });
    });

    if((stream.get_device()).is_cpu())
        stream.wait();

#ifdef GPU_DEBUG
    stream.wait();
#endif
      numberOfBlocks = 1;
      blockSize      = 32; //1024-256 SYCL_BUG_
      stream.submit([&](sycl::handler &cgh) {
        auto soa_kernel = soa;
        auto ws_kernel  = ws_d.get();
        constexpr int d = 8;
      constexpr int sb = 1 << d;
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              c_acc(sb, cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ct_acc(sb, cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              cu_acc(sb, cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ibs_acc(1, cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              p_acc(1, cgh); 
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              firstNeg_acc(1, cgh);
        cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { 
              sortByPt2Kernel(soa_kernel, ws_kernel, item,
                             (int32_t *)c_acc.get_pointer(), 
                             (int32_t *)ct_acc.get_pointer(), 
                             (int32_t *)cu_acc.get_pointer(), 
                             (int *)ibs_acc.get_pointer(), 
                             (int *)p_acc.get_pointer(),
                             (uint32_t *)firstNeg_acc.get_pointer());
      });
    });
    }

    if((stream.get_device()).is_cpu())
        stream.wait();
 
#ifdef GPU_DEBUG
    stream.wait();
#endif

    return vertices;
  }

}  // namespace gpuVertexFinder

#undef FROM
