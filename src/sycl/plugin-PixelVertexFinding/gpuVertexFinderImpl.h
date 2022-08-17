#include "SYCLCore/syclAtomic.h"

#include "gpuClusterTracksByDensity.h"
#include "gpuClusterTracksDBSCAN.h"
#include "gpuClusterTracksIterative.h"
#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

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
      auto it = cms::sycltools::AtomicAdd<uint32_t>(&data.ntrks, 1);
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
                             Hist *hist_acc,
                             Hist::Counter *hws_acc,
                             unsigned int *foundClusters_acc,
                             int* noise_acc,
                             uint32_t *it_acc,
                             float *zz_acc,
                             uint8_t *newV_acc,
                             float *ww_acc,
                             uint32_t *nq_acc,
                             float *znew_acc,
                             float *wnew_acc,
                             uint32_t *igv_acc,
                             uint16_t* sws_acc, 
                             int32_t *c_acc, 
                             int32_t *ct_acc, 
                             int32_t *cu_acc, 
                             int *ibs_acc, 
                             int *p_acc,
                             uint32_t *firstNeg_acc,
                             sycl::stream out
  ) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max, item, hist_acc, hws_acc, foundClusters_acc, out); 
    item.barrier();
    fitVertices(pdata, pws, 50., item, noise_acc, out);
    item.barrier();
    splitVertices(pdata, pws, 9.f, item, it_acc, zz_acc, newV_acc, ww_acc, nq_acc, znew_acc, wnew_acc, igv_acc, out);
    item.barrier();
    fitVertices(pdata, pws, 5000., item, noise_acc, out);
    item.barrier();
    sortByPt2(pdata, pws, item, sws_acc, c_acc, ct_acc, cu_acc, ibs_acc, p_acc, firstNeg_acc);
  }
#else
  void vertexFinderKernel1(gpuVertexFinder::ZVertices* pdata,
                           gpuVertexFinder::WorkSpace* pws,
                           int minT,      // min number of neighbours to be "seed"
                           float eps,     // max absolute distance to cluster
                           float errmax,  // max error to be "seed"
                           float chi2max  // max normalized distance to cluster,
                           sycl::nd_item<1> item,
                           int* noise_acc,

                           sycl::stream out) {
    (pdata, pws, minT, eps, errmax, chi2max, item, hist_acc, hws_acc, foundClusters_acc, out);
    item.barrier();
    fitVertices(pdata, pws, 50., item, noise_acc, out);
  }

  void vertexFinderKernel2(gpuVertexFinder::ZVertices* pdata, gpuVertexFinder::WorkSpace* pws,
                           sycl::nd_item<1> item, 
                           int* noise_acc,
                           uint16_t* sws_acc, 
                           int32_t *c_acc, 
                           int32_t *ct_acc, 
                           int32_t *cu_acc, 
                           int *ibs_acc, 
                           int *p_acc,
                           uint32_t *firstNeg_acc,
                           sycl::stream out
  ) {
    fitVertices(pdata, pws, 5000., item, noise_acc, out);
    item.barrier();
    sortByPt2(pdata, pws, item, sws_acc, c_acc, ct_acc, cu_acc, ibs_acc, p_acc, firstNeg_acc);
  }
#endif

ZVertexHeterogeneous Producer::makeAsync(sycl::queue stream, TkSoA const* tksoa, float ptMin) const {
    // std::cout << "producing Vertices on GPU" << std::endl;
        
    constexpr int MAXTK = 512;

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

    if (oneKernel_) {
      // implemented only for density clustesrs
#ifndef THREE_KERNELS
    numberOfBlocks = 1;
    blockSize      = 1024 - 256;
    stream.submit([&](sycl::handler &cgh) {
      auto soa_kernel  = soa;
      auto ws_kernel   = ws_d.get();
      auto minT_kernel = minT;
      auto eps_kernel  = eps;
      auto errmax_kernel = errmax;
      auto chi2max_kernel = chi2max;
      sycl::accessor<Hist, 1, sycl::access_mode::read_write, sycl::access::target::local>
              hist_acc(sycl::range<1>(sizeof(Hist)), cgh);
      sycl::accessor<Hist::Counter, 1, sycl::access_mode::read_write, sycl::access::target::local>
              hws_acc(sycl::range<1>(sizeof(Hist::Counter) * 32), cgh);
      sycl::accessor<unsigned int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              foundClusters_acc(sycl::range<1>(sizeof(unsigned int)), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              noise_acc(sycl::range<1>(sizeof(int)), cgh);
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              it_acc(sycl::range<1>(sizeof(uint32_t) * MAXTK), cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              zz_acc(sycl::range<1>(sizeof(float) * MAXTK), cgh);
      sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              newV_acc(sycl::range<1>(sizeof(uint8_t)) * MAXTK, cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ww_acc(sycl::range<1>(sizeof(float) * MAXTK), cgh);
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              nq_acc(sycl::range<1>(sizeof(uint32_t)), cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              znew_acc(sycl::range<1>(sizeof(float) * 2), cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              wnew_acc(sycl::range<1>(sizeof(float) * 2), cgh);
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              igv_acc(sycl::range<1>(sizeof(uint32_t)), cgh);
      sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              sws_acc(sycl::range<1>(sizeof(uint16_t)), cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              c_acc(sycl::range<1>(sizeof(int32_t)), cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ct_acc(sycl::range<1>(sizeof(int32_t)), cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              cu_acc(sycl::range<1>(sizeof(int32_t)), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ibs_acc(sycl::range<1>(sizeof(int)), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              p_acc(sycl::range<1>(sizeof(int)), cgh); 
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              firstNeg_acc(sycl::range<1>(sizeof(uint32_t)), cgh);
      sycl::stream out(1024, 768, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
              vertexFinderOneKernel(soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item,
                             (Hist *)hist_acc.get_pointer(),
                             (Hist::Counter *)hws_acc.get_pointer(),
                             (unsigned int *)foundClusters_acc.get_pointer(),
                             (int* )noise_acc.get_pointer(),
                             (uint32_t *)it_acc.get_pointer(),
                             (float *)zz_acc.get_pointer(),
                             (uint8_t *)newV_acc.get_pointer(),
                             (float *)ww_acc.get_pointer(),
                             (uint32_t *)nq_acc.get_pointer(),
                             (float *)znew_acc.get_pointer(),
                             (float *)wnew_acc.get_pointer(),
                             (uint32_t *)igv_acc.get_pointer(),
                             (uint16_t *)sws_acc.get_pointer(), 
                             (int32_t *)c_acc.get_pointer(), 
                             (int32_t *)ct_acc.get_pointer(), 
                             (int32_t *)cu_acc.get_pointer(), 
                             (int *)ibs_acc.get_pointer(), 
                             (int *)p_acc.get_pointer(),
                             (uint32_t *)firstNeg_acc.get_pointer(),
                             out);
      });
    });
#else
    numberOfBlocks = 1;
    blockSize      = 1024 - 256;
    stream.submit([&](sycl::handler &cgh) {
      auto soa_kernel = soa;
      auto ws_kernel  = ws_d.get();
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              noise_acc(sycl::range<1>(sizeof(int)), cgh);
      sycl::stream out(1024, 768, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
              vertexFinderKernel1(soa_kernel, 
                                  ws_kernel, 
                                  minT, 
                                  eps,
                                  errmax,
                                  chi2max,
                                  item,
                                  (int *)noise_acc.get_pointer(),
                                  out);
      });
    });

    numberOfBlocks = 1;
    blockSize      = 1024 - 256;
    stream.submit([&](sycl::handler &cgh) {
      auto soa_kernel = soa;
      auto ws_kernel  = ws_d.get();
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              it_acc(sycl::range<1>(sizeof(uint32_t) * MAXTK), cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              zz_acc(sycl::range<1>(sizeof(float) * MAXTK), cgh);
      sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              newV_acc(sycl::range<1>(sizeof(uint8_t)) * MAXTK, cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ww_acc(sycl::range<1>(sizeof(float) * MAXTK), cgh);
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              nq_acc(sycl::range<1>(sizeof(uint32_t)), cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              znew_acc(sycl::range<1>(sizeof(float) * 2), cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              wnew_acc(sycl::range<1>(sizeof(float) * 2), cgh);
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              igv_acc(sycl::range<1>(sizeof(uint32_t)), cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
              splitVerticesKernel(soa_kernel, ws_kernel, 9.f, item,
                           (uint32_t *)it_acc.get_pointer(),
                           (float *)zz_acc.get_pointer(),
                           (uint8_t *)newV_acc.get_pointer(),
                           (float *)ww_acc.get_pointer(),
                           (uint32_t *)nq_acc.get_pointer(),
                           (float *)znew_acc.get_pointer(),
                           (float *)wnew_acc.get_pointer(),
                           (uint32_t *)igv_acc.get_pointer(),
                           out);
      });
    });

    numberOfBlocks = 1;
    blockSize      = 1024 - 256;
    stream.submit([&](sycl::handler &cgh) {
      auto soa_kernel = soa;
      auto ws_kernel  = ws_d.get();
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              noise_acc(sycl::range<1>(sizeof(int)), cgh);
      sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              sws_acc(sycl::range<1>(sizeof(uint16_t)), cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              c_acc(sycl::range<1>(sizeof(int32_t)), cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ct_acc(sycl::range<1>(sizeof(int32_t)), cgh);
      sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              cu_acc(sycl::range<1>(sizeof(int32_t)), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ibs_acc(sycl::range<1>(sizeof(int)), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              p_acc(sycl::range<1>(sizeof(int)), cgh); 
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              firstNeg_acc(sycl::range<1>(sizeof(uint32_t)), cgh);
      sycl::stream out(1024, 768, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
              vertexFinderKernel2(soa_kernel, ws_kernel, item, 
                           (int *)noise_acc.get_pointer(),
                           (uint16_t *)sws_acc.get_pointer(), 
                           (int32_t *)c_acc.get_pointer(), 
                           (int32_t *)ct_acc.get_pointer(), 
                           (int32_t *)cu_acc.get_pointer(), 
                           (int *)ibs_acc.get_pointer(), 
                           (int *)p_acc.get_pointer(),
                           (uint32_t *)firstNeg_acc.get_pointer(),
                           out);
      });
    });
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
        sycl::accessor<Hist, 1, sycl::access_mode::read_write, sycl::access::target::local>
              hist_acc(sycl::range<1>(sizeof(Hist)), cgh);
        sycl::accessor<Hist::Counter, 1, sycl::access_mode::read_write, sycl::access::target::local>
              hws_acc(sycl::range<1>(sizeof(Hist::Counter) * 32), cgh);
        sycl::accessor<unsigned int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              foundClusters_acc(sycl::range<1>(sizeof(unsigned int)), cgh);
        sycl::stream out(1024, 768, cgh);
        cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
              clusterTracksByDensityKernel(soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item,
                                    (Hist *)hist_acc.get_pointer(), (Hist::Counter *)hws_acc.get_pointer(),
                                    (unsigned int *)foundClusters_acc.get_pointer(), out);               
      });
    });
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
        sycl::accessor<Hist, 1, sycl::access_mode::read_write, sycl::access::target::local>
              hist_acc(sycl::range<1>(sizeof(Hist)), cgh);
        sycl::accessor<Hist::Counter, 1, sycl::access_mode::read_write, sycl::access::target::local>
              hws_acc(sycl::range<1>(sizeof(Hist::Counter) * 32), cgh);
        sycl::accessor<unsigned int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              foundClusters_acc(sycl::range<1>(sizeof(unsigned int)), cgh);
        sycl::stream out(1024, 768, cgh);
        cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
              clusterTracksDBSCAN(soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item,
                                  (Hist *)hist_acc.get_pointer(),
                                  (Hist::Counter*) hws_acc.get_pointer(),
                                  (unsigned int*) foundClusters_acc.get_pointer(),
                                  out);
      });
    });
      } else if (useIterative_) {
      numberOfBlocks = 1;
      blockSize      = 1024 - 256;
      stream.submit([&](sycl::handler &cgh) {
        auto soa_kernel = soa;
        auto ws_kernel  = ws_d.get();
        auto minT_kernel = minT;
        auto eps_kernel  = eps;
        auto errmax_kernel = errmax;
        auto chi2max_kernel = chi2max;
        sycl::accessor<Hist, 1, sycl::access_mode::read_write, sycl::access::target::local>
              hist_acc(sycl::range<1>(sizeof(Hist)), cgh);
        sycl::accessor<Hist::Counter, 1, sycl::access_mode::read_write, sycl::access::target::local>
              hws_acc(sycl::range<1>(sizeof(Hist::Counter) * 32), cgh);
        sycl::accessor<unsigned int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              foundClusters_acc(sycl::range<1>(sizeof(unsigned int)), cgh);
        sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              nloops_acc(sycl::range<1>(sizeof(int)), cgh);
        sycl::stream out(1024, 768, cgh);
        cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
              clusterTracksIterative(soa_kernel, ws_kernel, minT_kernel, eps_kernel, errmax_kernel, chi2max_kernel, item, 
                                     (Hist *)hist_acc.get_pointer(), (Hist::Counter *)hws_acc.get_pointer(), 
                                     (unsigned int *)foundClusters_acc.get_pointer(), (int *)nloops_acc.get_pointer(), out);
      });
    });
      }
      numberOfBlocks = 1;
      blockSize      = 1024 - 256;
      stream.submit([&](sycl::handler &cgh) {
        auto soa_kernel = soa;
        auto ws_kernel  = ws_d.get();
        sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              noise_acc(sycl::range<1>(sizeof(int)), cgh);
        sycl::stream out(1024, 768, cgh);

      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
                fitVerticesKernel(soa_kernel, ws_kernel, 50., item, (int *)noise_acc.get_pointer(), out);
      });
    });
      // one block per vertex...
      numberOfBlocks = 1024;
      blockSize      = 128;
      stream.submit([&](sycl::handler &cgh) {
      auto soa_kernel = soa;
      auto ws_kernel  = ws_d.get();
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              it_acc(sycl::range<1>(sizeof(uint32_t) * MAXTK), cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              zz_acc(sycl::range<1>(sizeof(float) * MAXTK), cgh);
      sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              newV_acc(sycl::range<1>(sizeof(uint8_t) * MAXTK), cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ww_acc(sycl::range<1>(sizeof(float) * MAXTK), cgh);
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              nq_acc(sycl::range<1>(sizeof(uint32_t)), cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              znew_acc(sycl::range<1>(sizeof(float) * 2), cgh);
      sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
              wnew_acc(sycl::range<1>(sizeof(float) * 2), cgh);
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              igv_acc(sycl::range<1>(sizeof(uint32_t)), cgh);
      sycl::stream out(1024, 768, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
              splitVerticesKernel(soa_kernel, ws_kernel, 9.f, item,
                           (uint32_t *)it_acc.get_pointer(),
                           (float *)zz_acc.get_pointer(),
                           (uint8_t *)newV_acc.get_pointer(),
                           (float *)ww_acc.get_pointer(),
                           (uint32_t *)nq_acc.get_pointer(),
                           (float *)znew_acc.get_pointer(),
                           (float *)wnew_acc.get_pointer(),
                           (uint32_t *)igv_acc.get_pointer(),
                           out);
      });
    });
      numberOfBlocks = 1;
      blockSize      = 1024 - 256;
      stream.submit([&](sycl::handler &cgh) {
        auto soa_kernel = soa;
        auto ws_kernel  = ws_d.get();
        sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
              noise_acc(sycl::range<1>(sizeof(int)), cgh);
        sycl::stream out(1024, 768, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
                fitVerticesKernel(soa_kernel, ws_kernel, 5000., item, (int *)noise_acc.get_pointer(), out);

      });
    });

      stream.submit([&](sycl::handler &cgh) {
        auto soa_kernel = soa;
        auto ws_kernel  = ws_d.get();
        sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              sws_acc(sycl::range<1>(sizeof(uint16_t) * 32), cgh);
        sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
                c_acc(sycl::range<1>(sizeof(int32_t)), cgh);
        sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
                ct_acc(sycl::range<1>(sizeof(int32_t)), cgh);
        sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
                cu_acc(sycl::range<1>(sizeof(int32_t)), cgh);
        sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
                ibs_acc(sycl::range<1>(sizeof(int)), cgh);
        sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
                p_acc(sycl::range<1>(sizeof(int)), cgh);
        sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
                firstNeg_acc(sycl::range<1>(sizeof(uint32_t)), cgh);
        cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
          [=](sycl::nd_item<1> item){ 
               sortByPt2Kernel(soa_kernel, ws_kernel, item, 
                              (uint16_t *)sws_acc.get_pointer(), (int32_t *)c_acc.get_pointer(), 
                              (int32_t *)ct_acc.get_pointer(), (int32_t *)cu_acc.get_pointer(), 
                              (int *)ibs_acc.get_pointer(), (int *)p_acc.get_pointer(), (uint32_t *)firstNeg_acc.get_pointer());
      });
    });
    }
    return vertices;
  }

}  // namespace gpuVertexFinder

#undef FROM
