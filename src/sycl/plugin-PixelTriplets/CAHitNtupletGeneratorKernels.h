#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h

#include "SYCLDataFormats/PixelTrackHeterogeneous.h"
#include "GPUCACell.h"

// #define DUMP_GPU_TK_TUPLES

namespace cAHitNtupletGenerator {

  // counters
  struct Counters {
    unsigned long long nEvents;
    unsigned long long nHits;
    unsigned long long nCells;
    unsigned long long nTuples;
    unsigned long long nFitTracks;
    unsigned long long nGoodTracks;
    unsigned long long nUsedHits;
    unsigned long long nDupHits;
    unsigned long long nKilledCells;
    unsigned long long nEmptyCells;
    unsigned long long nZeroTrackCells;
  };

  using HitsView = TrackingRecHit2DSOAView;
  using HitsOnGPU = TrackingRecHit2DSOAView;

  using HitToTuple = CAConstants::HitToTuple;
  using TupleMultiplicity = CAConstants::TupleMultiplicity;

  using Quality = pixelTrack::Quality;
  using TkSoA = pixelTrack::TrackSoA;
  using HitContainer = pixelTrack::HitContainer;

  struct QualityCuts {
    // chi2 cut = chi2Scale * (chi2Coeff[0] + pT/GeV * (chi2Coeff[1] + pT/GeV * (chi2Coeff[2] + pT/GeV * chi2Coeff[3])))
    float chi2Coeff[4];
    float chi2MaxPt;  // GeV
    float chi2Scale;

    struct region {
      float maxTip;  // cm
      float minPt;   // GeV
      float maxZip;  // cm
    };

    region triplet;
    region quadruplet;
  };

  // params
  struct Params {
    Params(bool onGPU,
           uint32_t minHitsPerNtuplet,
           uint32_t maxNumberOfDoublets,
           bool useRiemannFit,
           bool fit5as4,
           bool includeJumpingForwardDoublets,
           bool earlyFishbone,
           bool lateFishbone,
           bool idealConditions,
           bool doStats,
           bool doClusterCut,
           bool doZ0Cut,
           bool doPtCut,
           float ptmin,
           float CAThetaCutBarrel,
           float CAThetaCutForward,
           float hardCurvCut,
           float dcaCutInnerTriplet,
           float dcaCutOuterTriplet,
           QualityCuts const& cuts)
        : onGPU_(onGPU),
          minHitsPerNtuplet_(minHitsPerNtuplet),
          maxNumberOfDoublets_(maxNumberOfDoublets),
          useRiemannFit_(useRiemannFit),
          fit5as4_(fit5as4),
          includeJumpingForwardDoublets_(includeJumpingForwardDoublets),
          earlyFishbone_(earlyFishbone),
          lateFishbone_(lateFishbone),
          idealConditions_(idealConditions),
          doStats_(doStats),
          doClusterCut_(doClusterCut),
          doZ0Cut_(doZ0Cut),
          doPtCut_(doPtCut),
          ptmin_(ptmin),
          CAThetaCutBarrel_(CAThetaCutBarrel),
          CAThetaCutForward_(CAThetaCutForward),
          hardCurvCut_(hardCurvCut),
          dcaCutInnerTriplet_(dcaCutInnerTriplet),
          dcaCutOuterTriplet_(dcaCutOuterTriplet),
          cuts_(cuts) {}

    const bool onGPU_;
    const uint32_t minHitsPerNtuplet_;
    const uint32_t maxNumberOfDoublets_;
    const bool useRiemannFit_;
    const bool fit5as4_;
    const bool includeJumpingForwardDoublets_;
    const bool earlyFishbone_;
    const bool lateFishbone_;
    const bool idealConditions_;
    const bool doStats_;
    const bool doClusterCut_;
    const bool doZ0Cut_;
    const bool doPtCut_;
    const float ptmin_;
    const float CAThetaCutBarrel_;
    const float CAThetaCutForward_;
    const float hardCurvCut_;
    const float dcaCutInnerTriplet_;
    const float dcaCutOuterTriplet_;

    // quality cuts
    QualityCuts cuts_{// polynomial coefficients for the pT-dependent chi2 cut
                      {0.68177776, 0.74609577, -0.08035491, 0.00315399},
                      // max pT used to determine the chi2 cut
                      10.,
                      // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
                      30.,
                      // regional cuts for triplets
                      {
                          0.3,  // |Tip| < 0.3 cm
                          0.5,  // pT > 0.5 GeV
                          12.0  // |Zip| < 12.0 cm
                      },
                      // regional cuts for quadruplets
                      {
                          0.5,  // |Tip| < 0.5 cm
                          0.3,  // pT > 0.3 GeV
                          12.0  // |Zip| < 12.0 cm
                      }};

  };  // Params

}  // namespace cAHitNtupletGenerator

template <typename TTraits>
class CAHitNtupletGeneratorKernels {
public:
  using Traits = TTraits;

  using QualityCuts = cAHitNtupletGenerator::QualityCuts;
  using Params = cAHitNtupletGenerator::Params;
  using Counters = cAHitNtupletGenerator::Counters;

  template <typename T>
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using HitsView = TrackingRecHit2DSOAView;
  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DHeterogeneous<Traits>;

  using HitToTuple = CAConstants::HitToTuple;
  using TupleMultiplicity = CAConstants::TupleMultiplicity;

  using Quality = pixelTrack::Quality;
  using TkSoA = pixelTrack::TrackSoA;
  using HitContainer = pixelTrack::HitContainer;

  CAHitNtupletGeneratorKernels(Params const& params) : m_params(params) {}
  ~CAHitNtupletGeneratorKernels() = default;

  TupleMultiplicity const* tupleMultiplicity() const { return device_tupleMultiplicity_.get(); }

  void launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, sycl::queue stream){
  // these are pointer on GPU!
  auto *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  // zero tuples
  cms::sycltools::launchZero(tuples_d, stream);

  auto nhits = hh.nHits();
  assert(nhits <= pixelGPUConstants::maxNumberOfHits);

  // std::cout << "N hits " << nhits << std::endl;
  // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

  //
  // applying conbinatoric cleaning such as fishbone at this stage is too expensive
  //

  auto nthTot = 64;
  auto stride = 4;
  auto blockSize = nthTot / stride;
  auto numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  auto rescale = numberOfBlocks / 65536;
  blockSize *= (rescale + 1);
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  assert(numberOfBlocks < 65536);
  assert(blockSize > 0 && 0 == blockSize % 16);
  sycl::range<3> blks(1, numberOfBlocks, 1);
  sycl::range<3> thrs(stride, blockSize, 1);

  stream.submit([&](sycl::handler &cgh) {
      auto device_hitTuple_apc_kernel     = device_hitTuple_apc_;
      auto device_hitToTuple_apc_kernel   = device_hitToTuple_apc_;  // needed only to be reset, ready for next kernel
      auto hh_kernel                      = hh.view();
      auto device_theCells_kernel         = device_theCells_.get();
      auto device_nCells_kernel           = device_nCells_;
      auto device_theCellNeighbors_kernel = device_theCellNeighbors_.get();
      auto device_isOuterHitOfCell_kernel = device_isOuterHitOfCell_.get();
      auto m_params_kernel = m_params;
      cgh.parallel_for(
          sycl::nd_range<3>(blks * thrs, thrs),
          [=](sycl::nd_item<3> item){ 
              kernel_connect(device_hitTuple_apc_kernel,
                             device_hitToTuple_apc_kernel,  // needed only to be reset, ready for next kernel
                             hh_kernel,
                             device_theCells_kernel,
                             device_nCells_kernel,
                             device_theCellNeighbors_kernel,
                             device_isOuterHitOfCell_kernel,
                             m_params_kernel.hardCurvCut_,
                             m_params_kernel.ptmin_,
                             m_params_kernel.CAThetaCutBarrel_,
                             m_params_kernel.CAThetaCutForward_,
                             m_params_kernel.dcaCutInnerTriplet_,
                             m_params_kernel.dcaCutOuterTriplet_,
                             item);
      });
    });
  //cudaCheck(cudaGetLastError());

  if (nhits > 1 && m_params.earlyFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    sycl::range<3> blks(1, numberOfBlocks, 1);
    sycl::range<3> thrs(stride, blockSize, 1);
    stream.submit([&](sycl::handler &cgh) {
      auto hh_kernel                      = hh.view();
      auto device_theCells_kernel         = device_theCells_.get();
      auto device_nCells_kernel           = device_nCells_;
      auto device_isOuterHitOfCell_kernel = device_isOuterHitOfCell_.get();
      cgh.parallel_for(
          sycl::nd_range<3>(blks * thrs, thrs),
          [=](sycl::nd_item<3> item){ 
              gpuPixelDoublets::fishbone(hh_kernel, 
                                         device_theCells_kernel, 
                                         device_nCells_kernel, 
                                         device_isOuterHitOfCell_kernel, 
                                         nhits, 
                                         false,
                                         item);
      
      });
    });

    //cudaCheck(cudaGetLastError());
  }

  blockSize = 64;
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      auto m_params_kernel = m_params;
      auto hh_kernel                      = hh.view();
      auto device_theCells_kernel         = device_theCells_.get();
      auto device_nCells_kernel           = device_nCells_;
      auto device_theCellTracks_kernel    = device_theCellTracks_.get();
      auto tuples_d_kernel                = tuples_d;
      auto device_hitTuple_apc_kernel     = device_hitTuple_apc_;
      auto quality_d_kernel               = quality_d;
      auto out = sycl::stream(1024, 768, h);
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_find_ntuplets(hh_kernel,
                                   device_theCells_kernel,
                                   device_nCells_kernel,
                                   device_theCellTracks_kernel,
                                   tuples_d_kernel,
                                   device_hitTuple_apc_kernel,
                                   quality_d_kernel,
                                   m_params_kernel.minHitsPerNtuplet_,
                                   item);
   
      });
    });
  
   //cudaCheck(cudaGetLastError());

  if (m_params.doStats_)
    stream.submit([&](sycl::handler &cgh) {
      auto hh_kernel                      = hh.view();
      auto device_theCells_kernel         = device_theCells_.get();
      auto device_nCells_kernel           = device_nCells_;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_mark_used(hh_kernel, device_theCells_kernel, device_nCells_kernel, item);
      });
    });
  //cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  stream.wait();
  //cudaCheck(cudaGetLastError());
#endif

  blockSize = 128;
  numberOfBlocks = (HitContainer::totbins() + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel                = tuples_d;
      auto device_hitTuple_apc_kernel     = device_hitTuple_apc_;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              cms::sycltools::finalizeBulk(device_hitTuple_apc_kernel, tuples_d_kernel, item);
      });
    });

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel                = tuples_d;
      auto quality_d_kernel               = quality_d;
      auto device_theCells_kernel         = device_theCells_.get();
      auto device_nCells_kernel           = device_nCells_;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_earlyDuplicateRemover(device_theCells_kernel, device_nCells_kernel, tuples_d_kernel, quality_d_kernel, item);  
      });
    });
  
  //cudaCheck(cudaGetLastError());

  blockSize = 128;
  numberOfBlocks = (3 * CAConstants::maxTuples() / 4 + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel                 = tuples_d;
      auto device_tupleMultiplicity_kernel = device_tupleMultiplicity_.get();
      auto quality_d_kernel                = quality_d;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_countMultiplicity(tuples_d_kernel, quality_d_kernel, device_tupleMultiplicity_kernel, item); 
      });
    });
  
  cms::sycltools::launchFinalize(device_tupleMultiplicity_.get(), stream);
  stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel                 = tuples_d;
      auto device_tupleMultiplicity_kernel = device_tupleMultiplicity_.get();
      auto quality_d_kernel                = quality_d;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_fillMultiplicity(tuples_d_kernel, quality_d_kernel, device_tupleMultiplicity_kernel, item);  
      });
    });
  
  //cudaCheck(cudaGetLastError());

  if (nhits > 1 && m_params.lateFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    sycl::range<3> blks(1, numberOfBlocks, 1);
    sycl::range<3> thrs(stride, blockSize, 1);
    stream.submit([&](sycl::handler &cgh) {
      auto hh_kernel                      = hh.view();
      auto device_theCells_kernel         = device_theCells_.get();
      auto device_nCells_kernel           = device_nCells_;
      auto device_isOuterHitOfCell_kernel = device_isOuterHitOfCell_.get();
      cgh.parallel_for(
          sycl::nd_range<3>(blks * thrs, thrs),
          [=](sycl::nd_item<3> item){ 
              gpuPixelDoublets::fishbone(hh_kernel, 
                                         device_theCells_kernel, 
                                         device_nCells_kernel, 
                                         device_isOuterHitOfCell_kernel, 
                                         nhits, 
                                         true,
                                         item);
      
      });
    });
    //cudaCheck(cudaGetLastError());
  }

  if (m_params.doStats_) {
    numberOfBlocks = (std::max(nhits, m_params.maxNumberOfDoublets_) + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel                 = tuples_d;
      auto device_tupleMultiplicity_kernel = device_tupleMultiplicity_.get();
      auto device_hitTuple_apc_kernel      = device_hitTuple_apc_;
      auto device_theCells_kernel          = device_theCells_.get();
      auto device_nCells_kernel            = device_nCells_;
      auto device_theCellNeighbors_kernel  = device_theCellNeighbors_.get();
      auto device_theCellTracks_kernel     = device_theCellTracks_.get();
      auto device_isOuterHitOfCell_kernel  = device_isOuterHitOfCell_.get();
      auto m_params_kernel                 = m_params;
      auto counters_kernel                 = counters_;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_checkOverflows(tuples_d_kernel,
                                    device_tupleMultiplicity_kernel,
                                    device_hitTuple_apc_kernel,
                                    device_theCells_kernel,
                                    device_nCells_kernel,
                                    device_theCellNeighbors_kernel,
                                    device_theCellTracks_kernel,
                                    device_isOuterHitOfCell_kernel,
                                    nhits,
                                    m_params_kernel.maxNumberOfDoublets_,
                                    counters_kernel,
                                    item);  
      });
    });
    //cudaCheck(cudaGetLastError());
  }
#ifdef GPU_DEBUG
  stream.wait();
  //cudaCheck(cudaGetLastError());
#endif

  // free space asap
  // device_isOuterHitOfCell_.reset();
};

  void classifyTuples(HitsOnCPU const& hh, TkSoA* tracks_d, sycl::queue stream){
  // these are pointer on GPU!
  auto const *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  auto blockSize = 64;

  // classify tracks based on kinematics
  auto numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      auto m_params_kernel = m_params;
      auto tuples_d_kernel = tuples_d;
      auto tracks_d_kernel = tracks_d;
      auto quality_d_kernel = quality_d;
      cgh.parallel_for(      
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_classifyTracks(tuples_d_kernel, 
                                    tracks_d_kernel, 
                                    m_params_kernel.cuts_, 
                                    quality_d_kernel,
                                    item);              
      });
    });
  //cudaCheck(cudaGetLastError());

  if (m_params.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      auto device_theCells_kernel = device_theCells_.get();
      auto device_nCells_kernel = device_nCells_;
      auto quality_d_kernel = quality_d;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_fishboneCleaner(device_theCells_kernel, device_nCells_kernel, quality_d_kernel, item); 
      });
    });
    
    //cudaCheck(cudaGetLastError());
  }

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      auto device_theCells_kernel = device_theCells_.get();
      auto device_nCells_kernel = device_nCells_;
      auto tuples_d_kernel = tuples_d;
      auto tracks_d_kernel = tracks_d;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_fastDuplicateRemover(device_theCells_kernel, device_nCells_kernel, tuples_d_kernel, tracks_d_kernel, item);
      });
    });
  
  //cudaCheck(cudaGetLastError());

  if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
    // fill hit->track "map"
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      auto device_hitToTuple_kernel = device_hitToTuple_.get();
      auto tuples_d_kernel = tuples_d;
      auto quality_d_kernel = quality_d;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_countHitInTracks(tuples_d_kernel, quality_d_kernel, device_hitToTuple_kernel, item); 
      });
    });
    
    //cudaCheck(cudaGetLastError());
    cms::sycltools::launchFinalize(device_hitToTuple_.get(), stream);
    //cudaCheck(cudaGetLastError());
    stream.submit([&](sycl::handler &cgh) {
      auto device_hitToTuple_kernel = device_hitToTuple_.get();
      auto tuples_d_kernel = tuples_d;
      auto quality_d_kernel = quality_d;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_fillHitInTracks(tuples_d_kernel, quality_d_kernel, device_hitToTuple_kernel, item);
      });
    });
    //cudaCheck(cudaGetLastError());
  }
  if (m_params.minHitsPerNtuplet_ < 4) {
    // remove duplicates (tracks that share a hit)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      auto hh_kernel = hh.view();
      auto tuples_d_kernel = tuples_d;
      auto tracks_d_kernel = tracks_d;
      auto quality_d_kernel = quality_d;
      auto device_hitToTuple_kernel = device_hitToTuple_.get();
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_tripletCleaner(hh_kernel, tuples_d_kernel, tracks_d_kernel, quality_d_kernel, device_hitToTuple_kernel, item);
      });
    });
    //cudaCheck(cudaGetLastError());
  }

  if (m_params.doStats_) {
    // counters (add flag???)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      auto device_hitToTuple_kernel = device_hitToTuple_.get();
      auto counters_kernel = counters_;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_doStatsForHitInTracks(device_hitToTuple_kernel, counters_kernel, item);
      });
    });
    //cudaCheck(cudaGetLastError());
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel = tuples_d;
      auto quality_d_kernel = quality_d;
      auto counters_kernel = counters_;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_doStatsForTracks(tuples_d_kernel, quality_d_kernel, counters_kernel, item);
      });
    });
    //cudaCheck(cudaGetLastError());
  }
#ifdef GPU_DEBUG
  stream.wait();
  //cudaCheck(cudaGetLastError());
#endif

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  ++iev;
  blockSize = 32;
  stream.submit([&](sycl::handler &cgh) {
      auto hh_kernel = hh.view();
      auto tuples_d_kernel = tuples_d;
      auto tracks_d_kernel = tracks_d;
      auto quality_d_kernel = quality_d;
      auto device_hitToTuple_kernel = device_hitToTuple_.get();
      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_print_found_ntuplets(hh_kernel, tuples_d_kernel, tracks_d_kernel, quality_d_kernel, device_hitToTuple_kernel, 100, iev, item);
      });
    });
#endif
};

  void fillHitDetIndices(HitsView const* hv, TkSoA* tracks_d, sycl::queue stream){
  auto blockSize = 128;
  auto numberOfBlocks = (HitContainer::capacity() + blockSize - 1) / blockSize;
  
  stream.submit([&](sycl::handler &cgh) {
    auto hitIndices_kernel = &tracks_d->hitIndices;
    auto hv_kernel         = hv;
    auto detIndices_kernel = &tracks_d->detIndices;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernel_fillHitDetIndices(hitIndices_kernel, hv_kernel, detIndices_kernel, item);
      });
    });
  
  //cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
  stream.wait();
  //cudaCheck(cudaGetLastError());
#endif
  };

  void buildDoublets(HitsOnCPU const& hh, sycl::queue stream){
  auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
  stream.wait();
  //cudaCheck(cudaGetLastError());
#endif

  // in principle we can use "nhits" to heuristically dimension the workspace...
  device_isOuterHitOfCell_ = cms::sycltools::make_device_unique<GPUCACell::OuterHitOfCell[]>(std::max(1U, nhits), stream);
  assert(device_isOuterHitOfCell_.get());

  cellStorage_ = cms::sycltools::make_device_unique<unsigned char[]>(
      CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellNeighbors) +
          CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellTracks),
      stream);
  device_theCellNeighborsContainer_ = (GPUCACell::CellNeighbors *)cellStorage_.get();
  device_theCellTracksContainer_ =
      (GPUCACell::CellTracks *)(cellStorage_.get() +
                                CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellNeighbors));

  {
    int threadsPerBlock = 128;
    // at least one block!
    int blocks = (std::max(1U, nhits) + threadsPerBlock - 1) / threadsPerBlock;
    stream.submit([&](sycl::handler &cgh) {
      auto device_theCellNeighbors_kernel  = device_theCellNeighbors_.get();
      auto device_theCellNeighborsContainer_kernel  = device_theCellNeighborsContainer_;
      auto device_isOuterHitOfCell_kernel  = device_isOuterHitOfCell_.get();
      auto device_theCellTracks_kernel = device_theCellTracks_.get();
      auto device_theCellTracksContainer_kernel = device_theCellTracksContainer_;
      cgh.parallel_for(
          sycl::nd_range<3>(blocks * sycl::range<3>(1, 1, threadsPerBlock), sycl::range<3>(1, 1, threadsPerBlock)),
          [=](sycl::nd_item<3> item){ 
              gpuPixelDoublets::initDoublets(device_isOuterHitOfCell_kernel,
                                             nhits,
                                             device_theCellNeighbors_kernel,
                                             device_theCellNeighborsContainer_kernel,
                                             device_theCellTracks_kernel,
                                             device_theCellTracksContainer_kernel,
                                             item);
 
      });
    });
        //cudaCheck(cudaGetLastError());
  }

  device_theCells_ = cms::sycltools::make_device_unique<GPUCACell[]>(m_params.maxNumberOfDoublets_, stream);

#ifdef GPU_DEBUG
  stream.wait();
  //cudaCheck(cudaGetLastError());
#endif

  if (0 == nhits)
    return;  // protect against empty events

  // FIXME avoid magic numbers
  auto nActualPairs = gpuPixelDoublets::nPairs;
  if (!m_params.includeJumpingForwardDoublets_)
    nActualPairs = 15;
  if (m_params.minHitsPerNtuplet_ > 3) {
    nActualPairs = 13;
  }

  assert(nActualPairs <= gpuPixelDoublets::nPairs);
  int stride = 4;
  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
  int blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  sycl::range<3> blks(1, blocks, 1);
  sycl::range<3> thrs(stride, threadsPerBlock, 1);
  stream.submit([&](sycl::handler &cgh) {
      auto nActualPairs_kernel             = nActualPairs;
      auto hh_kernel                       = hh.view();
      auto device_theCells_kernel          = device_theCells_.get();
      auto device_nCells_kernel            = device_nCells_;
      auto device_theCellNeighbors_kernel  = device_theCellNeighbors_.get();
      auto device_theCellTracks_kernel     = device_theCellTracks_.get();
      auto device_isOuterHitOfCell_kernel  = device_isOuterHitOfCell_.get();
      auto m_params_kernel                 = m_params;
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              innerLayerCumulativeSize_acc(sycl::range<1>(32), cgh); //FIXME_
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ntot_acc(sycl::range<1>(32), cgh);  //FIXME_ no idea if the arguments passed to the accessors are correct                   
      cgh.parallel_for(
          sycl::nd_range<3>(blks * thrs, thrs),
          [=](sycl::nd_item<3> item){ 
              gpuPixelDoublets::getDoubletsFromHisto(device_theCells_kernel,
                                                     device_nCells_kernel,
                                                     device_theCellNeighbors_kernel,
                                                     device_theCellTracks_kernel,
                                                     hh_kernel,
                                                     device_isOuterHitOfCell_kernel,
                                                     nActualPairs_kernel,
                                                     m_params_kernel.idealConditions_,
                                                     m_params_kernel.doClusterCut_,
                                                     m_params_kernel.doZ0Cut_,
                                                     m_params_kernel.doPtCut_,
                                                     m_params_kernel.maxNumberOfDoublets_,
                                                     item,
                                                     (uint32_t *)innerLayerCumulativeSize_acc.get_pointer(),
                                                     (uint32_t *)ntot_acc.get_pointer());   
      });
    });
  
  //cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  stream.wait();
  //cudaCheck(cudaGetLastError());
#endif
  };

  void allocateOnGPU(sycl::queue stream);
  
  void cleanup(sycl::queue stream);

  static void printCounters(Counters const* counters, sycl::queue stream){
  stream.submit([&](sycl::handler &cgh) {
      auto counters_kernel = counters;
      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item){ 
              kernel_printCounters(counters_kernel); 
      });
    });
  
};

  Counters* counters_ = nullptr;
 
private:
  // workspace
  unique_ptr<unsigned char[]> cellStorage_;
  unique_ptr<CAConstants::CellNeighborsVector> device_theCellNeighbors_;
  CAConstants::CellNeighbors* device_theCellNeighborsContainer_;
  unique_ptr<CAConstants::CellTracksVector> device_theCellTracks_;
  CAConstants::CellTracks* device_theCellTracksContainer_;

  unique_ptr<GPUCACell[]> device_theCells_;
  unique_ptr<GPUCACell::OuterHitOfCell[]> device_isOuterHitOfCell_;
  uint32_t* device_nCells_ = nullptr;

  unique_ptr<HitToTuple> device_hitToTuple_;
  cms::sycltools::AtomicPairCounter* device_hitToTuple_apc_ = nullptr;

  cms::sycltools::AtomicPairCounter* device_hitTuple_apc_ = nullptr;

  unique_ptr<TupleMultiplicity> device_tupleMultiplicity_;

  unique_ptr<cms::sycltools::AtomicPairCounter::c_type[]> device_storage_;
  // params
  Params const& m_params;
};

using CAHitNtupletGeneratorKernelsGPU = CAHitNtupletGeneratorKernels<cms::syclcompat::GPUTraits>;

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h

