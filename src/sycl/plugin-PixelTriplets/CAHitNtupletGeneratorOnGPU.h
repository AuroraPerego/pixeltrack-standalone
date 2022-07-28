#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h

#include <CL/sycl.hpp>

#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include "SYCLCore/SimpleVector.h"
#include "SYCLDataFormats/PixelTrackHeterogeneous.h"
#include "SYCLDataFormats/TrackingRecHit2DSYCL.h"
#include "Framework/Event.h"

#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "HelixFitOnGPU.h"


namespace {

  template <typename T>
  T sqr(T x) {
    return x * x;
  }

  cAHitNtupletGenerator::QualityCuts makeQualityCuts() {
    auto coeff = std::vector<double>{0.68177776, 0.74609577, -0.08035491, 0.00315399};  // chi2Coeff
    return cAHitNtupletGenerator::QualityCuts{// polynomial coefficients for the pT-dependent chi2 cut
                                              {(float)coeff[0], (float)coeff[1], (float)coeff[2], (float)coeff[3]},
                                              // max pT used to determine the chi2 cut
                                              10.f,  // chi2MaxPt
                                                     // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
                                              30.f,  // chi2Scale
                                                     // regional cuts for triplets
                                              {
                                                  0.3f,  //tripletMaxTip
                                                  0.5f,  // tripletMinPt
                                                  12.f   // tripletMaxZip
                                              },
                                              // regional cuts for quadruplets
                                              {
                                                  0.5f,  // quadrupletMaxTip
                                                  0.3f,  // quadrupletMinPt
                                                  12.f   // quadrupletMaxZip
                                              }};
  }
}  // namespace

namespace edm {
  class Event;
  class EventSetup;
  class ProductRegistry;
}  // namespace edm

class CAHitNtupletGeneratorOnGPU {
public:
  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DSYCL;
  using hindex_type = TrackingRecHit2DSOAView::hindex_type;

  using Quality = pixelTrack::Quality;
  using OutputSoA = pixelTrack::TrackSoA;
  using HitContainer = pixelTrack::HitContainer;
  using Tuple = HitContainer;

  using QualityCuts = cAHitNtupletGenerator::QualityCuts;
  using Params = cAHitNtupletGenerator::Params;
  using Counters = cAHitNtupletGenerator::Counters;

public:
  CAHitNtupletGeneratorOnGPU(edm::ProductRegistry& reg)
    : m_params(true,              // onGPU
               3,                 // minHitsPerNtuplet,
               458752,            // maxNumberOfDoublets
               false,             //useRiemannFit
               true,              // fit5as4,
               true,              //includeJumpingForwardDoublets
               true,              // earlyFishbone
               false,             // lateFishbone
               true,              // idealConditions
               false,             //fillStatistics
               true,              // doClusterCut
               true,              // doZ0Cut
               true,              // doPtCut
               0.899999976158,    // ptmin
               0.00200000009499,  // CAThetaCutBarrel
               0.00300000002608,  // CAThetaCutForward
               0.0328407224959,   // hardCurvCut
               0.15000000596,     // dcaCutInnerTriplet
               0.25,              // dcaCutOuterTriplet
               makeQualityCuts()) {
  #ifdef DUMP_GPU_TK_TUPLES
  printf("TK: %s %s % %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
         "tid",
         "qual",
         "nh",
         "charge",
         "pt",
         "eta",
         "phi",
         "tip",
         "zip",
         "chi2",
         "h1",
         "h2",
         "h3",
         "h4",
         "h5");
  #endif
  };

  ~CAHitNtupletGeneratorOnGPU() = default;

  PixelTrackHeterogeneous makeTuplesAsync(TrackingRecHit2DGPU const& hits_d, float bfield, sycl::queue stream) const{
    PixelTrackHeterogeneous tracks(cms::sycltools::make_device_unique<pixelTrack::TrackSoA>(stream));

  auto* soa = tracks.get();

  CAHitNtupletGeneratorKernelsGPU kernels(m_params);
  //kernels.counters_ = m_counters;

  //kernels.allocateOnGPU(stream);

  kernels.buildDoublets(hits_d, stream);
  kernels.launchKernels(hits_d, soa, stream);
  kernels.fillHitDetIndices(hits_d.view(), soa, stream);  // in principle needed only if Hits not "available"

  HelixFitOnGPU fitter(bfield, m_params.fit5as4_);
  fitter.allocateOnGPU(&(soa->hitIndices), kernels.tupleMultiplicity(), soa);
  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernels(hits_d.view(), hits_d.nHits(), CAConstants::maxNumberOfQuadruplets(), stream);
  } else {
    fitter.launchBrokenLineKernels(hits_d.view(), hits_d.nHits(), CAConstants::maxNumberOfQuadruplets(), stream);
  }
  kernels.classifyTuples(hits_d, soa, stream);

  return tracks;
  };

private:
  void buildDoublets(HitsOnCPU const& hh, sycl::queue stream) const;

  void hitNtuplets(HitsOnCPU const& hh, const edm::EventSetup& es, bool useRiemannFit, sycl::queue cudaStream);

  void launchKernels(HitsOnCPU const& hh, bool useRiemannFit, sycl::queue cudaStream) const;

  Params m_params;
  
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h

