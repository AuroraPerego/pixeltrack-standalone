#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h

#include <CL/sycl.hpp>

#include "SYCLCore/SimpleVector.h"
#include "SYCLDataFormats/PixelTrackHeterogeneous.h"
#include "SYCLDataFormats/TrackingRecHit2DSYCL.h"

#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "HelixFitOnGPU.h"

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
  CAHitNtupletGeneratorOnGPU(edm::ProductRegistry& reg);

  ~CAHitNtupletGeneratorOnGPU() = default;

  PixelTrackHeterogeneous makeTuplesAsync(TrackingRecHit2DGPU const& hits_d, float bfield, sycl::queue stream) const;

private:
  void buildDoublets(HitsOnCPU const& hh, sycl::queue stream) const;

  void hitNtuplets(HitsOnCPU const& hh, const edm::EventSetup& es, bool useRiemannFit, sycl::queue cudaStream);

  void launchKernels(HitsOnCPU const& hh, bool useRiemannFit, sycl::queue cudaStream) const;

  Params m_params;
  
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
