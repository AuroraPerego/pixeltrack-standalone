#include <CL/sycl.hpp>

#include "SYCLCore/Product.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"
#include "SYCLCore/ScopedContext.h"

#include "gpuVertexFinder.h"

class PixelVertexProducerSYCL : public edm::EDProducer {
public:
  explicit PixelVertexProducerSYCL(edm::ProductRegistry& reg);
  ~PixelVertexProducerSYCL() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  bool m_OnGPU;

  edm::EDGetTokenT<cms::sycltools::Product<PixelTrackHeterogeneous>> tokenGPUTrack_;
  edm::EDPutTokenT<ZVertexSYCLProduct> tokenGPUVertex_;
  edm::EDGetTokenT<PixelTrackHeterogeneous> tokenCPUTrack_;
  edm::EDPutTokenT<ZVertexHeterogeneous> tokenCPUVertex_;

  const gpuVertexFinder::Producer m_gpuAlgo;

  // Tracking cuts before sending tracks to vertex algo
  const float m_ptMin;
};

PixelVertexProducerSYCL::PixelVertexProducerSYCL(edm::ProductRegistry& reg)
    : m_OnGPU(true), //FIXME_ also this one and all the if m_OnGPU shoulde be removed
      m_gpuAlgo(true,   // oneKernel
                true,   // useDensity
                false,  // useDBSCAN
                false,  // useIterative
                2,      // minT
                0.07,   // eps
                0.01,   // errmax
                9       // chi2max
                ),
      m_ptMin(0.5)  // 0.5 GeV
{
  if (m_OnGPU) {
    tokenGPUTrack_ = reg.consumes<cms::sycltools::Product<PixelTrackHeterogeneous>>();
    tokenGPUVertex_ = reg.produces<ZVertexSYCLProduct>();
  } else {
    tokenCPUTrack_ = reg.consumes<PixelTrackHeterogeneous>();
    tokenCPUVertex_ = reg.produces<ZVertexHeterogeneous>();
  }
}

void PixelVertexProducerSYCL::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (m_OnGPU) {
    auto const& ptracks = iEvent.get(tokenGPUTrack_);

    cms::sycltools::ScopedContextProduce ctx{ptracks};
    auto const* tracks = ctx.get(ptracks).get();

    assert(tracks);

    ctx.emplace(iEvent, tokenGPUVertex_, m_gpuAlgo.makeAsync(ctx.stream(), tracks, m_ptMin));

  } else {
    auto const* tracks = iEvent.get(tokenCPUTrack_).get();
    assert(tracks);

    /*
    auto const & tsoa = *tracks;
    auto maxTracks = tsoa.stride();
    std::cout << "size of SoA " << sizeof(tsoa) << " stride " << maxTracks << std::endl;

    int32_t nt = 0;
    for (int32_t it = 0; it < maxTracks; ++it) {
      auto nHits = tsoa.nHits(it);
      assert(nHits==int(tsoa.hitIndices.size(it)));
      if (nHits == 0) break;  // this is a guard: maybe we need to move to nTracks...
      nt++;
    }
    std::cout << "found " << nt << " tracks in cpu SoA for Vertexing at " << tracks << std::endl;
    */

    iEvent.emplace(tokenCPUVertex_, m_gpuAlgo.make(tracks, m_ptMin));
  }
}

DEFINE_FWK_MODULE(PixelVertexProducerSYCL);
