#include <utility>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/Product.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/alpakaMemoryHelper.h"
#include "AlpakaDataFormats/ZVertexAlpaka.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"
#include "Framework/RunningAverage.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelVertexSoAFromAlpaka : public edm::EDProducerExternalWork {
  public:
    explicit PixelVertexSoAFromAlpaka(edm::ProductRegistry& reg);
    ~PixelVertexSoAFromAlpaka() override = default;

  private:
    void acquire(edm::Event const& iEvent,
                 edm::EventSetup const& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

    edm::EDGetTokenT<::cms::alpakatools::Product<Queue, ZVertexAlpaka>> tokenDevice_;
    edm::EDPutTokenT<ZVertexHost> tokenHost_;

    ZVertexHost soa_;
  };

  PixelVertexSoAFromAlpaka::PixelVertexSoAFromAlpaka(edm::ProductRegistry& reg)
      : tokenDevice_(reg.consumes<::cms::alpakatools::Product<Queue, ZVertexAlpaka>>()),
        tokenHost_(reg.produces<ZVertexHost>()),
        soa_(::cms::alpakatools::allocHostBuf<ZVertexSoA>(1u)) {}

  void PixelVertexSoAFromAlpaka::acquire(edm::Event const& iEvent,
                                         edm::EventSetup const& iSetup,
                                         edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    auto const& inputDataWrapped = iEvent.get(tokenDevice_);
    ::cms::alpakatools::ScopedContextAcquire<Queue> ctx{inputDataWrapped, std::move(waitingTaskHolder)};
    auto const& inputData = ctx.get(inputDataWrapped);

    soa_ = ::cms::alpakatools::allocHostBuf<ZVertexSoA>(1u);
    alpaka::memcpy(ctx.stream(), soa_, inputData, 1u);
  }

  void PixelVertexSoAFromAlpaka::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
    // No copies....
    iEvent.emplace(tokenHost_, std::move(soa_));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(PixelVertexSoAFromAlpaka);
