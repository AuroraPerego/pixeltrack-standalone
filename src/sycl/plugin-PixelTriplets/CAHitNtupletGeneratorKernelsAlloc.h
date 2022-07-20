#include "CAHitNtupletGeneratorKernels.h"

//#include "SYCLCore/syclCheck.h"

template <>
void CAHitNtupletGeneratorKernelsGPU::allocateOnGPU(sycl::queue stream) {
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  device_theCellNeighbors_ = Traits::template make_unique<CAConstants::CellNeighborsVector>(stream);
  device_theCellTracks_ = Traits::template make_unique<CAConstants::CellTracksVector>(stream);

  device_hitToTuple_ = Traits::template make_unique<HitToTuple>(stream);

  device_tupleMultiplicity_ = Traits::template make_unique<TupleMultiplicity>(stream);

  device_storage_ = Traits::template make_unique<cms::sycltools::AtomicPairCounter::c_type[]>(3, stream);

  device_hitTuple_apc_ = (cms::sycltools::AtomicPairCounter*)device_storage_.get();
  device_hitToTuple_apc_ = (cms::sycltools::AtomicPairCounter*)device_storage_.get() + 1;
  device_nCells_ = (uint32_t*)(device_storage_.get() + 2);

  stream.memset(device_nCells_, 0, sizeof(uint32_t));

  cms::sycltools::launchZero(device_tupleMultiplicity_.get(), stream);
  cms::sycltools::launchZero(device_hitToTuple_.get(), stream);  // we may wish to keep it in the edm...
}
