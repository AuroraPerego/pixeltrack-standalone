#include "CAHitNtupletGeneratorKernelsImpl.h"
 
template <>
void CAHitNtupletGeneratorKernels::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d, sycl::queue stream) {
  auto blockSize = 128;
  auto numberOfBlocks = (HitContainer::capacity() + blockSize - 1) / blockSize;
  
  stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernel_fillHitDetIndices(&tracks_d->hitIndices, hv, &tracks_d->detIndices, item);
      });
    });
  
  //cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
  stream.wait();
  //cudaCheck(cudaGetLastError());
#endif
}

template <>
void CAHitNtupletGeneratorKernels::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, sycl::queue stream) {
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
      auto m_params_kernel = m_params;
      cgh.parallel_for(
          sycl::nd_range<3>(blks * thrs, thrs),
          [=](sycl::nd_item<3> item){ 
              kernel_connect(device_hitTuple_apc_,
                             device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
                             hh.view(),
                             device_theCells_.get(),
                             device_nCells_,
                             device_theCellNeighbors_.get(),
                             device_isOuterHitOfCell_.get(),
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
      cgh.parallel_for(
          sycl::nd_range<3>(blks * thrs, thrs),
          [=](sycl::nd_item<3> item){ 
              gpuPixelDoublets::fishbone(hh.view(), 
                                         device_theCells_.get(), 
                                         device_nCells_, 
                                         device_isOuterHitOfCell_.get(), 
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
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_find_ntuplets(hh.view(),
                                   device_theCells_.get(),
                                   device_nCells_,
                                   device_theCellTracks_.get(),
                                   tuples_d,
                                   device_hitTuple_apc_,
                                   quality_d,
                                   m_params_kernel.minHitsPerNtuplet_,
                                   item);
   
      });
    });
  
   //cudaCheck(cudaGetLastError());

  if (m_params.doStats_)
    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_mark_used(hh.view(), device_theCells_.get(), device_nCells_, item);
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
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              cms::sycltools::finalizeBulk(device_hitTuple_apc_, tuples_d);
      });
    });

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_earlyDuplicateRemover(device_theCells_.get(), device_nCells_, tuples_d, quality_d, item);  
      });
    });
  
  //cudaCheck(cudaGetLastError());

  blockSize = 128;
  numberOfBlocks = (3 * CAConstants::maxTuples() / 4 + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_countMultiplicity(tuples_d, quality_d, device_tupleMultiplicity_.get(), item); 
      });
    });
  
  cms::sycltools::launchFinalize(device_tupleMultiplicity_.get(), stream);
  stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_fillMultiplicity(tuples_d, quality_d, device_tupleMultiplicity_.get(), item);  
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
      cgh.parallel_for(
          sycl::nd_range<3>(blks * thrs, thrs),
          [=](sycl::nd_item<3> item){ 
              gpuPixelDoublets::fishbone(hh.view(), 
                                         device_theCells_.get(), 
                                         device_nCells_, 
                                         device_isOuterHitOfCell_.get(), 
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
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_checkOverflows(tuples_d,
                                    device_tupleMultiplicity_.get(),
                                    device_hitTuple_apc_,
                                    device_theCells_.get(),
                                    device_nCells_,
                                    device_theCellNeighbors_.get(),
                                    device_theCellTracks_.get(),
                                    device_isOuterHitOfCell_.get(),
                                    nhits,
                                    m_params.maxNumberOfDoublets_,
                                    counters_,
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
}

template <>
void CAHitNtupletGeneratorKernels::buildDoublets(HitsOnCPU const &hh, sycl::queue stream) {
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
      cgh.parallel_for(
          sycl::nd_range<3>(blocks * sycl::range<3>(1, 1, threadsPerBlock), sycl::range<3>(1, 1, threadsPerBlock)),
          [=](sycl::nd_item<3> item){ 
              gpuPixelDoublets::initDoublets(device_isOuterHitOfCell_.get(),
                                             nhits,
                                             device_theCellNeighbors_.get(),
                                             device_theCellNeighborsContainer_,
                                             device_theCellTracks_.get(),
                                             device_theCellTracksContainer_,
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
    auto m_params_kernel = m_params
      cgh.parallel_for(
          sycl::nd_range<3>(blks * thrs, thrs),
          [=](sycl::nd_item<3> item){ 
              gpuPixelDoublets::getDoubletsFromHisto(device_theCells_.get(),
                                                     device_nCells_,
                                                     device_theCellNeighbors_.get(),
                                                     device_theCellTracks_.get(),
                                                     hh.view(),
                                                     device_isOuterHitOfCell_.get(),
                                                     nActualPairs,
                                                     m_params_kernel.idealConditions_,
                                                     m_params_kernel.doClusterCut_,
                                                     m_params_kernel.doZ0Cut_,
                                                     m_params_kernel.doPtCut_,
                                                     m_params_kernel.maxNumberOfDoublets_,
                                                     item);   
      });
    });
  
  //cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  stream.wait();
  //cudaCheck(cudaGetLastError());
#endif
}

template <>
void CAHitNtupletGeneratorKernels::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d, sycl::queue stream) {
  // these are pointer on GPU!
  auto const *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  auto blockSize = 64;

  // classify tracks based on kinematics
  auto numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      auto m_params_kernel = m_params;
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_classifyTracks(tuples_d, 
                                    tracks_d, 
                                    m_params_kernel.cuts_, 
                                    quality_d,
                                    item);              
      });
    });
  //cudaCheck(cudaGetLastError());

  if (m_params.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_fishboneCleaner(device_theCells_.get(), device_nCells_, quality_d, item); 
      });
    });
    
    //cudaCheck(cudaGetLastError());
  }

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_fastDuplicateRemover(device_theCells_.get(), device_nCells_, tuples_d, tracks_d, item);
      });
    });
  
  //cudaCheck(cudaGetLastError());

  if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
    // fill hit->track "map"
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_countHitInTracks(tuples_d, quality_d, device_hitToTuple_.get(), item); 
      });
    });
    
    //cudaCheck(cudaGetLastError());
    cms::sycltools::launchFinalize(device_hitToTuple_.get(), stream);
    //cudaCheck(cudaGetLastError());
    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_fillHitInTracks(tuples_d, quality_d, device_hitToTuple_.get(), item);
      });
    });
    //cudaCheck(cudaGetLastError());
  }
  if (m_params.minHitsPerNtuplet_ < 4) {
    // remove duplicates (tracks that share a hit)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_tripletCleaner(hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get(), item);
      });
    });
    //cudaCheck(cudaGetLastError());
  }

  if (m_params.doStats_) {
    // counters (add flag???)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_doStatsForHitInTracks(device_hitToTuple_.get(), counters_, item);
      });
    });
    //cudaCheck(cudaGetLastError());
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_doStatsForTracks(tuples_d, quality_d, counters_, item);
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
      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
              kernel_print_found_ntuplets(hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get(), 100, iev, item);
      });
    });
#endif
}

template <>
void CAHitNtupletGeneratorKernels::printCounters(Counters const *counters) {
  stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item){ 
              kernel_printCounters(counters); 
      });
    });
  
}
