#include "CAHitNtupletGeneratorKernelsImpl.h"

#define NTUPLE_DEBUG TRUE

template <>
void CAHitNtupletGeneratorKernelsGPU::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d, sycl::queue stream) {
  auto blockSize = 128;
  auto numberOfBlocks = (HitContainer::capacity() + blockSize - 1) / blockSize;
  
  stream.submit([&](sycl::handler &cgh) {
    auto hitIndices_kernel = &tracks_d->hitIndices;
    auto hv_kernel         = hv;
    auto detIndices_kernel = &tracks_d->detIndices;
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
                kernel_fillHitDetIndices(hitIndices_kernel, hv_kernel, detIndices_kernel, item);
      });
    });
  
#ifdef GPU_DEBUG
  stream.wait();
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, sycl::queue stream) {
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
  }

  blockSize = 64;
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  std::cout << m_params.maxNumberOfDoublets_ << " Entering kernel_find_ntuplets\n";
  stream.submit([&](sycl::handler &cgh) {
      auto m_params_kernel = m_params.minHitsPerNtuplet_;
      auto hh_kernel                      = hh.view();
      auto device_theCells_kernel         = device_theCells_.get();
      auto device_nCells_kernel           = device_nCells_;
      std::cout << device_nCells_kernel << " device_nCells_kernel\n";
      auto device_theCellTracks_kernel    = device_theCellTracks_.get();
      auto tuples_d_kernel                = tuples_d;
      auto device_hitTuple_apc_kernel     = device_hitTuple_apc_;
      auto quality_d_kernel               = quality_d;
      sycl::stream out(1024, 768, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_find_ntuplets(hh_kernel,
                                   device_theCells_kernel,
                                   device_nCells_kernel,
                                   device_theCellTracks_kernel,
                                   tuples_d_kernel,
                                   device_hitTuple_apc_kernel,
                                   quality_d_kernel,
                                   m_params_kernel,
                                   item,
                                   out);
   
      });
    });
    std::cout << "kernel_find_ntuplets done\n";

  if (m_params.doStats_)
    stream.submit([&](sycl::handler &cgh) {
      auto hh_kernel                      = hh.view();
      auto device_theCells_kernel         = device_theCells_.get();
      auto device_nCells_kernel           = device_nCells_;
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_mark_used(hh_kernel, device_theCells_kernel, device_nCells_kernel, item);
      });
    });

#ifdef GPU_DEBUG
  stream.wait();
#endif

  blockSize = 128;
  numberOfBlocks = (HitContainer::totbins() + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel                = tuples_d;
      auto device_hitTuple_apc_kernel     = device_hitTuple_apc_;
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
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
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_earlyDuplicateRemover(device_theCells_kernel, device_nCells_kernel, tuples_d_kernel, quality_d_kernel, item);  
      });
    });
  

  blockSize = 128;
  numberOfBlocks = (3 * CAConstants::maxTuples() / 4 + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel                 = tuples_d;
      auto device_tupleMultiplicity_kernel = device_tupleMultiplicity_.get();
      auto quality_d_kernel                = quality_d;
      sycl::stream out(1024, 768, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_countMultiplicity(tuples_d_kernel, quality_d_kernel, device_tupleMultiplicity_kernel, item, out); 
      });
    });
  
  cms::sycltools::launchFinalize(device_tupleMultiplicity_.get(), stream);
  stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel                 = tuples_d;
      auto device_tupleMultiplicity_kernel = device_tupleMultiplicity_.get();
      auto quality_d_kernel                = quality_d;
      sycl::stream out(1024, 768, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_fillMultiplicity(tuples_d_kernel, quality_d_kernel, device_tupleMultiplicity_kernel, item, out);  
      });
    });
  

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
      sycl::stream out(1024, 768, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
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
                                    item,
                                    out);  
      });
    });

  }
#ifdef GPU_DEBUG
  stream.wait();
#endif

  // free space asap
  // device_isOuterHitOfCell_.reset();*/
}

template <>
void CAHitNtupletGeneratorKernelsGPU::buildDoublets(HitsOnCPU const &hh, sycl::queue stream) {
  uint nhits = 100; //hh.nHits();
#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
  stream.wait();
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
      sycl::stream out(1024, 768, cgh);
      std::cout << "Entering initDoublets\n";
      cgh.parallel_for(
          sycl::nd_range<1>( blocks * threadsPerBlock, threadsPerBlock),
          [=](sycl::nd_item<1> item){
              gpuPixelDoublets::initDoublets(device_isOuterHitOfCell_kernel,
                                             nhits,
                                             device_theCellNeighbors_kernel,
                                             device_theCellNeighborsContainer_kernel,
                                             device_theCellTracks_kernel,
                                             device_theCellTracksContainer_kernel,
                                             item,
                                             out);
 
      });
    });
  }
  std::cout << "initDoublets done\n";
  device_theCells_ = cms::sycltools::make_device_unique<GPUCACell[]>(m_params.maxNumberOfDoublets_, stream);

#ifdef GPU_DEBUG
  stream.wait();
#endif
 std::cout << nhits << "\n";
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
      sycl::stream out(1024, 768, cgh);
      const int nPairsMax = CAConstants::maxNumberOfLayerPairs();
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              innerLayerCumulativeSize_acc(sycl::range<1>(sizeof(uint32_t)) * nPairsMax, cgh); //FIXME_
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ntot_acc(sycl::range<1>(sizeof(uint32_t)), cgh);  //FIXME_ no idea if the arguments passed to the accessors are correct                   
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
                                                     (uint32_t *)ntot_acc.get_pointer(),
                                                     out);   
      });
    });
   std::cout << "getdoubletsfromhisto done\n";

#ifdef GPU_DEBUG
  stream.wait();
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d, sycl::queue stream) {
  // these are pointer on GPU!
  auto const *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  auto blockSize = 64;

  // classify tracks based on kinematics
  auto numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
  /*stream.submit([&](sycl::handler &cgh) {
      auto m_params_kernel = m_params;
      auto tuples_d_kernel = tuples_d;
      auto tracks_d_kernel = tracks_d;
      auto quality_d_kernel = quality_d;
      sycl::stream out(1024, 768, cgh);
      cgh.parallel_for(      
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_classifyTracks(tuples_d_kernel, 
                                    tracks_d_kernel, 
                                    m_params_kernel.cuts_, 
                                    quality_d_kernel,
                                    item,
                                    out);              
      });
    });
*/
  if (m_params.lateFishbone_) {
 /*   // apply fishbone cleaning to good tracks
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      auto device_theCells_kernel = device_theCells_.get();
      auto device_nCells_kernel = device_nCells_;
      auto quality_d_kernel = quality_d;
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_fishboneCleaner(device_theCells_kernel, device_nCells_kernel, quality_d_kernel, item); 
      });
    });*/
    
  }

  // remove duplicates (tracks that share a doublet)
  /*numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  stream.submit([&](sycl::handler &cgh) {
      auto device_theCells_kernel = device_theCells_.get();
      auto device_nCells_kernel = device_nCells_;
      auto tuples_d_kernel = tuples_d;
      auto tracks_d_kernel = tracks_d;
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_fastDuplicateRemover(device_theCells_kernel, device_nCells_kernel, tuples_d_kernel, tracks_d_kernel, item);
      });
    });
  

  if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
    // fill hit->track "map"
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      auto device_hitToTuple_kernel = device_hitToTuple_.get();
      auto tuples_d_kernel = tuples_d;
      auto quality_d_kernel = quality_d;
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_countHitInTracks(tuples_d_kernel, quality_d_kernel, device_hitToTuple_kernel, item); 
      });
    });
    
    cms::sycltools::launchFinalize(device_hitToTuple_.get(), stream);
    stream.submit([&](sycl::handler &cgh) {
      auto device_hitToTuple_kernel = device_hitToTuple_.get();
      auto tuples_d_kernel = tuples_d;
      auto quality_d_kernel = quality_d;
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_fillHitInTracks(tuples_d_kernel, quality_d_kernel, device_hitToTuple_kernel, item);
      });
    });
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
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_tripletCleaner(hh_kernel, tuples_d_kernel, tracks_d_kernel, quality_d_kernel, device_hitToTuple_kernel, item);
      });
    });
  }

  if (m_params.doStats_) {
    // counters (add flag???)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      auto device_hitToTuple_kernel = device_hitToTuple_.get();
      auto counters_kernel = counters_;
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_doStatsForHitInTracks(device_hitToTuple_kernel, counters_kernel, item);
      });
    });
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel = tuples_d;
      auto quality_d_kernel = quality_d;
      auto counters_kernel = counters_;
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              kernel_doStatsForTracks(tuples_d_kernel, quality_d_kernel, counters_kernel, item);
      });
    });
  }*/
#ifdef GPU_DEBUG
  stream.wait();
#endif

#ifdef DUMP_GPU_TK_TUPLES
  //static std::atomic<int> iev(0);
  //++iev;
  blockSize = 32;
  /*stream.submit([&](sycl::handler &cgh) {
      auto hh_kernel = hh.view();
      auto tuples_d_kernel = tuples_d;
      auto tracks_d_kernel = tracks_d;
      auto quality_d_kernel = quality_d;
      sycl::stream out(1024, 768, cgh);
      auto device_hitToTuple_kernel = device_hitToTuple_.get();
      cgh.parallel_for(
          sycl::nd_range<1>(blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
              //kernel_print_found_ntuplets(hh_kernel, tuples_d_kernel, tracks_d_kernel, quality_d_kernel, device_hitToTuple_kernel, 100, iev, item, out);
      });
    });*/
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::printCounters(Counters const *counters, sycl::queue stream) {
  /*stream.submit([&](sycl::handler &cgh) {
      auto counters_kernel = counters;
      sycl::stream out(1024, 768, cgh);
      cgh.single_task([=](){ 
              kernel_printCounters(counters_kernel, out); 
      });
    });*/
  
}