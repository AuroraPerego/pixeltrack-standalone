//
// Original Author: Felice Pantaleo, CERN
//

// #define NTUPLE_DEBUG

#include <CL/sycl.hpp>
#include <cmath>
#include <cstdint>

#include "SYCLCore/AtomicPairCounter.h"
#include "SYCLCore/printf.h" 
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"
#include "CondFormats/pixelCPEforGPU.h"

#include "CAConstants.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "gpuFishbone.h"
#include "gpuPixelDoublets.h"

using sycl::abs;

using HitsOnGPU = TrackingRecHit2DSOAView;
using HitsOnCPU = TrackingRecHit2DSYCL;

using HitToTuple = CAConstants::HitToTuple;
using TupleMultiplicity = CAConstants::TupleMultiplicity;

using Quality = pixelTrack::Quality;
using TkSoA = pixelTrack::TrackSoA;
using HitContainer = pixelTrack::HitContainer;

void kernel_checkOverflows(HitContainer const *foundNtuplets,
                           CAConstants::TupleMultiplicity *tupleMultiplicity,
                           cms::sycltools::AtomicPairCounter *apc,
                           GPUCACell const *__restrict__ cells,
                           uint32_t const *__restrict__ nCells,
                           gpuPixelDoublets::CellNeighborsVector const *cellNeighbors,
                           gpuPixelDoublets::CellTracksVector const *cellTracks,
                           GPUCACell::OuterHitOfCell const *__restrict__ isOuterHitOfCell,
                           uint32_t nHits,
                           uint32_t maxNumberOfDoublets,
                           CAHitNtupletGeneratorKernelsGPU::Counters *counters,
                           sycl::nd_item<1> item) {
  auto first = item.get_local_id(0) + item.get_group(0) * item.get_local_range().get(0);

  auto &c = *counters;
  // counters once per event
  if (0 == first) {
    cms::sycltools::atomic_fetch_add<unsigned long long>(&c.nEvents, (unsigned long long)1);
    cms::sycltools::atomic_fetch_add<unsigned long long>(&c.nHits, (unsigned long long)nHits);
    cms::sycltools::atomic_fetch_add<unsigned long long>(&c.nCells, (unsigned long long)*nCells);
    cms::sycltools::atomic_fetch_add<unsigned long long>(&c.nTuples, (unsigned long long)apc->get().m);
    cms::sycltools::atomic_fetch_add<unsigned long long>(&c.nFitTracks, (unsigned long long)tupleMultiplicity->size());
  }

#ifdef NTUPLE_DEBUG
  if (0 == first) {
    printf("number of found cells %d, found tuples %d with total hits %d out of %d\n", *nCells, apc->get().m, apc->get().n, nHits);
    if (apc->get().m < CAConstants::maxNumberOfQuadruplets()) {
      assert(foundNtuplets->size(apc->get().m) == 0);
      assert(foundNtuplets->size() == apc->get().n);
    }
  }

  for (int idx = first, nt = foundNtuplets->nbins(); idx < nt; 
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    if (foundNtuplets->size(idx) > 5){
      printf("ERROR %d, %d\n", idx, foundNtuplets->size(idx));
    }
    assert(foundNtuplets->size(idx) < 6);
    for (auto ih = foundNtuplets->begin(idx); ih != foundNtuplets->end(idx); ++ih)
      assert(*ih < nHits);
  }
#endif

  if (0 == first) {
    if (apc->get().m >= CAConstants::maxNumberOfQuadruplets())
      printf("Tuples overflow\n");
    if (*nCells >= maxNumberOfDoublets)
      printf("Cells overflow\n");
    if (cellNeighbors && cellNeighbors->full())
      printf("cellNeighbors overflow\n");
    if (cellTracks && cellTracks->full())
      printf("cellTracks overflow\n");
  }

  for (int idx = first, nt = (*nCells); idx < nt;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto const &thisCell = cells[idx];
    if (thisCell.outerNeighbors().full()) //++tooManyNeighbors[thisCell.theLayerPairId];
      printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.theLayerPairId);
    if (thisCell.tracks().full())  //++tooManyTracks[thisCell.theLayerPairId];
      printf("Tracks overflow %d in %d\n", idx, thisCell.theLayerPairId);
    if (thisCell.theDoubletId < 0)
      cms::sycltools::atomic_fetch_add<unsigned long long>((unsigned long long *)(&c.nKilledCells), (unsigned long long)1);
    if (0 == thisCell.theUsed)
      cms::sycltools::atomic_fetch_add<unsigned long long>((unsigned long long *)(&c.nEmptyCells), (unsigned long long)1);
    if (thisCell.tracks().empty())
      cms::sycltools::atomic_fetch_add<unsigned long long>((unsigned long long *)(&c.nZeroTrackCells), (unsigned long long)1);
  }

  for (int idx = first, nt = nHits; idx < nt; 
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    if (isOuterHitOfCell[idx].full())  // ++tooManyOuterHitOfCell;
      printf("OuterHitOfCell overflow %d\n", idx);
  }
}

void kernel_fishboneCleaner(GPUCACell const *cells, uint32_t const *__restrict__ nCells, Quality *quality,
                            sycl::nd_item<1> item) {
  constexpr auto bad = trackQuality::bad;

  auto first = item.get_local_id(0) + item.get_group(0) * item.get_local_range().get(0);
  for (int idx = first, nt = (*nCells); idx < nt;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto const &thisCell = cells[idx];
    if (thisCell.theDoubletId >= 0)
      continue;

    for (auto it : thisCell.tracks())
      quality[it] = bad;
  }
}

void kernel_earlyDuplicateRemover(GPUCACell const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  HitContainer *foundNtuplets,
                                  Quality *quality,
                                  sycl::nd_item<1> item) {
  // constexpr auto bad = trackQuality::bad;
  constexpr auto dup = trackQuality::dup;
  // constexpr auto loose = trackQuality::loose;

  assert(nCells);
  auto first = item.get_local_id(0) + item.get_group(0) * item.get_local_range().get(0);
  for (int idx = first, nt = (*nCells); idx < nt;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto const &thisCell = cells[idx];

    if (thisCell.tracks().size() < 2)
      continue;
    //if (0==thisCell.theUsed) continue;
    // if (thisCell.theDoubletId < 0) continue;

    uint32_t maxNh = 0;

    // find maxNh
    for (auto it : thisCell.tracks()) {
      auto nh = foundNtuplets->size(it);
      maxNh = sycl::max(nh, maxNh);
    }

    for (auto it : thisCell.tracks()) {
      if (foundNtuplets->size(it) != maxNh)
        quality[it] = dup;  //no race:  simple assignment of the same constant
    }
  }
}

void kernel_fastDuplicateRemover(GPUCACell const *__restrict__ cells,
                                 uint32_t const *__restrict__ nCells,
                                 HitContainer const *__restrict__ foundNtuplets,
                                 TkSoA *__restrict__ tracks,
                                 sycl::nd_item<1> item) {
  constexpr auto bad = trackQuality::bad;
  constexpr auto dup = trackQuality::dup;
  constexpr auto loose = trackQuality::loose;

  assert(nCells);

  auto first = item.get_local_id(0) + item.get_group(0) * item.get_local_range().get(0);
  for (int idx = first, nt = (*nCells); idx < nt;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto const &thisCell = cells[idx];
    if (thisCell.tracks().size() < 2)
      continue;
    // if (thisCell.theDoubletId < 0) continue;

    float mc = 10000.f;
    uint16_t im = 60000;

    auto score = [&](auto it) {
      return abs(tracks->tip(it));  // tip
      // return tracks->chi2(it);  //chi2
    };

    // find min socre
    for (auto it : thisCell.tracks()) {
      if (tracks->quality(it) == loose && score(it) < mc) {
        mc = score(it);
        im = it;
      }
    }
    // mark all other duplicates
    for (auto it : thisCell.tracks()) {
      if (tracks->quality(it) != bad && it != im)
        tracks->quality(it) = dup;  //no race:  simple assignment of the same constant
    }
  }
}

void kernel_connect(cms::sycltools::AtomicPairCounter *apc1,
                    cms::sycltools::AtomicPairCounter *apc2,  // just to zero them,
                    GPUCACell::Hits const *__restrict__ hhp,
                    GPUCACell *cells,
                    uint32_t const *__restrict__ nCells,
                    gpuPixelDoublets::CellNeighborsVector *cellNeighbors,
                    GPUCACell::OuterHitOfCell const *__restrict__ isOuterHitOfCell,
                    float hardCurvCut,
                    float ptmin,
                    float CAThetaCutBarrel,
                    float CAThetaCutForward,
                    float dcaCutInnerTriplet,
                    float dcaCutOuterTriplet,
                    sycl::nd_item<3> item) {
  auto const &hh = *hhp;

  auto firstCellIndex = item.get_local_id(1) + item.get_group(1) * item.get_local_range().get(1);
  auto first = item.get_local_id(2);
  auto stride = item.get_local_range().get(2);

  if (0 == (firstCellIndex + first)) {
    (*apc1) = 0;
    (*apc2) = 0;
  }  // ready for next kernel

  for (int idx = firstCellIndex, nt = (*nCells); idx < nt;
       idx += item.get_group_range(1) * item.get_local_range().get(1)) {
    auto cellIndex = idx;
    auto &thisCell = cells[idx];
    //if (thisCell.theDoubletId < 0 || thisCell.theUsed>1)
    //  continue;
    auto innerHitId = thisCell.get_inner_hit_id();
    int numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
    auto vi = isOuterHitOfCell[innerHitId].data();

    constexpr uint32_t last_bpix1_detIndex = 96;
    constexpr uint32_t last_barrel_detIndex = 1184;
    auto ri = thisCell.get_inner_r(hh);
    auto zi = thisCell.get_inner_z(hh);

    auto ro = thisCell.get_outer_r(hh);
    auto zo = thisCell.get_outer_z(hh);
    auto isBarrel = thisCell.get_inner_detIndex(hh) < last_barrel_detIndex;
    for (int j = first; j < numberOfPossibleNeighbors; j += stride) {
      auto otherCell = vi[j]; 
      auto &oc = cells[otherCell];
      // if (cells[otherCell].theDoubletId < 0 ||
      //    cells[otherCell].theUsed>1 )
      //  continue;
      auto r1 = oc.get_inner_r(hh);
      auto z1 = oc.get_inner_z(hh);
      // auto isBarrel = oc.get_outer_detIndex(hh) < last_barrel_detIndex;
      bool aligned = GPUCACell::areAlignedRZ(r1,
                                             z1,
                                             ri,
                                             zi,
                                             ro,
                                             zo,
                                             ptmin,
                                             isBarrel ? CAThetaCutBarrel : CAThetaCutForward);  // 2.f*thetaCut); // FIXME tune cuts
      if (aligned &&
          thisCell.dcaCut(hh,
                          oc,
                          oc.get_inner_detIndex(hh) < last_bpix1_detIndex ? dcaCutInnerTriplet : dcaCutOuterTriplet, //same, 0.15 or 0.25
                          hardCurvCut)) {  // FIXME tune cuts -> 0.0328407224959
        oc.addOuterNeighbor(cellIndex, *cellNeighbors); 
        thisCell.theUsed |= 1;
        oc.theUsed |= 1;
      }
    }  // loop on inner cells
  }    // loop on outer cells
}

void kernel_find_ntuplets(GPUCACell::Hits const *__restrict__ hhp,
                          GPUCACell *__restrict__ cells,
                          uint32_t const *nCells,
                          gpuPixelDoublets::CellTracksVector *cellTracks,
                          HitContainer *foundNtuplets,
                          cms::sycltools::AtomicPairCounter *apc,
                          Quality *__restrict__ quality,
                          unsigned int minHitsPerNtuplet,
                          sycl::nd_item<1> item) {
  // recursive: not obvious to widen
  auto first = item.get_local_id(0) + item.get_group(0) * item.get_local_range().get(0);
  for (int idx = first, nt = (*nCells); idx < nt;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto const &hh = *hhp;
    auto const &thisCell = cells[idx];
    if (thisCell.theDoubletId < 0)
      continue;  // cut by earlyFishbone

    auto pid = thisCell.theLayerPairId;
    auto doit = minHitsPerNtuplet > 3 ? pid < 3 : pid < 8 || pid > 12;
    if (doit) {
      GPUCACell::TmpTuple stack;
      stack.reset();
      thisCell.find_ntuplets<6>(
          hh, cells, *cellTracks, *foundNtuplets, *apc, quality, stack, minHitsPerNtuplet, pid < 3); 
      assert(stack.empty());
    }
  }
}

void kernel_mark_used(GPUCACell::Hits const *__restrict__ hhp,
                      GPUCACell *__restrict__ cells,
                      uint32_t const *nCells,
                      sycl::nd_item<1> item) {
  // auto const &hh = *hhp;
  auto first = item.get_local_id(0) + item.get_group(0) * item.get_local_range().get(0);
  for (int idx = first, nt = (*nCells); idx < nt;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto &thisCell = cells[idx];
    if (!thisCell.tracks().empty())
      thisCell.theUsed |= 2;
  }
}

void kernel_countMultiplicity(HitContainer const *__restrict__ foundNtuplets,
                              Quality const *__restrict__ quality,
                              CAConstants::TupleMultiplicity *tupleMultiplicity,
                              sycl::nd_item<1> item) {
  auto first = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
  for (int it = first, nt = foundNtuplets->nbins(); it < nt;
       it += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto nhits = foundNtuplets->size(it);
    if (nhits < 3)
      continue;
    if (quality[it] == trackQuality::dup)
      continue;
    assert(quality[it] == trackQuality::bad);
    if (nhits > 5)
      printf("wrong mult %d with %d hits\n", it, nhits);
    assert(nhits < 8);
    tupleMultiplicity->countDirect(nhits);
  }
}

void kernel_fillMultiplicity(HitContainer const *__restrict__ foundNtuplets,
                             Quality const *__restrict__ quality,
                             CAConstants::TupleMultiplicity *tupleMultiplicity,
                             sycl::nd_item<1> item) {
  auto first = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
  for (int it = first, nt = foundNtuplets->nbins(); it < nt;
       it += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto nhits = foundNtuplets->size(it);
    if (nhits < 3)
      continue;
    if (quality[it] == trackQuality::dup)
      continue;
    assert(quality[it] == trackQuality::bad);
    if (nhits > 5)
      printf("wrong mult %d with %d hits\n", it, nhits);
    assert(nhits < 8);
    tupleMultiplicity->fillDirect(nhits, it);
  }
}

void kernel_classifyTracks(HitContainer const *__restrict__ tuples,
                           TkSoA const *__restrict__ tracks,
                           CAHitNtupletGeneratorKernelsGPU::QualityCuts cuts,
                           Quality *__restrict__ quality,
                           sycl::nd_item<1> item) {
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int it = first, nt = tuples->nbins(); it < nt;
       it += item.get_group_range(0) * item.get_local_range().get(0)) {
    auto nhits = tuples->size(it);
    if (nhits == 0)
      break;  // guard

    // if duplicate: not even fit
    if (quality[it] == trackQuality::dup)
      continue;

    assert(quality[it] == trackQuality::bad);

    // mark doublets as bad
    if (nhits < 3)
      continue;

    // if the fit has any invalid parameters, mark it as bad
    bool isNaN = false;
    for (int i = 0; i < 5; ++i) {
     // isNaN |= std::isnan(tracks->stateAtBS.state(it)(i));
    }
    if (isNaN) {
#ifdef NTUPLE_DEBUG
      printf("NaN in fit %d size %d chi2 %f\n", it, tuples->size(it), tracks->chi2(it));
#endif
      continue;
    }

    // compute a pT-dependent chi2 cut
    // default parameters:
    //   - chi2MaxPt = 10 GeV
    //   - chi2Coeff = { 0.68177776, 0.74609577, -0.08035491, 0.00315399 }
    //   - chi2Scale = 30 for broken line fit, 45 for Riemann fit
    // (see CAHitNtupletGeneratorGPU.cc)
    float pt = std::min<float>(tracks->pt(it), cuts.chi2MaxPt);
    float chi2Cut = cuts.chi2Scale *
                    (cuts.chi2Coeff[0] + pt * (cuts.chi2Coeff[1] + pt * (cuts.chi2Coeff[2] + pt * cuts.chi2Coeff[3])));
    // above number were for Quads not normalized so for the time being just multiple by ndof for Quads  (triplets to be understood)
    if (3.f * tracks->chi2(it) >= chi2Cut) {
#ifdef NTUPLE_DEBUG
      printf("Bad fit %d size %d pt %f eta %f chi2 %f\n",
             it,
             tuples->size(it),
             tracks->pt(it),
             tracks->eta(it),
             3.f * tracks->chi2(it));
#endif
      continue;
    }

    // impose "region cuts" based on the fit results (phi, Tip, pt, cotan(theta)), Zip)
    // default cuts:
    //   - for triplets:    |Tip| < 0.3 cm, pT > 0.5 GeV, |Zip| < 12.0 cm
    //   - for quadruplets: |Tip| < 0.5 cm, pT > 0.3 GeV, |Zip| < 12.0 cm
    // (see CAHitNtupletGeneratorGPU.cc)
    auto const &region = (nhits > 3) ? cuts.quadruplet : cuts.triplet;
    bool isOk = (abs(tracks->tip(it)) < region.maxTip) and (tracks->pt(it) > region.minPt) and
                (abs(tracks->zip(it)) < region.maxZip);

    if (isOk)
      quality[it] = trackQuality::loose;
  }
}

void kernel_doStatsForTracks(HitContainer const *__restrict__ tuples,
                             Quality const *__restrict__ quality,
                             CAHitNtupletGeneratorKernelsGPU::Counters *counters,
                             sycl::nd_item<1> item) {
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int idx = first, ntot = tuples->nbins(); idx < ntot;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    if (tuples->size(idx) == 0)
      break;  //guard
    if (quality[idx] != trackQuality::loose)
      continue;
    cms::sycltools::atomic_fetch_add<unsigned long long>(&(counters->nGoodTracks), (unsigned long long)1);
  }
}

void kernel_countHitInTracks(HitContainer const *__restrict__ tuples,
                             Quality const *__restrict__ quality,
                             CAHitNtupletGeneratorKernelsGPU::HitToTuple *hitToTuple,
                             sycl::nd_item<1> item) {
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int idx = first, ntot = tuples->nbins(); idx < ntot;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    if (tuples->size(idx) == 0)
      break;  // guard
    if (quality[idx] != trackQuality::loose)
      continue;
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->countDirect(*h);
  }
 }

void kernel_fillHitInTracks(HitContainer const *__restrict__ tuples,
                            Quality const *__restrict__ quality,
                            CAHitNtupletGeneratorKernelsGPU::HitToTuple *hitToTuple,
                            sycl::nd_item<1> item) {
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int idx = first, ntot = tuples->nbins(); idx < ntot;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    if (tuples->size(idx) == 0)
      break;  // guard
    if (quality[idx] != trackQuality::loose)
      continue;
    for (auto h = tuples->begin(idx); h != tuples->end(idx); ++h)
      hitToTuple->fillDirect(*h, idx);
  }
}

void kernel_fillHitDetIndices(HitContainer const *__restrict__ tuples,
                              TrackingRecHit2DSOAView const *__restrict__ hhp,
                              HitContainer *__restrict__ hitDetIndices,
                              sycl::nd_item<1> item) {
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  // copy offsets
  for (int idx = first, ntot = tuples->totbins(); idx < ntot;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    hitDetIndices->off[idx] = tuples->off[idx];
  }
  // fill hit indices
  auto const &hh = *hhp;
  auto nhits = hh.nHits();
  for (int idx = first, ntot = tuples->size(); idx < ntot;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    assert(tuples->bins[idx] < nhits);
    hitDetIndices->bins[idx] = hh.detectorIndex(tuples->bins[idx]);
  }
}

void kernel_doStatsForHitInTracks(CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ hitToTuple,
                                  CAHitNtupletGeneratorKernelsGPU::Counters *counters,
                                  sycl::nd_item<1> item) {
  auto &c = *counters;
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int idx = first, ntot = hitToTuple->nbins(); idx < ntot;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    if (hitToTuple->size(idx) == 0)
      continue;  // SHALL NOT BE break
    cms::sycltools::atomic_fetch_add<unsigned long long>(&c.nUsedHits, (unsigned long long)1);
    if (hitToTuple->size(idx) > 1)
      cms::sycltools::atomic_fetch_add<unsigned long long>(&c.nDupHits, (unsigned long long)1);
  }
}

void kernel_tripletCleaner(TrackingRecHit2DSOAView const *__restrict__ hhp,
                           HitContainer const *__restrict__ ptuples,
                           TkSoA const *__restrict__ ptracks,
                           Quality *__restrict__ quality,
                           CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple,
                           sycl::nd_item<1> item) {
  constexpr auto bad = trackQuality::bad;
  constexpr auto dup = trackQuality::dup;
  // constexpr auto loose = trackQuality::loose;

  auto &hitToTuple = *phitToTuple;
  auto const &foundNtuplets = *ptuples;
  auto const &tracks = *ptracks;

  //  auto const & hh = *hhp;
  // auto l1end = hh.hitsLayerStart_d[1];

  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int idx = first, ntot = hitToTuple.nbins(); idx < ntot;
       idx += item.get_group_range(0) * item.get_local_range().get(0)) {
    if (hitToTuple.size(idx) < 2)
      continue;

    float mc = 10000.f;
    uint16_t im = 60000;
    uint32_t maxNh = 0;

    // find maxNh
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      uint32_t nh = foundNtuplets.size(*it);
      maxNh = sycl::max(nh, maxNh);
    }
    // kill all tracks shorter than maxHn (only triplets???)
    for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
      uint32_t nh = foundNtuplets.size(*it);
      if (maxNh != nh)
        quality[*it] = dup;
    }

    if (maxNh > 3)
      continue;
    // if (idx>=l1end) continue;  // only for layer 1
    // for triplets choose best tip!
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (quality[it] != bad && abs(tracks.tip(it)) < mc) {
        mc = abs(tracks.tip(it));
        im = it;
      }
    }
    // mark duplicates
    for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
      auto const it = *ip;
      if (quality[it] != bad && it != im)
        quality[it] = dup;  //no race:  simple assignment of the same constant
    }
  }  // loop over hits
}

void kernel_print_found_ntuplets(TrackingRecHit2DSOAView const *__restrict__ hhp,
                                 HitContainer const *__restrict__ ptuples,
                                 TkSoA const *__restrict__ ptracks,
                                 Quality const *__restrict__ quality,
                                 CAHitNtupletGeneratorKernelsGPU::HitToTuple const *__restrict__ phitToTuple,
                                 uint32_t maxPrint,
                                 int iev,
                                 sycl::nd_item<1> item) {
  auto const &foundNtuplets = *ptuples;
  auto const &tracks = *ptracks;
  int first = item.get_local_range().get(0) * item.get_group(0) + item.get_local_id(0);
  for (int i = first, np = std::min(maxPrint, foundNtuplets.nbins()); i < np;
       i += item.get_local_range().get(0) * item.get_group_range(0)) {
    auto nh = foundNtuplets.size(i);
    if (nh < 3)
      continue;
    printf("TK: %d %d %d %f %f %f %f %f %f %f %d %d %d %d %d\n",
           10000 * iev + i,
           int(quality[i]),
           nh,
           tracks.charge(i),
           tracks.pt(i),
           tracks.eta(i),
           tracks.phi(i),
           tracks.tip(i),
           tracks.zip(i),
           //           asinhf(fit_results[i].par(3)),
           tracks.chi2(i),
           *foundNtuplets.begin(i),
           *(foundNtuplets.begin(i) + 1),
           *(foundNtuplets.begin(i) + 2),
           nh > 3 ? int(*(foundNtuplets.begin(i) + 3)) : -1,
           nh > 4 ? int(*(foundNtuplets.begin(i) + 4)) : -1);
  }
}

void kernel_printCounters(cAHitNtupletGenerator::Counters const *counters) {
  auto const &c = *counters;
  printf("|| Counters      | nEvents | nHits | nCells | nTuples | nFitTacks  |  nGoodTracks | nUsedHits | nDupHits | nKilledCells | nEmptyCells | nZeroTrackCells ||\n");
  printf("|| Counters Raw  | %lld |  %lld|  %lld|  %lld|  %lld|  %lld|  %lld|  %lld|  %lld|  %lld|  %lld||\n",
         c.nEvents,
         c.nHits,
         c.nCells,
         c.nTuples,
         c.nGoodTracks,
         c.nFitTracks,
         c.nUsedHits,
         c.nDupHits,
         c.nKilledCells,
         c.nEmptyCells,
         c.nZeroTrackCells);
  printf("|| Counters Norm | %lld |  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.3f|  %.3f||\n",
         c.nEvents,
         c.nHits / double(c.nEvents),
         c.nCells / double(c.nEvents),
         c.nTuples / double(c.nEvents),
         c.nFitTracks / double(c.nEvents),
         c.nGoodTracks / double(c.nEvents),
         c.nUsedHits / double(c.nEvents),
         c.nDupHits / double(c.nEvents),
         c.nKilledCells / double(c.nEvents),
         c.nEmptyCells / double(c.nCells),
         c.nZeroTrackCells / double(c.nCells));
}
