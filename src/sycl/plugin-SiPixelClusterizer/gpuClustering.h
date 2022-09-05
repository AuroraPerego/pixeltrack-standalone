#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

#include <CL/sycl.hpp>
#include <cstdint>
#include <cstdio>

#include "Geometry/phase1PixelTopology.h"
#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"
#include "assert.h"
#include <cassert>

#include "gpuClusteringConstants.h"

#define ABS(x) ((x < 0) ? -x : x)
namespace gpuClustering {

#define GPU_DEBUG TRUE
#ifdef GPU_DEBUG
  uint32_t gMaxHit = 0;
#endif

  void countModules(uint16_t const* __restrict__ id,
                    uint32_t* __restrict__ moduleStart,
                    int32_t* __restrict__ clusterId,
                    int numElements,
                    sycl::nd_item<1> item,
                    sycl::stream out) {
    int first = item.get_local_range(0) * item.get_group(0) + item.get_local_id(0);
    for (int i = first; i < numElements; i += item.get_group_range(0) * item.get_local_range(0)) {
      clusterId[i] = i;
      if (InvId == id[i])
        continue;
      auto j = i - 1;
      while (j >= 0 and id[j] == InvId)
        --j;
      if (j < 0 or id[j] != id[i]) {
        //auto loc = cms::sycltools::AtomicAdd(moduleStart, 1);
        //out << "out";
        auto loc = cms::sycltools::atomic_fetch_add<uint32_t>(moduleStart, static_cast<uint32_t>(1));
        moduleStart[loc + 1] = i;
      }
    }
    //out << "Count Modules moduleStart is: " << moduleStart[0] << "\n";
  }

  //init hist  (ymax=416 < 512 : 9bits)
  constexpr uint32_t maxPixInModule = 4000;
  constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2;
  using Hist = cms::sycltools::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;

  void findClus(uint16_t const* __restrict__ id,           // module id of each pixel
                uint16_t const* __restrict__ x,            // local coordinates of each pixel
                uint16_t const* __restrict__ y,            //
                uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
                uint32_t* __restrict__ nClustersInModule,  // output: number of clusters found in each module
                uint32_t* __restrict__ moduleId,           // output: module id of each module
                int32_t* __restrict__ clusterId,           // output: cluster id of each pixel
                int numElements,
                sycl::nd_item<1> item,
                uint32_t* gMaxHit,
                int* msize,
                //Hist* hist,
                //Hist::Counter* ws,  //was ws[32], maybe this can be set in accessore definition
                uint32_t* totGood,
                uint32_t* n40,
                uint32_t* n60,
                int* n0,
                unsigned int* foundClusters,
                sycl::stream out) {
    if (item.get_group(0) >= moduleStart[0])
      return;
    auto firstPixel = moduleStart[1 + item.get_group(0)];
    auto thisModuleId = id[firstPixel];
    assert(thisModuleId < MaxNumModules);
    //  #ifdef GPU_DEBUG
    //      if (thisModuleId % 100 == 1)
    //        if (item.get_local_id(0) == 0)
    //          out << "start clusterizer for module " << thisModuleId << " in block " << item.get_group(0) << "\n";
    //  #endif
    auto first = firstPixel + item.get_local_id(0);

    // find the index of the first pixel not belonging to this module (or invalid)
    *msize = numElements;
    item.barrier();

    // skip threads not associated to an existing pixel
    for (int i = first; i < numElements; i += item.get_local_range(0)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (id[i] != thisModuleId) {  //find the first pixel in a different module
        cms::sycltools::atomic_fetch_min_shared<int>(static_cast<int*>(msize), static_cast<int>(i));
        break;
      }
    }
    
     auto wsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t[32]>(item.get_group());
     uint32_t* ws = (uint32_t*)wsbuff.get();
     auto histbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<Hist>(item.get_group());
     Hist* hist = (Hist*)histbuff.get();

    //constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2
    for (auto j = item.get_local_id(0); j < Hist::totbins(); j += item.get_local_range(0)) {
      hist->off[j] = 0;
    }

    /*
         DPCT1065:2: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier( ) for better performance if there is no access to global memory.
         */
    item.barrier();
    assert((*msize == numElements) or ((*msize < numElements) and (id[(*msize)] != thisModuleId)));

    constexpr int maxPixInModule = 4000;
    // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
    if (0 == item.get_local_id(0)) {
      if ((*msize - static_cast<int>(firstPixel)) > maxPixInModule) {
        out << "too many pixels in module " << thisModuleId << ": " << *msize - static_cast<int>(firstPixel) << " > "
            << maxPixInModule << "\n";
        *msize = maxPixInModule + firstPixel;
      }
    }

    item.barrier();
    assert(*msize - static_cast<int>(firstPixel) <= maxPixInModule);
#ifdef GPU_DEBUG
    //__shared__ uint32_t totGood;
    *totGood = 0;
    item.barrier();
#endif

    // fill histo
    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId) {  // skip invalid pixels
        continue;
      }
      hist->count(y[i]);
#ifdef GPU_DEBUG
      cms::sycltools::atomic_fetch_add_shared<uint32_t>(totGood, static_cast<uint32_t>(1));
#endif
    }

    /*
             DPCT1065:4: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier( ) for better performance if there is no access to global memory.
             */
    item.barrier();
    if (item.get_local_id(0) < 32u)
      ws[item.get_local_id(0)] = 0;  // used by prefix scan...
    /*
             DPCT1065:5: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier( ) for better performance if there is no access to global memory.
             */
    item.barrier();
    hist->finalize(item, ws);
    /*
             DPCT1065:6: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier( ) for better performance if there is no access to global memory.
             */
    item.barrier();
    //  #ifdef GPU_DEBUG
    //      assert(hist->size() == totGood);
    //      if (thisModuleId % 100 == 1)
    //        if (item.get_local_id(0) == 0)
    //          out << "histo size" << hist->size() <<"\n";
    //  #endif
    for (int i = first; i < *msize; i += item.get_local_range(0)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      hist->fill(y[i], i - firstPixel);
    }

    // assume that we can cover the whole module with up to 16 blockDim.x-wide iterations
    constexpr int maxiter = 16;  // it was auto maxiter = hist->size(); ifndef CUDA_ARCH but ariable length arrays are not supported in SYC
    // allocate space for duplicate pixels: a pixel can appear more than once with different charge in the same event
    constexpr int maxNeighbours = 10;

    assert((hist->size() / item.get_local_range(0)) <= maxiter);
    //nearest neighbour
    uint16_t nn[maxiter][maxNeighbours];
    uint8_t nnn[maxiter];  // number of nn
    for (uint32_t k = 0; k < maxiter; ++k)
      nnn[k] = 0;
    /*
             DPCT1065:7: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier( ) for better performance if there is no access to global memory.
             */
    item.barrier();  // for hit filling
#ifdef GPU_DEBUG
        // look for anomalous high occupancy
    *n40 = *n60 = 0;
    item.barrier();
    for (auto j = item.get_local_id(0); j < Hist::nbins(); j += item.get_local_range(0)) {
      if (hist->size(j) > 60)
        cms::sycltools::atomic_fetch_add_shared<uint32_t>(n60, static_cast<uint32_t>(1));
      if (hist->size(j) > 40)
        cms::sycltools::atomic_fetch_add_shared<uint32_t>(n40, static_cast<uint32_t>(1));
    }
    item.barrier();
    if (0 == item.get_local_id(0)) {
      if (*n60 > 0)
        out << "columns with more than 60 px " << *n60 << " in " << thisModuleId << "\n";
      else if (*n40 > 0)
        out << "columns with more than 40 px " << *n40 << " in " << thisModuleId << "\n";
    }
    item.barrier();
#endif

    //out << hist->size() << " ";
    //fill NN
    //uint32_t k = 0u;
    for (auto j = item.get_local_id(0), k = (unsigned long)0U; j < hist->size(); j += item.get_local_range(0), ++k) {
      //k = j / 256;
      assert(k < maxiter);
      auto p = hist->begin() + j;
      auto i = *p + firstPixel;
      assert(id[i] != InvId);
      assert(id[i] == thisModuleId);  // same module
      int be = Hist::bin(y[i] + 1);
      auto e = hist->end(be);
      ++p;
      //out << *e << " ";
      assert(0 == nnn[k]);
      for (; p < e; ++p) {
        auto m = (*p) + firstPixel;
        assert(m != i);
        assert(int(y[m]) - int(y[i]) >= 0);
        assert(int(y[m]) - int(y[i]) <= 1);
        if (sycl::abs(int(x[m]) - int(x[i])) > 1)
          continue;
        auto l = nnn[k]++;
        assert(l < maxNeighbours);
        nn[k][l] = *p;
        //out << "nn[" << k << "][" << l << "]= " << nn[k][l] << " ";
      }
    }

      // for each pixel, look at all the pixels until the end of the module;
      // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
      // after the loop, all the pixel in each cluster should have the id equeal to the lowest
      // pixel in the cluster ( clus[i] == i ).
    bool more = true;
    int nloops = 0;
    /*
            DPCT1065:13: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier() for better performance if there is no access to global memory.
            */

    while ((item.barrier(), sycl::any_of_group(item.get_group(), more))) {
      if (1 == nloops % 2) {
        for (auto j = item.get_local_id(0), k = (unsigned long)0U; j < hist->size();
             j += item.get_local_range(0), ++k) {
          auto p = hist->begin() + j;
          auto i = *p + firstPixel;
          auto m = clusterId[i];
          while (m != clusterId[m])
            m = clusterId[m];
          clusterId[i] = m;
        }
      } else {
        more = false;
        for (auto j = item.get_local_id(0), k = (unsigned long)0U; j < hist->size();
             j += item.get_local_range(0), ++k) {
          auto p = hist->begin() + j;
          auto i = *p + firstPixel;
          for (int kk = 0; kk < nnn[k]; ++kk) {
            auto l = nn[k][kk];
            auto m = l + firstPixel;
            assert(m != i);
            auto old = cms::sycltools::atomic_fetch_min<int32_t>(static_cast<int32_t*>(&clusterId[m]),
                                                                 static_cast<int32_t>(clusterId[i]));
            //auto old = cms::sycltools::AtomicMin(&clusterId[m], clusterId[i]);
            if (old != clusterId[i]) {
              // end the loop only if no changes were applied
              more = true;
            }
            cms::sycltools::atomic_fetch_min<int32_t>(static_cast<int32_t*>(&clusterId[i]), static_cast<int32_t>(old));
            //cms::sycltools::AtomicMin(&clusterId[i], old);
          }  // nnloop
        }    // pixel loop
      }
      ++nloops;
    }  // end while

    // #ifdef GPU_DEBUG
    //       {
    //         if (item.get_local_id(0) == 0)
    //           *n0 = nloops;
    //          item.barrier();
    //         auto ok = *n0 == nloops;
    //         assert((item.barrier(), sycl::all_of_group(item.get_group(), ok)));
    //         if (thisModuleId % 100 == 1)
    //           if (item.get_local_id(0) == 0)
    //             out << "# loops " << nloops << "\n";
    //       }
    // #endif
        *foundClusters = 0;
        /*
        DPCT1065:8: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier( ) for better performance if there is no access to global memory.
        */
        item.barrier();
        // find the number of different clusters, identified by a pixels with clus[i] == i;
        // mark these pixels with a negative id.
        for (int i = first; i < *msize; i += item.get_local_range(0)) {
          if (id[i] == InvId)  // skip invalid pixels
            continue;
          if (clusterId[i] == i) {
            //out << "got here : " ;
            auto old = cms::sycltools::atomic_fetch_compare_inc_shared<unsigned int>(foundClusters,
                                         static_cast<unsigned int>(0xffffffff));
            clusterId[i] = -(old + 1);
          }
        }
        /*
        DPCT1065:9: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier( ) for better performance if there is no access to global memory.
        */
        item.barrier();
        // propagate the negative id to all the pixels in the cluster.
        for (int i = first; i < *msize; i += item.get_local_range(0)) {
          if (id[i] == InvId)  // skip invalid pixels
            continue;
          if (clusterId[i] >= 0) {
            // mark each pixel in a cluster with the same id as the first one
            clusterId[i] = clusterId[clusterId[i]];
          }
        }
        /*
        DPCT1065:10: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier( ) for better performance if there is no access to global memory.
        */
        item.barrier();
        // adjust the cluster id to be a positive value starting from 0
        for (int i = first; i < *msize; i += item.get_local_range(0)) {
          if (id[i] == InvId) {  // skip invalid pixels
            clusterId[i] = -9999;
            continue;
          }
          clusterId[i] = -clusterId[i] - 1;
        }
        /*
        DPCT1065:11: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier( ) for better performance if there is no access to global memory.
        */
        item.barrier();
        if (item.get_local_id(0) == 0) {
          nClustersInModule[thisModuleId] = *foundClusters;
          moduleId[item.get_group(0)] = thisModuleId;
    //  #ifdef GPU_DEBUG
    //        if (foundClusters > gMaxHit) {
    //          gMaxHit = foundClusters;
    //          if (*foundClusters > 8)
    //            out << "max hit " << foundClusters << " in " << thisModuleId << "\n";
    //        }
    //  #endif
    //  #ifdef GPU_DEBUG
    //        if (thisModuleId % 100 == 1)
    //          out << *foundClusters << " clusters in module " << thisModuleId << "\n";
    //  #endif
    }
    out << *foundClusters << " clusters in module " << thisModuleId << "\n";
  }

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h