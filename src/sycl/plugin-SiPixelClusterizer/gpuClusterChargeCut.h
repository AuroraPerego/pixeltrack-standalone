#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h

#include <CL/sycl.hpp>
#include <cstdint>
#include <cstdio>

#include "SYCLCore/syclAtomic.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/prefixScan.h"

#include "gpuClusteringConstants.h"

namespace gpuClustering {

  void clusterChargeCut(uint16_t* __restrict__ id,                 // module id of each pixel (modified if bad cluster)
                        uint16_t const* __restrict__ adc,          //  charge of each pixel
                        uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
                        uint32_t* __restrict__ nClustersInModule,  // modified: number of clusters found in each module
                        uint32_t const* __restrict__ moduleId,     // module id of each module
                        int32_t* __restrict__ clusterId,           // modified: cluster id of each pixel
                        uint32_t numElements,
                        sycl::nd_item<1> item,
                        int32_t* charge,
                        uint8_t* ok,
                        uint16_t* newclusId,
                        uint16_t* ws,
                        sycl::stream out) {
    if (item.get_group(0) >= moduleStart[0])
      return;

    auto firstPixel = moduleStart[1 + item.get_group(0)];
    auto thisModuleId = id[firstPixel];
    assert(thisModuleId < MaxNumModules);
    assert(thisModuleId == moduleId[item.get_group(0)]);

    auto nclus = nClustersInModule[thisModuleId];
    if (nclus == 0)
      return;

    if (item.get_local_id(0) == 0 && nclus > MaxNumClustersPerModules) {
      out << "Warning too many clusters in module " << thisModuleId << " in block " << item.get_group(0) << ": " << nclus << " > " << MaxNumClustersPerModules << "\n";
    }

    //find another way of printing! This is the stream!
    //stream_ct1 << "Warning too many clusters in module %d in block %d: %d > %d\n";

    auto first = firstPixel + item.get_local_id(0);

    if (nclus > MaxNumClustersPerModules) {
      // remove excess  FIXME find a way to cut charge first....
      for (auto i = first; i < numElements; i += item.get_local_range(0)) {
        if (id[i] == InvId)
          continue;  // not valid
        if (id[i] != thisModuleId)
          break;  // end of module
        if (clusterId[i] >= MaxNumClustersPerModules) {
          id[i] = InvId;
          clusterId[i] = InvId;
        }
      }
      nclus = MaxNumClustersPerModules;
    }

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (item.get_local_id(0) == 0)
        out << "start clusterizer for module " << thisModuleId << " in block " << item.get_group(0) << "\n";
#endif

    assert(nclus <= MaxNumClustersPerModules);
    for (auto i = item.get_local_id(0); i < nclus; i += item.get_local_range(0)) {
      charge[i] = 0;
    }
    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item.barrier();
    for (auto i = first; i < numElements; i += item.get_local_range(0)) {
    out << charge[clusterId[i]] << " ";
    //out << adc[i] << " ";
    }
    // for (auto i = first; i < numElements; i += item.get_local_range(0)) {
    //   out << charge[clusterId[i]] << " ";
    //   out << adc[i] << " ";
    //   if (clusterId[i] > 1024)
    //    out << "we will have an issue with clusterId[" << i << "] = " << clusterId[i] << "\n";
    //   if (id[i] == InvId)
    //     continue;  // not valid
    //   if (id[i] != thisModuleId)
    //     break;  // end of module
    //     //cms::sycltools::atomic_fetch_add_shared<int32_t>(&charge[clusterId[i]], 
    //                                                     //static_cast<int32_t>(adc[i]));
    //   //cms::sycltools::AtomicAdd(&charge[clusterId[i]], adc[i]);
    // }
    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
     item.barrier();

    // auto chargeCut = thisModuleId < 96 ? 2000 : 4000;  // move in constants (calib?)
    // for (auto i = item.get_local_id(0); i < nclus; i += item.get_local_range(0)) {
    //   newclusId[i] = ok[i] = charge[i] > chargeCut ? 1 : 0;
    // }

    // /*
    // DPCT1065:2: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    // */
    // item.barrier();

    // // renumber

    // cms::sycltools::blockPrefixScan(newclusId, nclus, item, ws);

    // assert(nclus >= newclusId[nclus - 1]);

    // if (nclus == newclusId[nclus - 1])
    //   return;

    // nClustersInModule[thisModuleId] = newclusId[nclus - 1];
    // /*
    // DPCT1065:3: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    // */
    // item.barrier();

    // // mark bad cluster again
    // for (auto i = item.get_local_id(0); i < nclus; i += item.get_local_range(0)) {
    //   if (0 == ok[i])
    //     newclusId[i] = InvId + 1;
    // }
    // /*
    // DPCT1065:4: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    // */
    // item.barrier();

    // // reassign id
    // for (auto i = first; i < numElements; i += item.get_local_range(0)) {
    //   if (id[i] == InvId)
    //     continue;  // not valid
    //   if (id[i] != thisModuleId)
    //     break;  // end of module
    //   clusterId[i] = newclusId[clusterId[i]] - 1;
    //   if (clusterId[i] == InvId)
    //     id[i] = InvId;
    // }

    //done
  }

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h