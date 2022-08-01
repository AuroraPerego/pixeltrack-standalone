#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "SYCLCore/ESProduct.h"
#include "SYCLCore/device_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include <CL/sycl.hpp>

#include <set>

class SiPixelFedCablingMapGPUWrapper {
public:
  explicit SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const &cablingMap,
                                          std::vector<unsigned char> modToUnp);
  ~SiPixelFedCablingMapGPUWrapper();

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelFedCablingMapGPU *getGPUProductAsync(sycl::queue stream) const;

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAllAsync(sycl::queue stream) const;

private:
  std::vector<unsigned char> modToUnpDefault;
  bool hasQuality_;

  SiPixelFedCablingMapGPU *cablingMapHost_ = nullptr;  // pointer to struct in CPU

  class GPUData {
  public:
    GPUData() = default;
    ~GPUData() { sycl::free(cablingMapDevice, q_); }
    
    void set_queue(sycl::queue queue) { q_ = queue; }

    SiPixelFedCablingMapGPU *cablingMapDevice = nullptr;  // pointer to struct in GPU

  private:
    sycl::queue q_;
  };
  cms::sycltools::ESProduct<GPUData> gpuData_;

  class ModulesToUnpack {
  public:
    ModulesToUnpack() = default;
    ~ModulesToUnpack() { sycl::free(modToUnpDefault, q_); }
    void set_queue(sycl::queue queue) { q_ = queue; }
    unsigned char *modToUnpDefault = nullptr;  // pointer to GPU
  private:
    sycl::queue q_;
  };
  cms::sycltools::ESProduct<ModulesToUnpack> modToUnp_;
};

#endif