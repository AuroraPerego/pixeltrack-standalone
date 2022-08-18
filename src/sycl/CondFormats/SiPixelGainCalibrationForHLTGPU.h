#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h

#include <CL/sycl.hpp>

#include "SYCLCore/ESProduct.h"
#include "SYCLCore/device_unique_ptr.h"

class SiPixelGainForHLTonGPU;
struct SiPixelGainForHLTonGPU_DecodingStructure;

class SiPixelGainCalibrationForHLTGPU {
public:
  explicit SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const &gain, std::vector<char> gainData);
  ~SiPixelGainCalibrationForHLTGPU();

  const SiPixelGainForHLTonGPU *getGPUProductAsync(sycl::queue stream) const;
  const SiPixelGainForHLTonGPU *getCPUProduct() const { return gainForHLTonHost_; }


   const void printgainData() const{
    for (int i=0; i<48316 ;i++)
    std::cout << gainData_[i] << std::endl;
  }


private:
  SiPixelGainForHLTonGPU *gainForHLTonHost_ = nullptr;
  std::vector<char> gainData_;
  struct GPUData {
    GPUData() = default;
    ~GPUData() {}

    // std::unique_ptr<SiPixelGainForHLTonGPU, cms::sycltools::device::impl::DeviceDeleter> gainForHLTonGPU;
    // std::unique_ptr<SiPixelGainForHLTonGPU_DecodingStructure[], cms::sycltools::device::impl::DeviceDeleter>
    //     gainDataOnGPU;

    SiPixelGainForHLTonGPU *gainForHLTonGPU = nullptr;
    SiPixelGainForHLTonGPU_DecodingStructure *gainDataOnGPU = nullptr;
    
  };
  cms::sycltools::ESProduct<GPUData> gpuData_;
};

#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h