#include <CL/sycl.hpp>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
//#include "CUDACore/cudaCheck.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  //std::unique_ptr<SiPixelGainForHLTonGPU> gainForHLTonHost_ = std::make_unique<SiPixelGainForHLTonGPU>();
  gainForHLTonHost_ = new SiPixelGainForHLTonGPU();
  *gainForHLTonHost_ = gain;
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {  
}

SiPixelGainCalibrationForHLTGPU::GPUData::~GPUData() {
}

const SiPixelGainForHLTonGPU* SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(sycl::queue syclStream) const {
    const auto& data = gpuData_.dataForCurrentDeviceAsync(syclStream, [this](GPUData& data, sycl::queue stream) {


      data.gainForHLTonGPU = (SiPixelGainForHLTonGPU *)sycl::malloc_device(sizeof(SiPixelGainForHLTonGPU), stream);
      data.gainDataOnGPU = (SiPixelGainForHLTonGPU_DecodingStructure *)sycl::malloc_device(this->gainData_.size(), stream);
 
      stream.memcpy(data.gainDataOnGPU, this->gainData_.data(), this->gainData_.size());
      
      stream.memcpy(data.gainForHLTonGPU, this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU));
      
      stream.memcpy(&(data.gainForHLTonGPU->v_pedestals), &(data.gainDataOnGPU), sizeof(SiPixelGainForHLTonGPU_DecodingStructure*));
      
  });
  return data.gainForHLTonGPU;
}
