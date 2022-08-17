#include <CL/sycl.hpp>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  gainForHLTonHost_ = new SiPixelGainForHLTonGPU();
  *gainForHLTonHost_ = gain;
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() { delete gainForHLTonHost_; }

const SiPixelGainForHLTonGPU* SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(sycl::queue stream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, sycl::queue stream) {
    // data.gainForHLTonGPU = cms::sycltools::make_device_unique_uninitialized<SiPixelGainForHLTonGPU>(stream);
    // data.gainDataOnGPU = cms::sycltools::make_device_unique_uninitialized<SiPixelGainForHLTonGPU_DecodingStructure[]>(
    //     this->gainData_.size(), stream);

    data.gainForHLTonGPU = (SiPixelGainForHLTonGPU *)sycl::malloc_device(sizeof(SiPixelGainForHLTonGPU), stream);
    data.gainDataOnGPU = (SiPixelGainForHLTonGPU_DecodingStructure *)sycl::malloc_device(this->gainData_.size(), stream);
    // for (int i=0; i<48316; i++)
    // std::cout << static_cast<int>(this->gainData_.data()[i]) << " ";

    stream.memcpy(data.gainDataOnGPU, this->gainData_.data(), this->gainData_.size()).wait();

    this->gainForHLTonHost_->v_pedestals = data.gainDataOnGPU;

    stream.memcpy(data.gainForHLTonGPU, this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU)).wait();

    //data.gainForHLTonGPU->v_pedestals = data.gainDataOnGPU;
    //stream.memcpy(&(data.gainForHLTonGPU->v_pedestals),
    //            &(data.gainDataOnGPU),
    //            sizeof(SiPixelGainForHLTonGPU_DecodingStructure*)).wait();

    //SiPixelGainForHLTonGPU_DecodingStructure* hostSock;

     SiPixelGainForHLTonGPU* hostSock = (SiPixelGainForHLTonGPU *)sycl::malloc_host(sizeof(SiPixelGainForHLTonGPU),stream);

     stream.memcpy(hostSock, (data.gainForHLTonGPU), sizeof(SiPixelGainForHLTonGPU)).wait();

     for (int i=0; i<50; i++)
     std::cout << static_cast<int>(hostSock->v_pedestals[i].gain) << " ";            
    //  for (int i=0; i<48316; i++)
    //  std::cout << static_cast<int>(data.gainForHLTonGPU->v_pedestals->ped) << " ";
  
  });
  return data.gainForHLTonGPU;
}