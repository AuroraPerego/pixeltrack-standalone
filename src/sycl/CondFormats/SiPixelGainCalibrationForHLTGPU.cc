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
    data.gainForHLTonGPU = cms::sycltools::make_device_unique_uninitialized<SiPixelGainForHLTonGPU>(stream);
    data.gainDataOnGPU = cms::sycltools::make_device_unique_uninitialized<SiPixelGainForHLTonGPU_DecodingStructure[]>(
        this->gainData_.size(), stream);

    stream.memcpy(data.gainDataOnGPU.get(), this->gainData_.data(), this->gainData_.size()).wait();

    this->gainForHLTonHost_->v_pedestals = data.gainDataOnGPU.get();

    stream.memcpy(data.gainForHLTonGPU.get(), this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU)).wait();

    // SiPixelGainForHLTonGPU* hostSock = (SiPixelGainForHLTonGPU *)sycl::malloc_host(sizeof(SiPixelGainForHLTonGPU),stream);

    stream.memcpy(data.gainForHLTonGPU.get()->v_pedestals, (data.gainForHLTonGPU.get()), sizeof(SiPixelGainForHLTonGPU_DecodingStructure)).wait();

  });
  return data.gainForHLTonGPU.get();
}