// C++ includes
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes
#include <CL/sycl.hpp>

// CMSSW includes
//#include "SYCLCore/syclCheck.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
    : modToUnpDefault(modToUnp.size()), hasQuality_(true) {
  cablingMapHost_ = new SiPixelFedCablingMapGPU();
  //std::unique_ptr<SiPixelFedCablingMapGPU> cablingMapHost = std::make_unique<SiPixelFedCablingMapGPU>();
  std::memcpy(cablingMapHost_, &cablingMap, sizeof(SiPixelFedCablingMapGPU));
  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() { delete cablingMapHost_; }

const SiPixelFedCablingMapGPU* SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(sycl::queue stream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, sycl::queue& stream) {
    // allocate
    data.set_queue(stream);
    data.cablingMapDevice = (SiPixelFedCablingMapGPU*)sycl::malloc_device(sizeof(SiPixelFedCablingMapGPU), stream);
    // transfer
    stream.memcpy(data.cablingMapDevice, this->cablingMapHost_, sizeof(SiPixelFedCablingMapGPU)).wait();
    return data;
  });
  return data.cablingMapDevice;
}

const unsigned char* SiPixelFedCablingMapGPUWrapper::getModToUnpAllAsync(sycl::queue stream) const {
  const auto& data = modToUnp_.dataForCurrentDeviceAsync(stream, [this](ModulesToUnpack& data, sycl::queue stream) {
    data.set_queue(stream);
    data.modToUnpDefault = (unsigned char*)sycl::malloc_device(pixelgpudetails::MAX_SIZE_BYTE_BOOL, stream);
    stream
        .memcpy(
            data.modToUnpDefault, this->modToUnpDefault.data(), this->modToUnpDefault.size() * sizeof(unsigned char))
        .wait();
  });
  return data.modToUnpDefault;
}