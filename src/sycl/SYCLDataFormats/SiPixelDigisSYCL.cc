#include "SYCLDataFormats/SiPixelDigisSYCL.h"

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
//#include "SYCLCore/copyAsync.h" check if it works w/o this, using directly memcpy

SiPixelDigisSYCL::SiPixelDigisSYCL(size_t maxFedWords, sycl::queue stream) {
  xx_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream, "xx_d");
  yy_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream, "yy_d");
  adc_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream, "adc_d");
  moduleInd_d = cms::sycltools::make_device_unique<uint16_t[]>(maxFedWords, stream, "moduleInd_d");
  clus_d = cms::sycltools::make_device_unique<int32_t[]>(maxFedWords, stream, "clus_d");

  pdigi_d = cms::sycltools::make_device_unique<uint32_t[]>(maxFedWords, stream, "pdigi_d");
  rawIdArr_d = cms::sycltools::make_device_unique<uint32_t[]>(maxFedWords, stream, "rawIdArr_d");

  auto view = cms::sycltools::make_host_unique<DeviceConstView>(stream);
  view->xx_ = xx_d.get(); 
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

  view_d = cms::sycltools::make_device_unique<DeviceConstView>(stream);
  stream.memcpy(view_d.get(), view.get(), sizeof(DeviceConstView));
  //cms::sycltools::copyAsync(view_d, view, stream);
}

cms::sycltools::host::unique_ptr<uint16_t[]> SiPixelDigisSYCL::adcToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<uint16_t[]>(nDigis(), stream);
  stream.memcpy(ret.get(), adc_d.get(), nDigis() * sizeof(uint16_t));
  //cms::sycltools::copyAsync(ret, adc_d, nDigis(), stream);
  return ret;
}

cms::sycltools::host::unique_ptr<int32_t[]> SiPixelDigisSYCL::clusToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<int32_t[]>(nDigis(), stream);
  stream.memcpy(ret.get(), clus_d.get(), nDigis() * sizeof(int32_t));
  //cms::sycltools::copyAsync(ret, clus_d, nDigis(), stream);
  return ret;
}

cms::sycltools::host::unique_ptr<uint32_t[]> SiPixelDigisSYCL::pdigiToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<uint32_t[]>(nDigis(), stream);
  stream.memcpy(ret.get(), pdigi_d.get(), nDigis() * sizeof(uint32_t));
  //cms::sycltools::copyAsync(ret, pdigi_d, nDigis(), stream);
  return ret;
}

cms::sycltools::host::unique_ptr<uint32_t[]> SiPixelDigisSYCL::rawIdArrToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<uint32_t[]>(nDigis(), stream);
  stream.memcpy(ret.get(), rawIdArr_d.get(), nDigis() * sizeof(uint32_t));
  //cms::sycltools::copyAsync(ret, rawIdArr_d, nDigis(), stream);
  return ret;
}

uint16_t* SiPixelDigisSYCL::moduleIndToHostAsync(sycl::queue stream) const {
  //auto ret = cms::sycltools::make_host_unique<uint16_t[]>(nDigis(), stream);
  std::cout << "ndigis is :" << nDigis() << std::endl;
  uint16_t *ret = (uint16_t*)sycl::malloc_host(sizeof(uint16_t) * 48316, stream);
  stream.memcpy(ret, moduleInd_d.get(), 48316 * sizeof(uint16_t)).wait();
  //cms::sycltools::copyAsync(ret, adc_d, nDigis(), stream);
  return ret;
}