#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#include "chooseDevice.h"

namespace cms::sycltools {
  std::vector<sycl::device>* enumerateDevices(bool verbose) {

    static std::vector<sycl::device> devices;
    //only the first time get devices. Else just return the vector of devices
    //already created.
    if (devices.empty())
    {
    devices = sycl::device::get_devices(sycl::info::device_type::all);
    if (verbose) {
      std::cerr << "Found " << devices.size() << " SYCL devices:" << std::endl;
      for (auto const& device : devices)
        std::cerr << "  - " << device.get_info<cl::sycl::info::device::name>() << std::endl;
      std::cerr << std::endl;
    }
    }
    return &devices;
  }

  sycl::device chooseDevice(edm::StreamID id) {
      auto const devices = *(enumerateDevices());
      sycl::device device;
      device = devices[0];
    
      std::cout << "Device selected: " << device.get_info<cl::sycl::info::device::name>() 
                << " with backend "    << device.get_backend() << std::endl;

    //
    //// For startes we "statically" assign the device based on
    //// edm::Stream number. This is suboptimal if the number of
    //// edm::Streams is not a multiple of the number of CUDA devices
    //// (and even then there is no load balancing).
    ////
    //// TODO: improve the "assignment" logic
    //auto device = devices[0]; //id % devices.size()];
    //for(int i = 0; i < (int)devices.size(); i++){
    //  if (devices[i].is_cpu()){
    //    device = devices[i];
    //    break;
    //  }
    //}
    //auto device = sycl::device(sycl::cpu_selector());    
    //std::cerr << "EDM stream " << id << " offload to " << device.get_info<cl::sycl::info::device::name>() << std::endl;
    return device;
  }
}  // namespace cms::sycltools
