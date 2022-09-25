#ifndef HeterogeneousCore_SYCLUtilities_interface_printf_h
#define HeterogeneousCore_SYCLUtilities_interface_printf_h

#include <CL/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

#define printf(FORMAT, ...) \
do { \
  static const CONSTANT char format[] = FORMAT; \
  sycl::ext::oneapi::experimental::printf(format, ##__VA_ARGS__); \
} while (false)

#endif //HeterogeneousCore_SYCLUtilities_interface_printf_h
