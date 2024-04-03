// The omission of #include guards is on purpose: it does make sense to #include
// this file multiple times, setting a different value of GPU_DEBUG beforehand.

#ifdef __SYCL_DEVICE_ONLY__
#ifndef GPU_DEBUG
// disable asserts
#ifndef NDEBUG
#define NDEBUG
#endif
#else
// enable asserts
#ifdef NDEBUG
#undef NDEBUG
#endif
#endif  // GPU_DEBUG
#endif  //__SYCL_DEVICE_ONLY__

#include <cassert>

//#include <sycl/sycl.hpp>

//extern "C"
//{
//extern void __assertfail(const char * __assertion,
//                                             const char *__file,
//                                             unsigned int __line,
//                                             const char *__function,
//                                             size_t charsize)
//#ifndef __SYCL_DEVICE_ONLY__
//{
//  fprintf(stderr, "Host-part of CUDA's assertfail: Assertion %s in file %s @ %ud, func %s\n",
//          __assertion, __file, __line, __function);
//}
//#endif
//;
//}
//
//void __assert_fail(const char *expr, const char *file, unsigned int line,
//                   const char *func) {
//  __assertfail(expr, file, line, func, 1);
//}
//
