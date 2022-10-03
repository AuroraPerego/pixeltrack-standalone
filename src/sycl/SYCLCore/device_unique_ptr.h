#ifndef HeterogeneousCore_SYCLUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_SYCLUtilities_interface_device_unique_ptr_h

#include <functional>
#include <memory>
#include <optional>
#include <iostream>

#include <CL/sycl.hpp>
#include "SYCLCore/getCachingAllocator.h"

namespace cms {
  namespace sycltools {
    namespace device {
      namespace impl {
        // Additional layer of types to distinguish from host::unique_ptr
        class DeviceDeleter {
        public:
          DeviceDeleter() = default;  // for edm::Wrapper
          DeviceDeleter(sycl::queue stream) : stream_{stream} {}
          DeviceDeleter(sycl::queue stream, std::string varName) : stream_{stream}, varName_{varName} {}

          void operator()(void *ptr) {
            if (stream_) {
              //if (ptr != nullptr && !varName_.empty())
              //  std::cout << "Deallocating " << varName_ << std::endl;
              auto dev = (*stream_).get_device();
              CachingAllocator& allocator = getCachingAllocator(dev);
              allocator.free(ptr);
            }
          }

        private:
          std::optional<sycl::queue> stream_;
          std::string varName_;
        };
      }  // namespace impl

      template <typename T>
      using unique_ptr = std::unique_ptr<T, impl::DeviceDeleter>;

      namespace impl {
        template <typename T>
        struct make_device_unique_selector {
          using non_array = cms::sycltools::device::unique_ptr<T>;
        };
        template <typename T>
        struct make_device_unique_selector<T[]> {
          using unbounded_array = cms::sycltools::device::unique_ptr<T[]>;
        };
        template <typename T, size_t N>
        struct make_device_unique_selector<T[N]> {
          struct bounded_array {};
        };
      }  // namespace impl
    }    // namespace device

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique(sycl::queue stream) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      CachingAllocator& allocator = getCachingAllocator(stream.get_device());
      void* mem = allocator.allocate(sizeof(T), stream, false);
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                              device::impl::DeviceDeleter{stream}};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique(size_t n,
                                                                                              sycl::queue stream,
                                                                                              std::string varName="") {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      CachingAllocator& allocator = getCachingAllocator(stream.get_device());
      void* mem = allocator.allocate(n * sizeof(element_type), stream, false);
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem), device::impl::DeviceDeleter{stream, varName}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_unique_selector<T>::bounded_array make_device_unique(Args &&...) = delete;

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique_uninitialized(
        sycl::queue stream) {
      CachingAllocator& allocator = getCachingAllocator(stream.get_device());
      void* mem = allocator.allocate(sizeof(T), stream, false);
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                              device::impl::DeviceDeleter{stream}};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique_uninitialized(
        size_t n, sycl::queue stream) {
      using element_type = typename std::remove_extent<T>::type;
      CachingAllocator& allocator = getCachingAllocator(stream.get_device());
      void* mem = allocator.allocate(n * sizeof(element_type), stream, false);
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem), device::impl::DeviceDeleter{stream}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_unique_selector<T>::bounded_array make_device_unique_uninitialized(Args &&...) =
        delete;
  }  // namespace sycltools
}  // namespace cms

#endif
