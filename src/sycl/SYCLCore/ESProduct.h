#ifndef HeterogeneousCore_SYCLCore_ESProduct_h
#define HeterogeneousCore_SYCLCore_ESProduct_h

#include <atomic>
#include <cassert>
#include <mutex>
#include <vector>

#include "SYCLCore/ScopedSetDevice.h"
#include "SYCLCore/chooseDevice.h"
#include "SYCLCore/getDeviceIndex.h"
#include "SYCLCore/eventWorkHasCompleted.h"

namespace cms {
  namespace sycltools {

    template <typename T>
    class ESProduct {
    public:
      ESProduct() : gpuDataPerDevice_(enumerateDevices().size()) {}

      ~ESProduct() = default;

      // transferAsync should be a function of (T&, cudaStream_t)
      // which enqueues asynchronous transfers (possibly kernels as well)
      // to the CUDA stream
      template <typename F>
      const T& dataForCurrentDeviceAsync(sycl::queue stream, F transferAsync) const {
	      int dev_idx = getDeviceIndex(stream.get_device());
        auto& data = gpuDataPerDevice_[dev_idx];
      
        // If the GPU data has already been filled, we can return it immediately
        if (not data.m_filled.load()) {
          // It wasn't, so need to fill it
          std::scoped_lock<std::mutex> lk{data.m_mutex};

          if (data.m_filled.load()) {
            // Other thread marked it filled while we were locking the mutex, so we're free to return it
            return data.m_data;
          }

          if (data.m_fillingStream) {
            // Someone else is filling
            // Check first if the recorded event has occurred
            assert(data.m_event);
            if (eventWorkHasCompleted(*data.m_event)) {
              // It was, so data is accessible from all CUDA streams on
              // the device. Set the 'filled' for all subsequent calls and
              // return the value
              auto should_be_false = data.m_filled.exchange(true);
              assert(not should_be_false);
              data.m_fillingStream.reset();
              data.m_event.reset();
            } else if (*data.m_fillingStream != stream) {
              // Filling is still going on. For other CUDA stream, add
              // wait on the CUDA stream and return the value. Subsequent
              // work queued on the stream will wait for the event to
              // occur (i.e. transfer to finish).

              stream.ext_oneapi_submit_barrier({*data.m_event});
              //was cudaCheck(cudaStreamWaitEvent(cudaStream, data.m_event.get(), 0),
              //          "Failed to make a stream to wait for an event");
            }
            // else: filling is still going on. But for the same CUDA
            // stream (which would be a bit strange but fine), we can just
            // return as all subsequent work should be enqueued to the
            // same CUDA stream (or stream to be explicitly synchronized
            // by the caller)
          } else {
            // Now we can be sure that the data is not yet on the GPU, and
            // this thread is the first to try that.
            transferAsync(data.m_data, stream);
            assert(not data.m_fillingStream);
            data.m_fillingStream = stream;
            // Record in the cudaStream an event to mark the readiness of the
            // EventSetup data on the GPU, so other streams can check for it
            assert(not data.m_event);
            data.m_event = stream.ext_oneapi_submit_barrier(); 
            //was cudaCheck(cudaEventRecord(data.m_event.get(), stream)); 

            // Now the filling has been enqueued to the cudaStream, so we
            // can return the GPU data immediately, since all subsequent
            // work must be either enqueued to the cudaStream, or the cudaStream
            // must be synchronized by the caller
          }
        }

        return data.m_data;
      }

    private:
      struct Item {
        mutable std::mutex m_mutex;
        mutable std::optional<sycl::event> m_event;          // guarded by m_mutex
        // non-null if some thread is already filling (cudaStream_t is just a pointer)
        mutable std::optional<sycl::queue> m_fillingStream;  // guarded by m_mutex
        mutable std::atomic<bool> m_filled = false;          // easy check if data has been filled already or not
        mutable T m_data;                                    // guarded by m_mutex
      };

      std::vector<Item> gpuDataPerDevice_;
    };
  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_SYCLCore_ESProduct_h
