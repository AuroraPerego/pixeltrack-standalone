//TODO

#include "SYCLCore/EventCache.h"
//#include "SYCLCore/syclCheck.h"
#include "SYCLCore/currentDevice.h"
#include "SYCLCore/deviceCount.h"
#include "SYCLCore/eventWorkHasCompleted.h"
#include "SYCLCore/ScopedSetDevice.h"

namespace cms::sycltools {
  /*void EventCache::Deleter::operator()(sycl::event event) const {
    if (device_ != -1) {
      ScopedSetDevice deviceGuard{device_};
      cudaCheck(cudaEventDestroy(event));
    }
  }*/

  // EventCache should be constructed by the first call to
  // getEventCache() only if we have CUDA devices present
  EventCache::EventCache() : cache_(deviceCount()) {}

  SharedEventPtr EventCache::get(sycl::device dev) {
    //const auto dev = stream.get_device();
    vector<sycl::device> device_list = sycl::device::get_devices(sycl::info::device_type::all);
	  int dev_idx = distance(device_list.begin(), find(device_list.begin(), device_list.end(), dev));
    auto event = makeOrGet(dev_idx);
    // captured work has completed, or a just-created event
    if (eventWorkHasCompleted(event.get())) {
      return event;
    }

    // Got an event with incomplete captured work. Try again until we
    // get a completed (or a just-created) event. Need to keep all
    // incomplete events until a completed event is found in order to
    // avoid ping-pong with an incomplete event.
    std::vector<SharedEventPtr> ptrs{std::move(event)};
    bool completed;
    do {
      event = makeOrGet(dev_idx);
      completed = eventWorkHasCompleted(event.get());
      if (not completed) {
        ptrs.emplace_back(std::move(event));
      }
    } while (not completed);
    return event;
  }

  SharedEventPtr EventCache::makeOrGet(int dev) {
    return cache_[dev].makeOrGet([dev]() {
      sycl::event event;
      // it should be a bit faster to ignore timings
      //cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      return std::unique_ptr<BareEvent, Deleter>(event, Deleter{dev});
    });
  }

  void EventCache::clear() {
    // Reset the contents of the caches, but leave an
    // edm::ReusableObjectHolder alive for each device. This is needed
    // mostly for the unit tests, where the function-static
    // EventCache lives through multiple tests (and go through
    // multiple shutdowns of the framework).
    cache_.clear();
    cache_.resize(deviceCount());
  }

  EventCache& getEventCache() {
    // the public interface is thread safe
    static EventCache cache;
    return cache;
  }
}  // namespace cms::cuda
