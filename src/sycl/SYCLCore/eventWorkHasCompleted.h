#ifndef HeterogeneousCore_SYCLUtilities_eventWorkHasCompleted_h
#define HeterogeneousCore_SYCLUtilities_eventWorkHasCompleted_h

//#include "SYCLCore/syclCheck.h"

#include <CL/sycl.hpp>

namespace cms {
  namespace sycltools {
    /**
   * Returns true if the work captured by the event (=queued to the
   * CUDA stream at the point of cudaEventRecord()) has completed.
   *
   * Returns false if any captured work is incomplete.
   *
   * In case of errors, throws an exception.
   */
    inline bool eventWorkHasCompleted(sycl::event* event) {
    sycl::event e;
    const auto ret = e.get_info<sycl::info::event::command_execution_status>();

      if (ret == sycl::info::event_command_status::complete) { //FIXME_ test and see if ret is int or string
        return true;
      } else { // the other possibilities are submitted, running

        return false;
      }
      // leave error case handling to cudaCheck
      //cudaCheck(ret);
      return false;  // to keep compiler happy
    }
  }  // namespace sycltools
}  // namespace cms

#endif
