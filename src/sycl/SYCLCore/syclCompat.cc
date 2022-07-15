#include "SYCLCore/syclCompat.h"

namespace cms {
  namespace syclcompat {
    sycl::range<3> blockIdx;
    sycl::range<3> gridDim;
  }  // namespace syclcompat
}  // namespace cms

namespace {
  struct InitGrid {
    InitGrid() { cms::syclcompat::resetGrid(); }
  };

  const InitGrid initGrid;

}  // namespace
