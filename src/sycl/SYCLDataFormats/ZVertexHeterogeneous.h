#ifndef SYCLDataFormatsVertexZVertexHeterogeneous_H
#define SYCLDataFormatsVertexZVertexHeterogeneous_H

#include "SYCLDataFormats/ZVertexSoA.h"
#include "SYCLDataFormats/HeterogeneousSoA.h"
#include "SYCLDataFormats/PixelTrackHeterogeneous.h"

using ZVertexHeterogeneous = HeterogeneousSoA<ZVertexSoA>;
#ifndef SYCL_LANGUAGE_VERSION
#include "SYCLCore/Product.h"
using ZVertexSYCLProduct = cms::cuda::Product<ZVertexHeterogeneous>;
#endif

#endif
