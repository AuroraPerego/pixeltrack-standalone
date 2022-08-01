#ifndef SYCLDataFormatsVertexZVertexHeterogeneous_H
#define SYCLDataFormatsVertexZVertexHeterogeneous_H

#include "SYCLDataFormats/ZVertexSoA.h"
#include "SYCLDataFormats/HeterogeneousSoA.h"
#include "SYCLDataFormats/PixelTrackHeterogeneous.h"

using ZVertexHeterogeneous = HeterogeneousSoA<ZVertexSoA>;
#ifndef SYCL_LANGUAGE_VERSION
#endif
#include "SYCLCore/Product.h"
using ZVertexSYCLProduct = cms::sycltools::Product<ZVertexHeterogeneous>;

#endif
