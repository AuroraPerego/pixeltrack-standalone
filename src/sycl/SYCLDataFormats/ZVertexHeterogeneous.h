#ifndef SYCLDataFormatsVertexZVertexHeterogeneous_H
#define SYCLDataFormatsVertexZVertexHeterogeneous_H

#include "SYCLDataFormats/ZVertexSoA.h"
#include "SYCLDataFormats/HeterogeneousSoA.h"
#include "SYCLDataFormats/PixelTrackHeterogeneous.h"

using ZVertexHeterogeneous = HeterogeneousSoA<ZVertexSoA>;
//TODO_ since this is the only line, consider moving it in HeterogeneousSoA.h

//#ifndef SYCL_LANGUAGE_VERSION TODO_
//#include "SYCLCore/Product.h"
//using ZVertexSYCLProduct = cms::sycltools::Product<ZVertexHeterogeneous>;
//#endif

#endif
