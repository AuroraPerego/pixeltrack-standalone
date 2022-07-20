#include "RiemannFitOnGPU.h"
#include "SYCLCore/device_unique_ptr.h"

void HelixFitOnGPU::launchRiemannKernels(HitsView const *hv,
                                         uint32_t nhits,
                                         uint32_t maxNumberOfTuples,
                                         sycl::queue stream) {
  assert(tuples_d);

  auto blockSize = 64;
  auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

  //  Fit internals
  auto hitsGPU_ = cms::sycltools::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double), stream);
  auto hits_geGPU_ = cms::sycltools::make_device_unique<float[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float), stream);
  auto fast_fit_resultsGPU_ = cms::sycltools::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double), stream);
  auto circle_fit_resultsGPU_holder =
      cms::sycltools::make_device_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit), stream);
  Rfit::circle_fit *circle_fit_resultsGPU_ = (Rfit::circle_fit *)(circle_fit_resultsGPU_holder.get());

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // triplets
    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelFastFit<3>(tuples_d, 
                                 tupleMultiplicity_d, 
                                 3, 
                                 hv, 
                                 hitsGPU_.get(), 
                                 hits_geGPU_.get(), 
                                 fast_fit_resultsGPU_.get(), 
                                 offset, 
                                 item);
      });
    });
    //cudaCheck(cudaGetLastError());
    
    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelCircleFit<3>(tupleMultiplicity_d,
                                   3,
                                   bField_,
                                   hitsGPU_.get(),
                                   hits_geGPU_.get(),
                                   fast_fit_resultsGPU_.get(),
                                   circle_fit_resultsGPU_,
                                   offset, 
                                   item);
      });
    });
    //cudaCheck(cudaGetLastError());

    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelLineFit<3>(tupleMultiplicity_d,
                                 3,
                                 bField_,
                                 outputSoa_d,
                                 hitsGPU_.get(),
                                 hits_geGPU_.get(),
                                 fast_fit_resultsGPU_.get(),
                                 circle_fit_resultsGPU_,
                                 offset, 
                                 item);
      });
    });
    //cudaCheck(cudaGetLastError());

    // quads
    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelFastFit<4>(tuples_d, 
                                 tupleMultiplicity_d, 
                                 4, 
                                 hv, 
                                 hitsGPU_.get(), 
                                 hits_geGPU_.get(), 
                                 fast_fit_resultsGPU_.get(), 
                                 offset, 
                                 item);
      });
    });
    //cudaCheck(cudaGetLastError());
    
    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelCircleFit<4>(tupleMultiplicity_d,
                                   4,
                                   bField_,
                                   hitsGPU_.get(),
                                   hits_geGPU_.get(),
                                   fast_fit_resultsGPU_.get(),
                                   circle_fit_resultsGPU_,
                                   offset, 
                                   item);
      });
    });
    //cudaCheck(cudaGetLastError());

    stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelLineFit<4>(tupleMultiplicity_d,
                                 4,
                                 bField_,
                                 outputSoa_d,
                                 hitsGPU_.get(),
                                 hits_geGPU_.get(),
                                 fast_fit_resultsGPU_.get(),
                                 circle_fit_resultsGPU_,
                                 offset, 
                                 item);
      });
    });
    //cudaCheck(cudaGetLastError());
    
    if (fit5as4_) {
      // penta
      stream.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelFastFit<4>(tuples_d, 
                                 tupleMultiplicity_d, 
                                 5, 
                                 hv, 
                                 hitsGPU_.get(), 
                                 hits_geGPU_.get(), 
                                 fast_fit_resultsGPU_.get(), 
                                 offset, 
                                 item);
        });
      });
      //cudaCheck(cudaGetLastError());
      
      stream.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
            [=](sycl::nd_item<3> item){ 
                  kernelCircleFit<4>(tupleMultiplicity_d,
                                     5,
                                     bField_,
                                     hitsGPU_.get(),
                                     hits_geGPU_.get(),
                                     fast_fit_resultsGPU_.get(),
                                     circle_fit_resultsGPU_,
                                     offset, 
                                     item);
        });
      });
      //cudaCheck(cudaGetLastError());
  
      stream.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
            [=](sycl::nd_item<3> item){ 
                  kernelLineFit<4>(tupleMultiplicity_d,
                                   5,
                                   bField_,
                                   outputSoa_d,
                                   hitsGPU_.get(),
                                   hits_geGPU_.get(),
                                   fast_fit_resultsGPU_.get(),
                                   circle_fit_resultsGPU_,
                                   offset, 
                                   item);
        });
      });
      //cudaCheck(cudaGetLastError());

    } else {
      // penta all 5
      stream.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
            [=](sycl::nd_item<3> item){ 
                  kernelFastFit<5>(tuples_d, 
                                   tupleMultiplicity_d, 
                                   5, 
                                   hv, 
                                   hitsGPU_.get(), 
                                   hits_geGPU_.get(), 
                                   fast_fit_resultsGPU_.get(), 
                                   offset, 
                                   item);
          });
      });
      //cudaCheck(cudaGetLastError());
      
      stream.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
            [=](sycl::nd_item<3> item){ 
                  kernelCircleFit<5>(tupleMultiplicity_d,
                                     5,
                                     bField_,
                                     hitsGPU_.get(),
                                     hits_geGPU_.get(),
                                     fast_fit_resultsGPU_.get(),
                                     circle_fit_resultsGPU_,
                                     offset, 
                                     item);
        });
      });
      //cudaCheck(cudaGetLastError());
  
      stream.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
            [=](sycl::nd_item<3> item){ 
                  kernelLineFit<5>(tupleMultiplicity_d,
                                   5,
                                   bField_,
                                   outputSoa_d,
                                   hitsGPU_.get(),
                                   hits_geGPU_.get(),
                                   fast_fit_resultsGPU_.get(),
                                   circle_fit_resultsGPU_,
                                   offset, 
                                   item);
        });
      });
    }
  }
}
