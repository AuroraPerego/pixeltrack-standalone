#include "BrokenLineFitOnGPU.h"
#include "SYCLCore/device_unique_ptr.h"

void HelixFitOnGPU::launchBrokenLineKernels(HitsView const *hv,
                                            uint32_t hitsInFit,
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

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    stream.submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> done_acc(sycl::range<1>(sizeof(int)), cgh);       
      auto tuples_d_kernel            = tuples_d; 
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d; 
      auto hv_kernel                  = hv;
      auto hitsGPU_kernel             = hitsGPU_.get(); 
      auto hits_geGPU_kernel          = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
                kernelBLFastFit<3>(tuples_d_kernel, 
                                   tupleMultiplicity_d_kernel, 
                                   hv_kernel, 
                                   hitsGPU_kernel, 
                                   hits_geGPU_kernel, 
                                   fast_fit_resultsGPU_kernel, 
                                   3, 
                                   offset,
                                   item,
                                   (int *)done_acc.get_pointer());
      });
    });

    if ((stream.get_device()).is_cpu())
      stream.wait();
    
    stream.submit([&](sycl::handler &cgh) {
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d; 
      auto bField_kernel              = bField_;
      auto hitsGPU_kernel             = hitsGPU_.get(); 
      auto hits_geGPU_kernel          = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      auto outputSoa_d_kernel         = outputSoa_d;

      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
                kernelBLFit<3>(tupleMultiplicity_d_kernel,
                               bField_kernel,
                               outputSoa_d_kernel,
                               hitsGPU_kernel,
                               hits_geGPU_kernel,
                               fast_fit_resultsGPU_kernel,
                               3,
                               offset,
                               item);
      });
    });
    
    if ((stream.get_device()).is_cpu())
      stream.wait();

    // fit quads
    stream.submit([&](sycl::handler &cgh) {
     sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> done_acc(sycl::range<1>(sizeof(int)), cgh);       
      auto tuples_d_kernel            = tuples_d; 
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d; 
      auto hv_kernel                  = hv;
      auto hitsGPU_kernel             = hitsGPU_.get(); 
      auto hits_geGPU_kernel          = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();

      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
                kernelBLFastFit<4>(tuples_d_kernel, 
                                   tupleMultiplicity_d_kernel, 
                                   hv_kernel, 
                                   hitsGPU_kernel, 
                                   hits_geGPU_kernel, 
                                   fast_fit_resultsGPU_kernel, 
                                   4, 
                                   offset,
                                   item,
                                   (int *)done_acc.get_pointer());
      });
    });
    if ((stream.get_device()).is_cpu())
      stream.wait();

    stream.submit([&](sycl::handler &cgh) {
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d; 
      auto bField_kernel              = bField_;
      auto hitsGPU_kernel             = hitsGPU_.get(); 
      auto hits_geGPU_kernel          = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      auto outputSoa_d_kernel         = outputSoa_d;

      cgh.parallel_for(
          sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
          [=](sycl::nd_item<1> item){ 
                kernelBLFit<4>(tupleMultiplicity_d_kernel,
                               bField_kernel,
                               outputSoa_d_kernel,
                               hitsGPU_kernel,
                               hits_geGPU_kernel,
                               fast_fit_resultsGPU_kernel,
                               4,
                               offset,
                               item);
      });
    });
    if ((stream.get_device()).is_cpu())
      stream.wait();

    if (fit5as4_) {
      // fit penta (only first 4)
      stream.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> done_acc(sycl::range<1>(sizeof(int)), cgh);       
        auto tuples_d_kernel            = tuples_d; 
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d; 
        auto hv_kernel                  = hv;
        auto hitsGPU_kernel             = hitsGPU_.get(); 
        auto hits_geGPU_kernel          = hits_geGPU_.get(); 
        auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
  
        cgh.parallel_for(
            sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
            [=](sycl::nd_item<1> item){ 
                  kernelBLFastFit<4>(tuples_d_kernel, 
                                   tupleMultiplicity_d_kernel, 
                                   hv_kernel, 
                                   hitsGPU_kernel, 
                                   hits_geGPU_kernel, 
                                   fast_fit_resultsGPU_kernel, 
                                   5, 
                                   offset,
                                   item,
                                   (int *)done_acc.get_pointer());
          });
        });
    if ((stream.get_device()).is_cpu())
      stream.wait();
    
    stream.submit([&](sycl::handler &cgh) {
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d; 
      auto bField_kernel              = bField_;
      auto hitsGPU_kernel             = hitsGPU_.get(); 
      auto hits_geGPU_kernel          = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      auto outputSoa_d_kernel         = outputSoa_d;
  
        cgh.parallel_for(
            sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
            [=](sycl::nd_item<1> item){ 
                  kernelBLFit<4>(tupleMultiplicity_d_kernel,
                               bField_kernel,
                               outputSoa_d_kernel,
                               hitsGPU_kernel,
                               hits_geGPU_kernel,
                               fast_fit_resultsGPU_kernel,
                               5,
                               offset,
                               item);
        });
      });
    if ((stream.get_device()).is_cpu())
      stream.wait();

    } else {
      // fit penta (all 5)
      stream.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> done_acc(sycl::range<1>(sizeof(int)), cgh);       
        auto tuples_d_kernel            = tuples_d; 
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d; 
        auto hv_kernel                  = hv;
        auto hitsGPU_kernel             = hitsGPU_.get(); 
        auto hits_geGPU_kernel          = hits_geGPU_.get(); 
        auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
  
        cgh.parallel_for(
            sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
            [=](sycl::nd_item<1> item){ 
                  kernelBLFastFit<5>(tuples_d_kernel, 
                                   tupleMultiplicity_d_kernel, 
                                   hv_kernel, 
                                   hitsGPU_kernel, 
                                   hits_geGPU_kernel, 
                                   fast_fit_resultsGPU_kernel, 
                                   5, 
                                   offset,
                                   item,
                                   (int *)done_acc.get_pointer());
          });
        });
      if ((stream.get_device()).is_cpu())
        stream.wait();

      stream.submit([&](sycl::handler &cgh) {
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d; 
      auto bField_kernel              = bField_;
      auto hitsGPU_kernel             = hitsGPU_.get(); 
      auto hits_geGPU_kernel          = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      auto outputSoa_d_kernel         = outputSoa_d;
  
        cgh.parallel_for(
            sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
            [=](sycl::nd_item<1> item){ 
                  kernelBLFit<5>(tupleMultiplicity_d_kernel,
                               bField_kernel,
                               outputSoa_d_kernel,
                               hitsGPU_kernel,
                               hits_geGPU_kernel,
                               fast_fit_resultsGPU_kernel,
                               5,
                               offset,
                               item);
        });
      });
    if ((stream.get_device()).is_cpu())
      stream.wait();
    }
  }  // loop on concurrent fits
}

