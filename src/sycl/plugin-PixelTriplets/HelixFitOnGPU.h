#ifndef RecoPixelVertexing_PixelTrackFitting_plugins_HelixFitOnGPU_h
#define RecoPixelVertexing_PixelTrackFitting_plugins_HelixFitOnGPU_h

#include "SYCLDataFormats/PixelTrackHeterogeneous.h"
#include "SYCLDataFormats/TrackingRecHit2DSYCL.h"
#include "SYCLCore/device_unique_ptr.h"

#include "RiemannFitOnGPU.h"
#include "CAConstants.h"
#include "FitResult.h"

namespace Rfit {
  // in case of memory issue can be made smaller
  constexpr uint32_t maxNumberOfConcurrentFits() { return CAConstants::maxNumberOfTuples(); }
  constexpr uint32_t stride() { return maxNumberOfConcurrentFits(); }
  using Matrix3x4d = Eigen::Matrix<double, 3, 4>;
  using Map3x4d = Eigen::Map<Matrix3x4d, 0, Eigen::Stride<3 * stride(), stride()> >;
  using Matrix6x4f = Eigen::Matrix<float, 6, 4>;
  using Map6x4f = Eigen::Map<Matrix6x4f, 0, Eigen::Stride<6 * stride(), stride()> >;

  // hits
  template <int N>
  using Matrix3xNd = Eigen::Matrix<double, 3, N>;
  template <int N>
  using Map3xNd = Eigen::Map<Matrix3xNd<N>, 0, Eigen::Stride<3 * stride(), stride()> >;
  // errors
  template <int N>
  using Matrix6xNf = Eigen::Matrix<float, 6, N>;
  template <int N>
  using Map6xNf = Eigen::Map<Matrix6xNf<N>, 0, Eigen::Stride<6 * stride(), stride()> >;
  // fast fit
  using Map4d = Eigen::Map<Vector4d, 0, Eigen::InnerStride<stride()> >;

}  // namespace Rfit

class HelixFitOnGPU {
public:
  using HitsView = TrackingRecHit2DSOAView;

  using Tuples = pixelTrack::HitContainer;
  using OutputSoA = pixelTrack::TrackSoA;

  using TupleMultiplicity = CAConstants::TupleMultiplicity;

  explicit HelixFitOnGPU(float bf, bool fit5as4) : bField_(bf), fit5as4_(fit5as4) {}
  ~HelixFitOnGPU() { deallocateOnGPU(); }

  void setBField(double bField) { bField_ = bField; }
  void launchRiemannKernels(HitsView const *hv,
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
      auto tuples_d_kernel = tuples_d;
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
      auto hv_kernel = hv;
      auto hitsGPU_kernel = hitsGPU_.get(); 
      auto hits_geGPU_kernel = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelFastFit<3>(tuples_d_kernel, 
                                 tupleMultiplicity_d_kernel, 
                                 3, 
                                 hv_kernel, 
                                 hitsGPU_kernel, 
                                 hits_geGPU_kernel, 
                                 fast_fit_resultsGPU_kernel, 
                                 offset, 
                                 item);
      });
    });
    //cudaCheck(cudaGetLastError());
    
    stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
	auto bField_kernel = bField_;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
      auto hitsGPU_kernel = hitsGPU_.get(); 
      auto hits_geGPU_kernel = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelCircleFit<3>(tupleMultiplicity_d_kernel,
                                   3,
                                   bField_kernel,
                                   hitsGPU_kernel,
                                   hits_geGPU_kernel,
                                   fast_fit_resultsGPU_kernel,
                                   circle_fit_resultsGPU_kernel,
                                   offset, 
                                   item);
      });
    });
    //cudaCheck(cudaGetLastError());

    stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
	auto bField_kernel = bField_;
	auto outputSoa_d_kernel = outputSoa_d;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
      auto hitsGPU_kernel = hitsGPU_.get(); 
      auto hits_geGPU_kernel = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelLineFit<3>(tupleMultiplicity_d_kernel,
                                 3,
                                 bField_kernel,
                                 outputSoa_d_kernel,
                                 hitsGPU_kernel,
                                 hits_geGPU_kernel,
                                 fast_fit_resultsGPU_kernel,
                                 circle_fit_resultsGPU_kernel,
                                 offset, 
                                 item);
      });
    });
    //cudaCheck(cudaGetLastError());

    // quads
    stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel = tuples_d;
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
      auto hv_kernel = hv;
      auto hitsGPU_kernel = hitsGPU_.get(); 
      auto hits_geGPU_kernel = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelFastFit<4>(tuples_d_kernel, 
                                 tupleMultiplicity_d_kernel, 
                                 4, 
                                 hv_kernel, 
                                 hitsGPU_kernel, 
                                 hits_geGPU_kernel, 
                                 fast_fit_resultsGPU_kernel, 
                                 offset, 
                                 item);
      });
    });
    //cudaCheck(cudaGetLastError());
    
    stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
	auto bField_kernel = bField_;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
      auto hitsGPU_kernel = hitsGPU_.get(); 
      auto hits_geGPU_kernel = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelCircleFit<4>(tupleMultiplicity_d_kernel,
                                   4,
                                   bField_kernel,
                                   hitsGPU_kernel,
                                   hits_geGPU_kernel,
                                   fast_fit_resultsGPU_kernel,
                                   circle_fit_resultsGPU_kernel,
                                   offset, 
                                   item);
      });
    });
    //cudaCheck(cudaGetLastError());

    stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
	auto bField_kernel = bField_;
	auto outputSoa_d_kernel = outputSoa_d;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
      auto hitsGPU_kernel = hitsGPU_.get(); 
      auto hits_geGPU_kernel = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelLineFit<4>(tupleMultiplicity_d_kernel,
                                 4,
                                 bField_kernel,
                                 outputSoa_d_kernel,
                                 hitsGPU_kernel,
                                 hits_geGPU_kernel,
                                 fast_fit_resultsGPU_kernel,
                                 circle_fit_resultsGPU_kernel,
                                 offset, 
                                 item);
      });
    });
    //cudaCheck(cudaGetLastError());
    
    if (fit5as4_) {
      // penta
      stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel = tuples_d;
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
      auto hv_kernel = hv;
      auto hitsGPU_kernel = hitsGPU_.get(); 
      auto hits_geGPU_kernel = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for(
          sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item){ 
                kernelFastFit<4>(tuples_d_kernel, 
                                 tupleMultiplicity_d_kernel, 
                                 5, 
                                 hv_kernel, 
                                 hitsGPU_kernel, 
                                 hits_geGPU_kernel, 
                                 fast_fit_resultsGPU_kernel, 
                                 offset, 
                                 item);
        });
      });
      //cudaCheck(cudaGetLastError());
      
      stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
	auto bField_kernel = bField_;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
        auto hitsGPU_kernel = hitsGPU_.get(); 
        auto hits_geGPU_kernel = hits_geGPU_.get(); 
        auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
        cgh.parallel_for(
            sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
            [=](sycl::nd_item<3> item){ 
                  kernelCircleFit<4>(tupleMultiplicity_d_kernel,
                                     5,
                                     bField_kernel,
                                     hitsGPU_kernel,
                                     hits_geGPU_kernel,
                                     fast_fit_resultsGPU_kernel,
                                     circle_fit_resultsGPU_kernel,
                                     offset, 
                                     item);
        });
      });
      //cudaCheck(cudaGetLastError());
  
      stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
	auto bField_kernel = bField_;
	auto outputSoa_d_kernel = outputSoa_d;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
        auto hitsGPU_kernel = hitsGPU_.get(); 
        auto hits_geGPU_kernel = hits_geGPU_.get(); 
        auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
        cgh.parallel_for(
            sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
            [=](sycl::nd_item<3> item){ 
                  kernelLineFit<4>(tupleMultiplicity_d_kernel,
                                   5,
                                   bField_kernel,
                                   outputSoa_d_kernel,
                                   hitsGPU_kernel,
                                   hits_geGPU_kernel,
                                   fast_fit_resultsGPU_kernel,
                                   circle_fit_resultsGPU_kernel,
                                   offset, 
                                   item);
        });
      });
      //cudaCheck(cudaGetLastError());

    } else {
      // penta all 5
      stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel = tuples_d;
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
      auto hv_kernel = hv;
      auto hitsGPU_kernel = hitsGPU_.get(); 
      auto hits_geGPU_kernel = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for(
            sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
            [=](sycl::nd_item<3> item){ 
                  kernelFastFit<5>(tuples_d_kernel, 
                                   tupleMultiplicity_d_kernel, 
                                   5, 
                                   hv_kernel, 
                                   hitsGPU_kernel, 
                                   hits_geGPU_kernel, 
                                   fast_fit_resultsGPU_kernel, 
                                   offset, 
                                   item);
          });
      });
      //cudaCheck(cudaGetLastError());
      
      stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
	auto bField_kernel = bField_;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
        auto hitsGPU_kernel = hitsGPU_.get(); 
      auto hits_geGPU_kernel = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for(
            sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
            [=](sycl::nd_item<3> item){ 
                  kernelCircleFit<5>(tupleMultiplicity_d_kernel,
                                     5,
                                     bField_kernel,
                                     hitsGPU_kernel,
                                     hits_geGPU_kernel,
                                     fast_fit_resultsGPU_kernel,
                                     circle_fit_resultsGPU_kernel,
                                     offset, 
                                     item);
        });
      });
      //cudaCheck(cudaGetLastError());
  
      stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
	auto bField_kernel = bField_;
	auto outputSoa_d_kernel = outputSoa_d;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
        auto hitsGPU_kernel = hitsGPU_.get(); 
      auto hits_geGPU_kernel = hits_geGPU_.get(); 
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for(
            sycl::nd_range<3>(numberOfBlocks / 4 * sycl::range<3>(1, 1, blockSize), sycl::range<3>(1, 1, blockSize)),
            [=](sycl::nd_item<3> item){ 
                  kernelLineFit<5>(tupleMultiplicity_d_kernel,
                                   5,
                                   bField_kernel,
                                   outputSoa_d_kernel,
                                   hitsGPU_kernel,
                                   hits_geGPU_kernel,
                                   fast_fit_resultsGPU_kernel,
                                   circle_fit_resultsGPU_kernel,
                                   offset, 
                                   item);
        });
      });
    }
  }
};
  void launchBrokenLineKernels(HitsView const *hv, uint32_t nhits, uint32_t maxNumberOfTuples, sycl::queue stream);

  void allocateOnGPU(Tuples const *tuples, TupleMultiplicity const *tupleMultiplicity, OutputSoA *outputSoA);
  void deallocateOnGPU();

private:
  static constexpr uint32_t maxNumberOfConcurrentFits_ = Rfit::maxNumberOfConcurrentFits();

  // fowarded
  Tuples const *tuples_d = nullptr;
  TupleMultiplicity const *tupleMultiplicity_d = nullptr;
  OutputSoA *outputSoa_d;
  float bField_;

  const bool fit5as4_;
};

#endif  // RecoPixelVertexing_PixelTrackFitting_plugins_HelixFitOnGPU_h

