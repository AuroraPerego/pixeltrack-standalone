#ifndef HeterogeneousCore_SYCLUtilities_interface_HistoContainer_h
#define HeterogeneousCore_SYCLUtilities_interface_HistoContainer_h

#include <CL/sycl.hpp>
#include <algorithm>
#ifndef __CUDA_ARCH__ 
#include <atomic>
#endif  // __CUDA_ARCH__
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "SYCLCore/AtomicPairCounter.h"
//#include "SYCLCore/syclCheck.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclstdAlgorithm.h"
#include "SYCLCore/prefixScan.h"

namespace cms {
  namespace sycltools {

    template <typename Histo, typename T>
    void countFromVector(Histo *__restrict__ h,
                         uint32_t nh,
                         T const *__restrict__ v,
                         uint32_t const *__restrict__ offsets,
                         sycl::nd_item<3> item) {
      int first = item.get_local_range().get(2) * item.get_group(2) + item.get_local_id(2);
      for (int i = first, nt = offsets[nh]; i < nt;
           i += item.get_group_range(2) * item.get_local_range().get(2)) {
        auto off = sycl_std::upper_bound(offsets, offsets + nh + 1, i);
        assert((*off) > 0);
        int32_t ih = off - offsets - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        (*h).count(v[i], ih);
      }
    }

    template <typename Histo, typename T>
    void fillFromVector(Histo *__restrict__ h,
                        uint32_t nh,
                        T const *__restrict__ v,
                        uint32_t const *__restrict__ offsets,
                        sycl::nd_item<3> item) {
      int first = item.get_local_range().get(2) * item.get_group(2) + item.get_local_id(2);
      for (int i = first, nt = offsets[nh]; i < nt;
           i += item.get_group_range(2) * item.get_local_range().get(2)) {
        auto off = sycl_std::upper_bound(offsets, offsets + nh + 1, i);
        assert((*off) > 0);
        int32_t ih = off - offsets - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        (*h).fill(v[i], i, ih);
      }
    }

    template <typename Histo>
    inline __attribute__((always_inline)) void launchZero(Histo *__restrict__ h,
                                                          sycl::queue stream
#ifndef SYCL_LANGUAGE_VERSION
                                                          = sycl::queue(sycl::default_selector{})
#endif
    ) {
      uint32_t *poff = (uint32_t *)((char *)(h) + offsetof(Histo, off));
      int32_t size = offsetof(Histo, bins) - offsetof(Histo, off);
      assert(size >= int(sizeof(uint32_t) * Histo::totbins()));
#ifdef SYCL_LANGUAGE_VERSION
      stream.memset(poff, 0, size);
#else
      ::memset(poff, 0, size);
#endif
    }

    template <typename Histo>
    inline __attribute__((always_inline)) void launchFinalize(Histo *__restrict__ h,
                                                              sycl::queue stream
#ifndef SYCL_LANGUAGE_VERSION
                                                              = sycl::queue(sycl::default_selector{})
#endif
    ) {
#ifdef SYCL_LANGUAGE_VERSION
      uint32_t *poff = (uint32_t *)((char *)(h) + offsetof(Histo, off));
      int32_t *ppsws = (int32_t *)((char *)(h) + offsetof(Histo, psws));
      auto nthreads = 1024;
      auto nblocks = (Histo::totbins() + nthreads - 1) / nthreads;
      stream.submit([&](sycl::handler &cgh) {
          sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              local_psum_acc(sycl::range<1>(sizeof(int32_t) * nblocks), cgh);
          sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ws_acc(sycl::range<1>(32), cgh); //FIXME_ why 32? 
          sycl::accessor<bool, 0, sycl::access_mode::read_write, sycl::access::target::local>
              isLastBlockDone_acc(cgh);

          auto Histo_totbins_kernel = Histo::totbins();

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * sycl::range<3>(1, 1, nthreads), sycl::range<3>(1, 1, nthreads)),
              [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                    multiBlockPrefixScan(poff,
                                          poff,
                                          Histo_totbins_kernel,
                                          ppsws,
                                          item,
                                          (uint8_t *)local_psum_acc.get_pointer(),
                                          (uint32_t *)ws_acc.get_pointer(),
                                          isLastBlockDone_acc.get_pointer());
              });
                  });
      //cudaCheck(0);
#else
      h->finalize();
#endif
    }

    template <typename Histo, typename T>
    inline __attribute__((always_inline)) void fillManyFromVector(Histo *__restrict__ h,
                                                                  uint32_t nh,
                                                                  T const *__restrict__ v,
                                                                  uint32_t const *__restrict__ offsets,
                                                                  uint32_t totSize,
                                                                  int nthreads,
                                                                  sycl::queue *stream
#ifndef SYCL_LANGUAGE_VERSION
                                                                  = sycl::queue(sycl::default_selector())
#endif
    ) {
      launchZero(h, stream);
#ifdef SYCL_LANGUAGE_VERSION
      auto nblocks = (totSize + nthreads - 1) / nthreads;
      stream.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) * sycl::range<3>(1, 1, nthreads), sycl::range<3>(1, 1, nthreads)),
                          [=](sycl::nd_item<3> item) {
                                countFromVector(h, nh, v, offsets, item);
                          });
      //cudaCheck(0);
      launchFinalize(h, stream);
      stream.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) * sycl::range<3>(1, 1, nthreads), sycl::range<3>(1, 1, nthreads)),
                          [=](sycl::nd_item<3> item) {
                                fillFromVector(h, nh, v, offsets, item);
                          });
      //cudaCheck(0);
#else
      countFromVector(h, nh, v, offsets);
      h->finalize();
      fillFromVector(h, nh, v, offsets);
#endif
    }

    template <typename Assoc>
    void finalizeBulk(AtomicPairCounter const *apc, Assoc *__restrict__ assoc) {
      assoc->bulkFinalizeFill(*apc);
    }

    // iteratate over N bins left and right of the one containing "v"
    template <typename Hist, typename V, typename Func>
    __forceinline void forEachInBins(Hist const &hist, V value, int n, Func func) {
      int bs = Hist::bin(value);
      int be = std::min(int(Hist::nbins() - 1), bs + n);
      bs = std::max(0, bs - n);
      assert(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }

    // iteratate over bins containing all values in window wmin, wmax
    template <typename Hist, typename V, typename Func>
    __forceinline void forEachInWindow(Hist const &hist, V wmin, V wmax, Func const &func) {
      auto bs = Hist::bin(wmin);
      auto be = Hist::bin(wmax);
      assert(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }

    template <typename T,                  // the type of the discretized input values
              uint32_t NBINS,              // number of bins
              uint32_t SIZE,               // max number of element
              uint32_t S = sizeof(T) * 8,  // number of significant bits in T
              typename I = uint32_t,  // type stored in the container (usually an index in a vector of the input values)
              uint32_t NHISTS = 1     // number of histos stored
              >
    class HistoContainer {
    public:
      using Counter = uint32_t;

      using CountersOnly = HistoContainer<T, NBINS, 0, S, I, NHISTS>;

      using index_type = I;
      using UT = typename std::make_unsigned<T>::type;

      static constexpr uint32_t ilog2(uint32_t v) {
        constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
        constexpr uint32_t s[] = {1, 2, 4, 8, 16};

        uint32_t r = 0;  // result of log2(v) will go here
        for (auto i = 4; i >= 0; i--)
          if (v & b[i]) {
            v >>= s[i];
            r |= s[i];
          }
        return r;
      }

      static constexpr uint32_t sizeT() { return S; }
      static constexpr uint32_t nbins() { return NBINS; }
      static constexpr uint32_t nhists() { return NHISTS; }
      static constexpr uint32_t totbins() { return NHISTS * NBINS + 1; }
      static constexpr uint32_t nbits() { return ilog2(NBINS - 1) + 1; }
      static constexpr uint32_t capacity() { return SIZE; }

      static constexpr auto histOff(uint32_t nh) { return NBINS * nh; }

      static constexpr UT bin(T t) {
        constexpr uint32_t shift = sizeT() - nbits();
        constexpr uint32_t mask = (1 << nbits()) - 1;
        return (t >> shift) & mask;
      }

      void zero() {
        for (auto &i : off)
          i = 0;
      }

      //analog of cuda atomicAdd
      template T <typename T>
      inline T atomicAdd(T* i, T* j){
        return sycl::atomic<T>(sycl::global_ptr<T>(i)).fetch_add(j);
      }

      __forceinline void add(CountersOnly const &co) {
        for (uint32_t i = 0; i < totbins(); ++i) {
#ifdef __CUDA_ARCH__
          atomicAdd<uint32_t>(off +i, co.off[i]);
#else
          auto &a = (std::atomic<Counter> &)(off[i]);
          a += co.off[i];
#endif
        }
      }

      static __forceinline uint32_t atomicIncrement(Counter &x) {
#ifdef __CUDA_ARCH__
        return atomicAdd<cms::sycltools::HistoContainer<T, NBINS, SIZE, S, I, NHISTS>::Counter>>(&x, 1);
#else
        auto &a = (std::atomic<Counter> &)(x);
        return a++;
#endif
      }

      static __forceinline uint32_t atomicDecrement(Counter &x) {
#ifdef __CUDA_ARCH__
        return atomicAdd<cms::sycltools::HistoContainer<T, NBINS, SIZE, S, I, NHISTS>::Counter>>(&x, -1);
#else
        auto &a = (std::atomic<Counter> &)(x);
        return a--;
#endif
      }

      __forceinline void countDirect(T b) {
        assert(b < nbins());
        atomicIncrement(off[b]);
      }

      __forceinline void fillDirect(T b, index_type j) {
        assert(b < nbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

      __forceinline int32_t bulkFill(AtomicPairCounter &apc, index_type const *v, uint32_t n) {
        auto c = apc.add(n);
        if (c.m >= nbins())
          return -int32_t(c.m);
        off[c.m] = c.n;
        for (uint32_t j = 0; j < n; ++j)
          bins[c.n + j] = v[j];
        return c.m;
      }

      __forceinline void bulkFinalize(AtomicPairCounter const &apc) {
        off[apc.get().m] = apc.get().n;
      }

      __forceinline void bulkFinalizeFill(AtomicPairCounter const &apc, sycl::nd_item<3> item) {
        auto m = apc.get().m;
        auto n = apc.get().n;
        if (m >= nbins()) {  // overflow!
          off[nbins()] = uint32_t(off[nbins() - 1]);
          return;
        }
        auto first = m + item.get_local_range().get(2) * item.get_group(2) + item.get_local_id(2);
        for (auto i = first; i < totbins(); i += item.get_group_range(2) * item.get_local_range().get(2)) {
          off[i] = n;
        }
      }

      __forceinline void count(T t) {
        uint32_t b = bin(t);
        assert(b < nbins());
        atomicIncrement(off[b]);
      }

      __forceinline void fill(T t, index_type j) {
        uint32_t b = bin(t);
        assert(b < nbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

      __forceinline void count(T t, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        atomicIncrement(off[b]);
      }

      __forceinline void fill(T t, index_type j, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

      __forceinline void finalize(sycl::nd_item<3> item, Counter *ws = nullptr) {
        assert(off[totbins() - 1] == 0);
        blockPrefixScan(off, totbins(), ws, item);
        assert(off[totbins() - 1] == off[totbins() - 2]);
      }

      constexpr auto size() const { return uint32_t(off[totbins() - 1]); }
      constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

      constexpr index_type const *begin() const { return bins; }
      constexpr index_type const *end() const { return begin() + size(); }

      constexpr index_type const *begin(uint32_t b) const { return bins + off[b]; }
      constexpr index_type const *end(uint32_t b) const { return bins + off[b + 1]; }

      Counter off[totbins()];
      int32_t psws;  // prefix-scan working space
      index_type bins[capacity()];
    };

    template <typename I,        // type stored in the container (usually an index in a vector of the input values)
              uint32_t MAXONES,  // max number of "ones"
              uint32_t MAXMANYS  // max number of "manys"
              >
    using OneToManyAssoc = HistoContainer<uint32_t, MAXONES, MAXMANYS, sizeof(uint32_t) * 8, I, 1>;

  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_SYCLUtilities_interface_HistoContainer_h
