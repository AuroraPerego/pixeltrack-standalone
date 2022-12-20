#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <type_traits>

#include "CUDACore/cudaCheck.h"
#include "CUDACore/launch.h"
#include "CUDACore/radixSort.h"

using namespace cms::cuda;

template <typename T>
struct RS {
  using type = std::uniform_int_distribution<T>;
  static auto ud() { return type(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()); }
  static constexpr T imax = std::numeric_limits<T>::max();
};

template <>
struct RS<float> {
  using T = float;
  using type = std::uniform_real_distribution<float>;
  static auto ud() { return type(-std::numeric_limits<T>::max() / 2, std::numeric_limits<T>::max() / 2); }
  //  static auto ud() { return type(0,std::numeric_limits<T>::max()/2);}
  static constexpr int imax = std::numeric_limits<int>::max();
};

// A templated unsigned integer type with N bytes
template <int N>
struct uintN;

template <>
struct uintN<8> {
  using type = uint8_t;
};

template <>
struct uintN<16> {
  using type = uint16_t;
};

template <>
struct uintN<32> {
  using type = uint32_t;
};

template <>
struct uintN<64> {
  using type = uint64_t;
};

template <int N>
using uintN_t = typename uintN<N>::type;

// A templated unsigned integer type with the same size as T
template <typename T>
using uintT_t = uintN_t<sizeof(T) * 8>;

// Keep only the `N` most significant bytes of `t`, and set the others to zero
template <int N, typename T, typename SFINAE = std::enable_if_t<N <= sizeof(T)>>
void truncate(T& t) {
  const int shift = 8 * (sizeof(T) - N);
  union {
    T t;
    uintT_t<T> u;
  } c;
  c.t = t;
  c.u = c.u >> shift << shift;
  t = c.t;
}

template <typename T, int NS = sizeof(T), typename U = T, typename LL = long long>
void go() {

  constexpr int blocks = 10;
  constexpr int blockSize = 256;
  constexpr int N = blockSize * blocks;
  T v[N];
  uint16_t ind[N];

  constexpr bool sgn = T(-1) < T(0);
  std::cout << "Will sort " << N << (sgn ? " signed" : " unsigned")
            << (std::numeric_limits<T>::is_integer ? " 'ints'" : " 'float'") << " of size " << sizeof(T) << " using "
            << NS << " significant bytes" << std::endl;

      uint64_t imax = uint64_t(RS<T>::imax) + 1LL;
      for (uint64_t j = 0; j < N; j++) {
        v[j] = (j % imax);
        if (j % 2)
          v[j] = -v[j];
      }
    uint32_t offsets[blocks + 1];
    offsets[0] = 0;
    for (int j = 1; j < blocks + 1; ++j) {
      offsets[j] = offsets[j - 1] + blockSize - 3 * j;
      assert(offsets[j] <= N);
    }
    std::random_shuffle(v, v + N);

    T* v_d;
    uint16_t *ind_d, *ws_d;
    uint32_t* off_d;
    cudaCheck(cudaMalloc((void**)(&v_d), N * sizeof(U)));
    cudaCheck(cudaMalloc((void**)(&ind_d), N * sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void**)(&ws_d), N * sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void**)(&off_d), (blocks + 1) * sizeof(uint32_t)));
    cudaCheck(cudaMemcpy(v_d, v, N * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(off_d, offsets, 4 * (blocks + 1), cudaMemcpyHostToDevice));

      std::cout << "launch for " << offsets[blocks] << std::endl;

    auto ntXBl __attribute__((unused)) = 1 == 0 % 4 ? 256 : 256;

    constexpr int MaxSize = 256;
      cms::cuda::launch(
          radixSortMultiWrapper<U, NS>, {blocks, ntXBl, MaxSize * 2}, v_d, ind_d, off_d, nullptr);

    cudaCheck(cudaMemcpy(ind, ind_d, 2 * N, cudaMemcpyDeviceToHost));


      std::cout << "done for " << offsets[blocks] << std::endl;
    for (int ib = 0; ib < blocks; ++ib) {
      std::set<uint16_t> inds;
      if (offsets[ib + 1] > offsets[ib])
        inds.insert(ind[offsets[ib]]);
      for (auto j = offsets[ib] + 1; j < offsets[ib + 1]; j++) {
        inds.insert(ind[j]);
        auto a = v + offsets[ib];
        auto k1 = a[ind[j]];
        auto k2 = a[ind[j - 1]];
        truncate<NS>(k1);
        truncate<NS>(k2);
        if (k1 < k2)
          std::cout << ib << " not ordered at " << ind[j] << " : " << a[ind[j]] << ' ' << a[ind[j - 1]] << std::endl;
      }
      if (!inds.empty()) {
        assert(0 == *inds.begin());
        assert(inds.size() - 1 == *inds.rbegin());
      }
      if (inds.size() != (offsets[ib + 1] - offsets[ib]))
        std::cout << "error " << ib << ' ' << inds.size() << "!=" << (offsets[ib + 1] - offsets[ib])
                  << std::endl;
      assert(inds.size() == (offsets[ib + 1] - offsets[ib]));
    }
    cudaFree(v_d);
    cudaFree(ind_d);
    cudaFree(ws_d);
    cudaFree(off_d);
  }

int main() {

  go<int32_t>();

  return 0;
}
