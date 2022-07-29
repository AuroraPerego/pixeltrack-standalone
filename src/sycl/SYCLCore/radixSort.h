#ifndef HeterogeneousCoreSYCLUtilities_radixSort_H
#define HeterogeneousCoreSYCLUtilities_radixSort_H

#ifdef SYCL_LANGUAGE_VERSION
#include <cstdint>
#include <type_traits>

#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/syclAtomic.h"

template <typename T>
inline void dummyReorder(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {}

template <typename T>
inline void reorderSigned(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size, sycl::nd_item<3> item, uint32_t * firstNeg) {
  //move negative first...

  int32_t first = item.get_local_id(2);
  *firstNeg = a[ind[0]] < 0 ? 0 : size;
  item.barrier();

  // find first negative
  for (auto i = first; i < size - 1; i += item.get_local_range().get(2)) {
    if ((a[ind[i]] ^ a[ind[i + 1]]) < 0)
      *firstNeg = i + 1;
  }

  item.barrier();

  auto ii = first;
  for (auto i = *firstNeg + item.get_local_id(2); i < size; i += item.get_local_range().get(2)) {
    ind2[ii] = ind[i];
    ii += item.get_local_range().get(2);
  }
  item.barrier();
  ii = size - *firstNeg + item.get_local_id(2);
  assert(ii >= 0);
  for (auto i = first; i < *firstNeg; i += item.get_local_range().get(2)) {
    ind2[ii] = ind[i];
    ii += item.get_local_range().get(2);
  }
  item.barrier();
  for (auto i = first; i < size; i += item.get_local_range().get(2))
    ind[i] = ind2[i];
}

template <typename T>
inline void reorderFloat(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size, sycl::nd_item<3> item, uint32_t * firstNeg) {
  //move negative first...

  int32_t first = item.get_local_id(2);
  *firstNeg = a[ind[0]] < 0 ? 0 : size;
  item.barrier();

  // find first negative
  for (auto i = first; i < size - 1; i += item.get_local_range().get(2)) {
    if ((a[ind[i]] ^ a[ind[i + 1]]) < 0)
      *firstNeg = i + 1;
  }

  item.barrier();

  int ii = size - *firstNeg - item.get_local_id(2) - 1;
  for (auto i = *firstNeg + item.get_local_id(2); i < size; i += item.get_local_range().get(2)) {
    ind2[ii] = ind[i];
    ii -= item.get_local_range().get(2);
  }
  item.barrier();
  ii = size - *firstNeg + item.get_local_id(2);
  assert(ii >= 0);
  for (auto i = first; i < *firstNeg; i += item.get_local_range().get(2)) {
    ind2[ii] = ind[i];
    ii += item.get_local_range().get(2);
  }
  item.barrier();
  for (auto i = first; i < size; i += item.get_local_range().get(2))
    ind[i] = ind2[i];
}

template <typename T,  // shall be interger
          int NS,      // number of significant bytes to use in sorting
          typename RF>
__forceinline void radixSortImpl(
    T const* __restrict__ a, uint16_t* ind, uint16_t* ind2, uint32_t size, RF reorder, //check how many args reorder wants
    sycl::nd_item<3> item, int32_t* c, int32_t* ct, int32_t* cu, int* ibs, int* p) {
  constexpr int d = 8, w = 8 * sizeof(T);
  constexpr int sb = 1 << d;
  constexpr int ps = int(sizeof(T)) - NS;

  assert(size > 0);
  assert(item.get_local_range().get(2) >= sb);

  // bool debug = false; // item.get_local_id(2)==0 && blockIdx.x==5;

  *p = ps;

  auto j = ind;
  auto k = ind2;

  int32_t first = item.get_local_id(2);
  for (auto i = first; i < size; i += item.get_local_range().get(2))
    j[i] = i;
  item.barrier();

  while ((item.barrier(), sycl::all_of_group(item.get_group(), p < w / d))) {
    if (item.get_local_id(2) < sb)
      c[item.get_local_id(2)] = 0;
    item.barrier();

    // fill bins
    for (auto i = first; i < size; i += item.get_local_range().get(2)) {
      auto bin = (a[j[i]] >> d * p) & (sb - 1);
      cms::sycltools::AtomicAdd(&c[bin], 1);
    }
    item.barrier();

    // prefix scan "optimized"???...
    if (item.get_local_id(2) < sb) {
      auto x = c[item.get_local_id(2)];
      auto laneId = item.get_local_id(2) & 0x1f;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        //sycl::shift_group_right
        auto y = sycl::shift_sub_group_right(0xffffffff, x, offset);
        if (laneId >= offset)
          x += y;
      }
      ct[item.get_local_id(2)] = x;
    }
    item.barrier();
    if (item.get_local_id(2) < sb) {
      auto ss = (item.get_local_id(2) / 32) * 32 - 1;
      c[item.get_local_id(2)] = ct[item.get_local_id(2)];
      for (int i = ss; i > 0; i -= 32)
        c[item.get_local_id(2)] += ct[i];
    }
    /* 
    //prefix scan for the nulls  (for documentation)
    if (item.get_local_id(2)==0)
      for (int i = 1; i < sb; ++i) c[i] += c[i-1];
    */

    // broadcast
    *ibs = size - 1;
    item.barrier();
    while ((item.barrier(), sycl::all_of_group(item.get_group(), ibs > 0))) {
      int i = *ibs - item.get_local_id(2);
      if (item.get_local_id(2) < sb) {
        cu[item.get_local_id(2)] = -1;
        ct[item.get_local_id(2)] = -1;
      }
      item.barrier();
      int32_t bin = -1;
      if (item.get_local_id(2) < sb) {
        if (i >= 0) {
          bin = (a[j[i]] >> d * *p) & (sb - 1);
          ct[item.get_local_id(2)] = bin;
          cms::sycltools::AtomicMax(&cu[bin], int(i));
        }
      }
      item.barrier();
      if (item.get_local_id(2) < sb) {
        if (i >= 0 && i == cu[bin])  // ensure to keep them in order
          for (int ii = item.get_local_id(2); ii < sb; ++ii)
            if (ct[ii] == bin) {
              auto oi = ii - item.get_local_id(2);
              // assert(i>=oi);if(i>=oi)
              k[--c[bin]] = j[i - oi];
            }
      }
      item.barrier();
      if (bin >= 0)
        assert(c[bin] >= 0);
      if (item.get_local_id(2) == 0)
        ibs -= sb;
      item.barrier();
    }

    /*
    // broadcast for the nulls  (for documentation)
    if (item.get_local_id(2)==0)
    for (int i=size-first-1; i>=0; i--) { // =item.get_local_range().get(2)) {
      auto bin = (a[j[i]] >> d*p)&(sb-1);
      auto ik = atomicSub(&c[bin],1);
      k[ik-1] = j[i];
    }
    */

    item.barrier();
    assert(c[0] == 0);

    // swap j and k (local, ok)
    auto t = j;
    j = k;
    k = t;

    if (item.get_local_id(2) == 0)
      ++(*p);
    item.barrier();
  }

  if ((w != 8) && (0 == (NS & 1)))
    assert(j == ind);  // w/d is even so ind is correct

  if (j != ind)  // odd...
    for (auto i = first; i < size; i += item.get_local_range().get(2))
      ind[i] = ind2[i];

  item.barrier();

  // now move negative first... (if signed)
  reorder(a, ind, ind2, size);
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_unsigned<T>::value, T>::type* = nullptr>
__forceinline void radixSort(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size, 
sycl::nd_item<3> item, int32_t *c, int32_t *ct, int32_t *cu, int *ibs, int *p) {
  radixSortImpl<T, NS>(a, ind, ind2, size, dummyReorder<T>, item, c, ct, cu, ibs, p);
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type* = nullptr>
__forceinline void radixSort(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size, 
sycl::nd_item<3> item, int32_t *c, int32_t *ct, int32_t *cu, int *ibs, int *p) {
  radixSortImpl<T, NS>(a, ind, ind2, size, reorderSigned<T>, item, c, ct, cu, ibs, p);
}

template <typename T,
          int NS = sizeof(T),  // number of significant bytes to use in sorting
          typename std::enable_if<std::is_floating_point<T>::value, T>::type* = nullptr>
__forceinline void radixSort(T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size, 
sycl::nd_item<3> item, int32_t *c, int32_t *ct, int32_t *cu, int *ibs, int *p) {
  using I = int;
  radixSortImpl<I, NS>((I const*)(a), ind, ind2, size, reorderFloat<I>, item, c, ct, cu, ibs, p);
}

template <typename T, int NS = sizeof(T)>
__forceinline void radixSortMulti(T const* v,
                                  uint16_t* index,
                                  uint32_t const* offsets,
                                  uint16_t* workspace,
                                  sycl::nd_item<3> item,
                                  uint8_t *ws_local,
                                  int32_t *c,
                                  int32_t *ct,
                                  int32_t *cu,
                                  int *ibs,
                                  int *p) {
  auto ws = (uint16_t *)dpct_local;

  auto a = v + offsets[blockIdx.x];
  auto ind = index + offsets[blockIdx.x];
  auto ind2 = nullptr == workspace ? ws : workspace + offsets[blockIdx.x];
  auto size = offsets[blockIdx.x + 1] - offsets[blockIdx.x];
  assert(offsets[blockIdx.x + 1] >= offsets[blockIdx.x]);
  if (size > 0)
    radixSort<T, NS>(a, ind, ind2, size, item, c, ct, cu, ibs, p);
}

namespace cms {
  namespace sycltools {

    template <typename T, int NS = sizeof(T)>
    void radixSortMultiWrapper(T const* v, uint16_t* index, uint32_t const* offsets, uint16_t* workspace,
        sycl::nd_item<3> item, uint8_t *ws_local, int32_t *c, int32_t *ct, int32_t *cu, int *ibs, int *p) {
      radixSortMulti<T, NS>(v, index, offsets, workspace, item, ws_local, c, ct, cu, ibs, p);
    }

    template <typename T, int NS = sizeof(T)>
    void radixSortMultiWrapper2(T const* v, uint16_t* index, uint32_t const* offsets, uint16_t* workspace,
         sycl::nd_item<3> item, uint8_t *ws_local, int32_t *c, int32_t *ct, int32_t *cu, int *ibs, int *p) {
      radixSortMulti<T, NS>(v, index, offsets, workspace, item, ws_local, c, ct, cu, ibs, p);
    }

  }  // namespace sycltools
}  // namespace cms

#endif  // SYCL_LANGUAGE_VERSION

#endif  // HeterogeneousCoreSYCLUtilities_radixSort_H
