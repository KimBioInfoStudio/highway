// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// SIMD-accelerated Adler-32 checksum, compatible with zlib's adler32().
//
// Adler-32 splits into two 16-bit sums modulo 65521 (largest prime < 2^16):
//   s1 = 1 + sum(data[i])
//   s2 = sum(s1 after each byte)
//   result = (s2 << 16) | s1
//
// For n bytes, s2 = s2_init + n*s1_init + sum_{k=0}^{n-1} (n-k)*data[k].
//
// The SIMD approach processes C chunks of N bytes each:
//   s1 += ReduceSum(vs1)            [vectorized byte sums]
//   s2 += C*N*s1_init               [scalar s1 contribution]
//      +  N * ReduceSum(vs2)        [inter-chunk weighting via "vs2 += vs1"]
//      +  ReduceSum(vintra)         [intra-chunk position weights N..1]
//
// Modular reduction is applied every NMAX=5552 bytes (the largest value
// for which the u32 accumulators provably do not overflow).

// clang-format off
#if defined(HIGHWAY_HWY_CONTRIB_CHECKSUM_ADLER32_INL_H_) == defined(HWY_TARGET_TOGGLE)  // NOLINT
// clang-format on
#ifdef HIGHWAY_HWY_CONTRIB_CHECKSUM_ADLER32_INL_H_
#undef HIGHWAY_HWY_CONTRIB_CHECKSUM_ADLER32_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_CHECKSUM_ADLER32_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace detail {

// Largest block size before u32 overflow: 255*5552*(5552+1)/2 + 5552*65520
// = 4,294,839,440 < 2^32 - 1.
static constexpr size_t kAdler32Nmax = 5552;
static constexpr uint32_t kAdler32Mod = 65521;

static HWY_INLINE uint32_t Adler32Scalar(uint32_t s1, uint32_t s2,
                                          const uint8_t* HWY_RESTRICT data,
                                          size_t size) {
  size_t offset = 0;
  while (offset < size) {
    const size_t block = HWY_MIN(size - offset, kAdler32Nmax);
    for (size_t i = 0; i < block; ++i) {
      s1 += data[offset + i];
      s2 += s1;
    }
    s1 %= kAdler32Mod;
    s2 %= kAdler32Mod;
    offset += block;
  }
  return (s2 << 16) | s1;
}

}  // namespace detail

// Computes the Adler-32 checksum of data[0..size).
// `adler` is the running checksum (1 for the first call).
static HWY_NOINLINE uint32_t Adler32(const uint8_t* HWY_RESTRICT data,
                                     size_t size, uint32_t adler = 1) {
  uint32_t s1 = adler & 0xFFFF;
  uint32_t s2 = (adler >> 16) & 0xFFFF;

  if (size == 0) return (s2 << 16) | s1;

#if HWY_TARGET == HWY_SCALAR
  return detail::Adler32Scalar(s1, s2, data, size);
#else
  const ScalableTag<uint8_t> du8;
  const ScalableTag<uint16_t> du16;
  const ScalableTag<uint32_t> du32;
  const size_t N8 = Lanes(du8);
  const size_t N16 = Lanes(du16);  // == N8 / 2

  // Weight vectors for intra-chunk position weighting.
  // Low half positions [0, N16): weight = N8, N8-1, ..., N8-N16+1
  const auto vweight_lo =
      Sub(Set(du16, static_cast<uint16_t>(N8)), Iota(du16, 0));
  // High half positions [N16, N8): weight = N16, N16-1, ..., 1
  const auto vweight_hi =
      Sub(Set(du16, static_cast<uint16_t>(N16)), Iota(du16, 0));

  size_t offset = 0;
  while (offset < size) {
    const size_t block = HWY_MIN(size - offset, detail::kAdler32Nmax);
    const uint8_t* HWY_RESTRICT block_data = data + offset;

    if (block < N8) {
      // Block too small for vectors; use scalar.
      const uint32_t result =
          detail::Adler32Scalar(s1, s2, block_data, block);
      s1 = result & 0xFFFF;
      s2 = (result >> 16) & 0xFFFF;
      offset += block;
      continue;
    }

    // vs1 accumulates byte sums per lane across chunks.
    // vs2 accumulates running prefix sums of vs1 (via "vs2 += vs1").
    // vintra accumulates intra-chunk position-weighted sums.
    auto vs1 = Zero(du32);
    auto vs2 = Zero(du32);
    auto vintra = Zero(du32);
    size_t chunk_count = 0;
    size_t i = 0;

    while (i + N8 <= block) {
      // Inter-chunk weighting: vs2 accumulates vs1 *before* adding new bytes.
      vs2 = Add(vs2, vs1);

      const auto vbytes = LoadU(du8, block_data + i);

      // Widen u8 -> u16 (two halves).
      const auto vlo16 = PromoteLowerTo(du16, vbytes);
      const auto vhi16 = PromoteUpperTo(du16, vbytes);

      // s1: sum of bytes, widened u16 -> u32 via pairwise sums.
      vs1 = Add(vs1, Add(SumsOf2(vlo16), SumsOf2(vhi16)));

      // Intra-chunk weighting: dot(bytes, [N8, N8-1, ..., 1]).
      // Products fit u16: max 255 * N8 <= 255 * 64 = 16320 < 65536.
      vintra = Add(vintra, Add(SumsOf2(Mul(vlo16, vweight_lo)),
                               SumsOf2(Mul(vhi16, vweight_hi))));

      i += N8;
      chunk_count++;
    }

    // Reduce vectors to scalars.
    // s2 = s2_init + C*N8*s1_init + N8*ReduceSum(vs2) + ReduceSum(vintra)
    s2 += static_cast<uint32_t>(chunk_count * N8) * s1 +
          static_cast<uint32_t>(N8) * ReduceSum(du32, vs2) +
          ReduceSum(du32, vintra);
    s1 += ReduceSum(du32, vs1);

    // Scalar tail for remaining bytes in this block.
    for (; i < block; ++i) {
      s1 += block_data[i];
      s2 += s1;
    }

    s1 %= detail::kAdler32Mod;
    s2 %= detail::kAdler32Mod;
    offset += block;
  }

  return (s2 << 16) | s1;
#endif  // HWY_TARGET == HWY_SCALAR
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_CHECKSUM_ADLER32_INL_H_
